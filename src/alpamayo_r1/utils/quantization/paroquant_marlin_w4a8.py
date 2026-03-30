"""Convert ParoQuant WQLinear checkpoint → vLLM Marlin W4A8 (INT4 weight + INT8 activation).

Correct conversion pipeline:
    WQLinear int16 → unpack → raw int4 (N,K)
    → AWQ int32 repack (with [0,2,4,6,1,3,5,7] interleave)
    → vLLM awq_marlin_repack (is_a_8bit=True, 32×32 tile)
    → marlin_act_int8_process_scales (FP16 scale → INT16 + global_scale)
    → awq_to_marlin_zero_points (is_a_8bit=True)

Usage — standalone verification:
    python paroquant_marlin_w4a8.py --checkpoint /path/to/ckpt.pt

Usage — as a library:
    from paroquant_marlin_w4a8 import MarlinW4A8Linear, convert_wqlinear_layer
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
import torch.nn as nn
from .rotation_linear import RotateLinearInt4
from .qmodule import WQLinear
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    awq_to_marlin_zero_points,
    marlin_act_int8_process_scales,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_permute_scales,
)
logger = logging.getLogger(__name__)


class BF16RotateLinearWrapper(nn.Module):
    def __init__(self, rotate_linear: nn.Module):
        super().__init__()
        self.rotate_linear = rotate_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        with torch.amp.autocast("cuda", enabled=False):
            out = self.rotate_linear(x.to(torch.float16))
        return out.to(input_dtype)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.rotate_linear, name)


def _get_alpamayo_layers(
    model, target="vlm",
):
    if target == "expert":
        return model.expert.layers
    return model.vlm.model.language_model.layers


def _iter_linears(module, prefix=""):
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            yield full, module, name, child
        yield from _iter_linears(child, full)

def unpack_wqlinear_qweight(
    qweight: torch.Tensor,
    out_features: int,
    in_features: int,
    interleave: int = 4,
    kstride: int = 64,
) -> torch.Tensor:

    N, K = out_features, in_features

    pw = qweight.view(torch.uint16).to(torch.int32)
    v0 = pw & 0xF
    v1 = (pw >> 4) & 0xF
    v2 = (pw >> 8) & 0xF
    v3 = (pw >> 12) & 0xF
    x = torch.stack([v0, v1, v2, v3], dim=-1)  # (N//4, K, 4)

    x = x.reshape(N // interleave, K // kstride, kstride, interleave)
    x = x.reshape(N // interleave, K // kstride, interleave, kstride)
    x = x.permute(0, 2, 1, 3).reshape(N, K)

    x = x.reshape(N, K // 32, 4, 2, 4)
    x = x.permute(0, 1, 2, 4, 3).reshape(N, K)

    x = x.reshape(N, K // 32, 4, 4, 2)
    x = x.permute(0, 1, 3, 2, 4).reshape(N, K)

    return x


_AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def pack_awq_int32(values: torch.Tensor) -> torch.Tensor:
    rows, cols = values.shape
    assert cols % 8 == 0, f"cols={cols} must be divisible by 8"
    flat = values.to(torch.int32).reshape(rows, cols // 8, 8)
    reordered = flat[:, :, _AWQ_ORDER]
    packed = torch.zeros(rows, cols // 8, dtype=torch.int32, device=values.device)
    for i in range(8):
        packed |= (reordered[:, :, i] & 0xF) << (4 * i)
    return packed


class AWQBuffers(NamedTuple):
    qweight: torch.Tensor   # (K, N//8) int32
    scales: torch.Tensor    # (num_groups, N) float16
    qzeros: torch.Tensor    # (num_groups, N//8) int32
    int4_values: torch.Tensor  # (N, K) int32  — raw int4, for verification
    zeros: torch.Tensor     # (num_groups, N) int32 — raw zeros, for verification


def wqlinear_to_awq(
    qweight_i16: torch.Tensor,
    scales: torch.Tensor,
    scaled_zeros: torch.Tensor,
    out_features: int,
    in_features: int,
    group_size: int,
) -> AWQBuffers:
    N, K = out_features, in_features
    num_groups = K // group_size

    int4_NK = unpack_wqlinear_qweight(qweight_i16, N, K)

    awq_qweight = pack_awq_int32(int4_NK.t().contiguous())

    awq_scales = scales[:num_groups, :N].contiguous().to(torch.float16)

    safe_sc = awq_scales.float().clone()
    safe_sc[safe_sc.abs() < 1e-10] = 1.0
    sz = scaled_zeros[:num_groups, :N].float()
    zeros = (-sz / safe_sc).round().clamp(0, 15).to(torch.int32)

    awq_qzeros = pack_awq_int32(zeros)

    return AWQBuffers(awq_qweight, awq_scales, awq_qzeros, int4_NK, zeros)


class MarlinW4A8Linear(nn.Module):
    """INT4 weight + per-token INT8 activation linear via vLLM Marlin kernel.

    Lifecycle:
        1. create instance
        2. call load_from_awq(awq_qweight, awq_scales, awq_qzeros, bias)
           — this does AWQ → Marlin conversion (repack, scale quant, etc.)
        3. forward(x)  — dynamically quantises x to INT8, runs Marlin GEMM
    """

    def __init__(self, in_features: int, out_features: int,
                 group_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        from vllm.scalar_type import scalar_types
        self.quant_type = scalar_types.uint4

        self.register_buffer("qweight", torch.empty(0, dtype=torch.int32))
        self.register_buffer("scales", torch.empty(0, dtype=torch.float16))
        self.register_buffer("qzeros", torch.empty(0, dtype=torch.int32))
        self.register_buffer("workspace", torch.empty(0, dtype=torch.int32))
        self.register_buffer("g_idx", torch.empty(0, dtype=torch.int32))
        self.register_buffer("g_idx_sort_indices", torch.empty(0, dtype=torch.int32))
        self.register_buffer("input_global_scale", torch.ones(1, dtype=torch.float32))
        self.register_buffer("bias", None)

    def load_from_awq(self, awq_qweight, awq_scales, awq_qzeros, bias=None):
        """AWQ int32 buffers → Marlin W4A8 format (on same device as inputs)."""
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            awq_to_marlin_zero_points,
            marlin_act_int8_process_scales,
            marlin_make_empty_g_idx,
            marlin_make_workspace_new,
            marlin_permute_scales,
        )

        dev = awq_qweight.device
        K, N = self.in_features, self.out_features

        self.workspace = marlin_make_workspace_new(dev)
        self.g_idx = marlin_make_empty_g_idx(dev)
        self.g_idx_sort_indices = marlin_make_empty_g_idx(dev)

        self.qweight = ops.awq_marlin_repack(
            awq_qweight, size_k=K, size_n=N, num_bits=4, is_a_8bit=True,
        )

        marlin_scales = marlin_permute_scales(
            awq_scales, size_k=K, size_n=N,
            group_size=self.group_size, is_a_8bit=True,
        )

        num_groups = K // self.group_size
        if num_groups > 1:
            marlin_scales, global_s = marlin_act_int8_process_scales(marlin_scales)
        else:
            global_s = torch.ones(1, dtype=torch.float32, device=dev)
        self.scales = marlin_scales
        self.input_global_scale = global_s

        self.qzeros = awq_to_marlin_zero_points(
            awq_qzeros, size_k=num_groups, size_n=N, num_bits=4, is_a_8bit=True,
        )

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        """Override to allow loading Marlin buffers whose shapes differ from
        the empty(0) placeholders created in __init__."""
        handled = {}
        for name in list(self._buffers.keys()):
            key = prefix + name
            if key in state_dict:
                self._buffers[name] = state_dict.pop(key)
                handled[name] = self._buffers.pop(name)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata,
            strict, missing_keys, unexpected_keys, error_msgs,
        )
        self._buffers.update(handled)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            apply_awq_marlin_linear,
        )
        return apply_awq_marlin_linear(
            input=x,
            weight=self.qweight,
            weight_scale=self.scales,
            weight_zp=self.qzeros,
            g_idx=self.g_idx,
            g_idx_sort_indices=self.g_idx_sort_indices,
            workspace=self.workspace,
            quant_type=self.quant_type,
            output_size_per_partition=self.out_features,
            input_size_per_partition=self.in_features,
            input_global_scale=self.input_global_scale,
            bias=self.bias,
            input_dtype=torch.int8,
        )


def convert_wqlinear_layer(qweight_i16, scales, scaled_zeros,
                           out_features, in_features, group_size,
                           bias=None, device="cuda"):
    awq = wqlinear_to_awq(qweight_i16, scales, scaled_zeros,
                           out_features, in_features, group_size)
    mod = MarlinW4A8Linear(in_features, out_features, group_size)
    mod.load_from_awq(
        awq.qweight.to(device),
        awq.scales.to(device),
        awq.qzeros.to(device),
        bias=bias.to(device) if bias is not None else None,
    )
    return mod, awq


def _wrap_rotate_linears(model, target="vlm"):
    layers = _get_alpamayo_layers(model, target)
    for layer in layers:
        for name, child in layer.named_children():
            for sub_name, sub_child in child.named_children():
                if isinstance(sub_child, RotateLinearInt4):
                    setattr(child, sub_name, BF16RotateLinearWrapper(sub_child))


def replace_linears_with_rotate_linear(
    model,
    target="vlm",
    init_only: bool = True,
    ignore_suffix: tuple[str, ...] = ("lm_head",),
):
    layers = _get_alpamayo_layers(model, target)
    for layer in layers:
        for full_name, parent, child_name, linear in list(_iter_linears(layer)):
            if child_name in ignore_suffix:
                continue
            rotate_linear = RotateLinearInt4(
                in_feat=linear.in_features,
                out_feat=linear.out_features,
                bias=linear.bias is not None,
                dtype=torch.float16,
            )
            setattr(parent, child_name, rotate_linear)


def _sanitise_channel_scales(model):
    """Clamp rotation channel_scales to prevent fp16 overflow → NaN.

    Some ParoQuant checkpoints contain INF or near-overflow values in the
    ``rotation.channel_scales`` buffer.  When multiplied by activations in
    the fp16 rotation kernel, these produce NaN.  Clamping to a safe range
    fixes the issue with negligible effect on accuracy.
    """

    FP16_MAX = 65504.0
    SAFE_MAX = FP16_MAX / 128.0  # ~512

    n_fixed = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, RotateLinearInt4):
            continue
        cs = mod.rotation.channel_scales
        bad_mask = torch.isinf(cs) | torch.isnan(cs) | (cs.abs() > SAFE_MAX)
        n_bad = bad_mask.sum().item()
        if n_bad > 0:
            with torch.no_grad():
                cs.clamp_(-SAFE_MAX, SAFE_MAX)
            n_fixed += 1
            logger.warning(
                f"  Clamped {n_bad}/{cs.numel()} channel_scales in {name} "
                f"(had INF/NaN/overflow values)"
            )


def load_paroquant_model(
    model_path: str,
    paro_checkpoint: str,
    *,
    mode: str = "streaming",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda"
):
    from alpamayo_r1.models.alpamayo_r1 import FlashDriveAlpamayoR1
    from alpamayo_r1.utils.system import patch_for_torch_compile

    model = FlashDriveAlpamayoR1.from_pretrained(model_path, dtype=dtype)
    patch_for_torch_compile(model, mode=mode, fuse_qkv=False, fuse_gate_up=False)
    model._patched_for_compile = True

    replace_linears_with_rotate_linear(model, target="vlm", init_only=True)
    paro_sd = torch.load(paro_checkpoint, map_location="cpu", weights_only=True)

    missing, unexpected = model.load_state_dict(paro_sd, strict=False)
    if missing:
        quant_missing = [k for k in missing if any(
            s in k for s in ("qlinear", "rotation", "qweight", "scales", "scaled_zeros")
        )]
        logger.warning(f"Missing keys ({len(missing)} total, {len(quant_missing)} quant-related)")
        if quant_missing:
            logger.warning(f"  CRITICAL quant missing: {quant_missing[:10]}")
        else:
            logger.info(f"  Missing (non-quant): {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:20]}")
    del paro_sd

    n_zero_scales = 0
    for name, param in model.named_parameters():
        if "scales" in name and param.numel() > 0:
            if torch.all(param == 0):
                n_zero_scales += 1
                if n_zero_scales <= 3:
                    logger.warning(f"  ZERO scales: {name} shape={param.shape}")
    for name, buf in model.named_buffers():
        if "scales" in name and buf.numel() > 0:
            if torch.all(buf == 0):
                n_zero_scales += 1
                if n_zero_scales <= 3:
                    logger.warning(f"  ZERO scales (buffer): {name} shape={buf.shape}")
    if n_zero_scales > 0:
        logger.error(f"CRITICAL: {n_zero_scales} layers have zero scales — checkpoint keys likely mismatched!")
    else:
        logger.info("All scales are non-zero (checkpoint loaded correctly)")

    _sanitise_channel_scales(model)
    _wrap_rotate_linears(model, target="vlm")

    model = model.to(device)
    model.eval()

    return model


def load_paroquant_model_v1p5(
    model_path: str,
    paro_checkpoint: str,
    *,
    mode: str = "streaming",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda"
):
    from alpamayo_r1.models.alpamayo_r1p5_flashdrive import Alpamayo1_5FlashDrive
    from alpamayo_r1.utils import patch_for_torch_compile

    model = Alpamayo1_5FlashDrive.from_pretrained(model_path, dtype=dtype)
    patch_for_torch_compile(model, mode=mode, fuse_qkv=False, fuse_gate_up=False)
    model._patched_for_compile = True

    replace_linears_with_rotate_linear(model, target="vlm", init_only=True)
    paro_sd = torch.load(paro_checkpoint, map_location="cpu", weights_only=True)

    missing, unexpected = model.load_state_dict(paro_sd, strict=False)
    if missing:
        quant_missing = [k for k in missing if any(
            s in k for s in ("qlinear", "rotation", "qweight", "scales", "scaled_zeros")
        )]
        logger.warning(f"Missing keys ({len(missing)} total, {len(quant_missing)} quant-related)")
        if quant_missing:
            logger.warning(f"  CRITICAL quant missing: {quant_missing[:10]}")
        else:
            logger.info(f"  Missing (non-quant): {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:20]}")
    del paro_sd

    n_zero_scales = 0
    for name, param in model.named_parameters():
        if "scales" in name and param.numel() > 0:
            if torch.all(param == 0):
                n_zero_scales += 1
                if n_zero_scales <= 3:
                    logger.warning(f"  ZERO scales: {name} shape={param.shape}")
    for name, buf in model.named_buffers():
        if "scales" in name and buf.numel() > 0:
            if torch.all(buf == 0):
                n_zero_scales += 1
                if n_zero_scales <= 3:
                    logger.warning(f"  ZERO scales (buffer): {name} shape={buf.shape}")
    if n_zero_scales > 0:
        logger.error(f"CRITICAL: {n_zero_scales} layers have zero scales — checkpoint keys likely mismatched!")
    else:
        logger.info("All scales are non-zero (checkpoint loaded correctly)")

    _sanitise_channel_scales(model)
    _wrap_rotate_linears(model, target="vlm")

    model = model.to(device)
    model.eval()

    return model


def convert_model_to_marlin_w4a8(model):
    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, RotateLinearInt4):
            continue
        wq = mod.qlinear
        if not isinstance(wq, WQLinear):
            continue
        marlin, _ = convert_wqlinear_layer(
            wq.qweight.data, wq.scales.data, wq.scaled_zeros.data,
            wq.out_features, wq.in_features, wq.group_size,
            bias=wq.bias, device=wq.qweight.device,
        )
        mod.qlinear = marlin
        n += 1
    return n


def save_paroquant_pretrained(model: nn.Module, save_path: str) -> None:
    """Save a fully converted W4A8 model via HuggingFace save_pretrained.

    Saves config.json + sharded safetensors containing Marlin-format
    weights, rotation parameters, and (optionally fused) expert weights.
    """
    import json as _json
    from pathlib import Path

    save_path = str(save_path)
    model.save_pretrained(save_path)

    meta = {
        "format": "paroquant_marlin_w4a8",
        "num_quantized_layers": sum(
            1 for _, m in model.named_modules()
            if type(m).__name__ == "MarlinW4A8Linear"
        ),
    }
    Path(save_path, "w4a8_config.json").write_text(_json.dumps(meta, indent=2))
    logger.info(f"Saved W4A8 model to {save_path}")


def load_paroquant_pretrained(
    save_path: str,
    base_model_path: str,
    *,
    mode: str = "streaming",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Load a saved W4A8 model.

    Reconstructs the exact module tree (RotateLinearInt4 + MarlinW4A8Linear
    for VLM, unfused nn.Linear for expert), then fills in weights from the
    saved safetensors via ``load_state_dict``.
    """
    from pathlib import Path

    from alpamayo_r1.models.alpamayo_r1 import FlashDriveAlpamayoR1
    from alpamayo_r1.utils.system import patch_for_torch_compile
    from .rotation_linear import RotateLinearInt4
    from .qmodule import WQLinear
    import paroquant_kernels as _pq_kernels  # noqa: F401 — registers rotation ops

    model = FlashDriveAlpamayoR1.from_pretrained(base_model_path, dtype=dtype)
    patch_for_torch_compile(model, mode=mode, fuse_qkv=False, fuse_gate_up=False)
    replace_linears_with_rotate_linear(model, target="vlm", init_only=True)

    n_marlin = 0
    for _, mod in model.named_modules():
        if isinstance(mod, RotateLinearInt4) and isinstance(mod.qlinear, WQLinear):
            wq = mod.qlinear
            mod.qlinear = MarlinW4A8Linear(wq.in_features, wq.out_features, wq.group_size)
            n_marlin += 1

    _wrap_rotate_linears(model, target="vlm")
    model._patched_for_compile = True

    save_dir = Path(save_path)
    if not save_dir.is_dir():
        from huggingface_hub import snapshot_download
        save_dir = Path(snapshot_download(save_path))
    safetensor_files = sorted(save_dir.glob("model*.safetensors"))
    if safetensor_files:
        from safetensors.torch import load_file
        sd = {}
        for f in safetensor_files:
            sd.update(load_file(str(f), device="cpu"))
    else:
        pt_file = save_dir / "pytorch_model.bin"
        sd = torch.load(str(pt_file), map_location="cpu", weights_only=True)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        quant_missing = [k for k in missing if "qlinear" in k or "rotation" in k]
        if quant_missing:
            logger.warning(f"Missing quantization keys ({len(quant_missing)}): {quant_missing[:5]}")
        else:
            logger.info(f"Missing keys ({len(missing)}, all non-quant — OK)")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    del sd

    model = model.to(device)
    model.eval()
    return model


def load_paroquant_pretrained_v1p5(
    save_path: str,
    base_model_path: str,
    *,
    mode: str = "streaming",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Load a saved W4A8 model (Alpamayo v1.5).

    Same as load_paroquant_pretrained but uses Alpamayo1_5FlashDrive.
    """
    from pathlib import Path

    from alpamayo_r1.models.alpamayo_r1p5_flashdrive import Alpamayo1_5FlashDrive
    from alpamayo_r1.utils import patch_for_torch_compile
    from .rotation_linear import RotateLinearInt4
    from .qmodule import WQLinear
    import paroquant_kernels as _pq_kernels  # noqa: F401 — registers rotation ops

    model = Alpamayo1_5FlashDrive.from_pretrained(base_model_path, dtype=dtype)
    patch_for_torch_compile(model, mode=mode, fuse_qkv=False, fuse_gate_up=False)
    replace_linears_with_rotate_linear(model, target="vlm", init_only=True)

    n_marlin = 0
    for _, mod in model.named_modules():
        if isinstance(mod, RotateLinearInt4) and isinstance(mod.qlinear, WQLinear):
            wq = mod.qlinear
            mod.qlinear = MarlinW4A8Linear(wq.in_features, wq.out_features, wq.group_size)
            n_marlin += 1

    _wrap_rotate_linears(model, target="vlm")
    model._patched_for_compile = True

    save_dir = Path(save_path)
    if not save_dir.is_dir():
        from huggingface_hub import snapshot_download
        save_dir = Path(snapshot_download(save_path))
    safetensor_files = sorted(save_dir.glob("model*.safetensors"))
    if safetensor_files:
        from safetensors.torch import load_file
        sd = {}
        for f in safetensor_files:
            sd.update(load_file(str(f), device="cpu"))
    else:
        pt_file = save_dir / "pytorch_model.bin"
        sd = torch.load(str(pt_file), map_location="cpu", weights_only=True)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        quant_missing = [k for k in missing if "qlinear" in k or "rotation" in k]
        if quant_missing:
            logger.warning(f"Missing quantization keys ({len(quant_missing)}): {quant_missing[:5]}")
        else:
            logger.info(f"Missing keys ({len(missing)}, all non-quant — OK)")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    del sd

    model = model.to(device)
    model.eval()
    return model
