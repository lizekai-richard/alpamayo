#!/usr/bin/env python3
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
import os
import sys
from typing import NamedTuple

# inference_engine lives in the paroquant repo root, not in the installed paroquant package.
# Add paroquant repo root to path so "from inference_engine ..." works.
_paroquant_root = os.environ.get("PAROQUANT_ROOT")
if _paroquant_root and _paroquant_root not in sys.path:
    sys.path.insert(0, _paroquant_root)

import torch
import torch.nn as nn
from inference_engine.model_executor.modules.rotation_linear import RotateLinearInt4
from inference_engine.model_executor.modules.qmodule import WQLinear
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
    """Wraps RotateLinearInt4 to accept bfloat16 input.

    The rotation CUDA kernel only supports float16/float32, but the model
    runs in bfloat16. This wrapper casts bf16→fp16 on input and fp16→bf16
    on output, keeping the rest of the model numerically stable.
    """

    def __init__(self, rotate_linear: nn.Module):
        super().__init__()
        self.rotate_linear = rotate_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        # Disable autocast so internal ops stay in float16 (rotation kernel
        # only supports float16/float32; autocast would re-cast to bfloat16).
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
    """Return the transformer layer list for the given component."""
    if target == "expert":
        return model.expert.layers
    return model.vlm.model.language_model.layers


def _iter_linears(module, prefix=""):
    """Yield (full_name, parent_module, child_name, nn.Linear) for all linears."""
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            yield full, module, name, child
        yield from _iter_linears(child, full)

# ────────────────────────────────────────────────────────────
# 1. WQLinear int16 → raw int4
# ────────────────────────────────────────────────────────────

def unpack_wqlinear_qweight(
    qweight: torch.Tensor,
    out_features: int,
    in_features: int,
    interleave: int = 4,
    kstride: int = 64,
) -> torch.Tensor:
    """Reverse WQLinear's pack_intweight to recover per-element int4 values.

    pack_intweight applies three transforms in order:
        A) K-dim shuffle-1: reshape(N, K//32, 4,4,2).transpose(0,1,3,2,4)
        B) K-dim shuffle-2: reshape(N, K//32, 4,4,2).transpose(0,1,2,4,3)
        C) Row interleave:  reshape(N//4, 4, K//64, 64).transpose(0,2,1,3)
           then flat-reinterpret to (..., 64, 4) and pack 4 nibbles → int16.

    This function reverses C → B → A exactly.

    Returns
    -------
    torch.Tensor, shape (out_features, in_features), dtype int32, values in [0, 15].
    """
    N, K = out_features, in_features

    # --- undo nibble packing ---
    pw = qweight.view(torch.uint16).to(torch.int32)
    v0 = pw & 0xF
    v1 = (pw >> 4) & 0xF
    v2 = (pw >> 8) & 0xF
    v3 = (pw >> 12) & 0xF
    x = torch.stack([v0, v1, v2, v3], dim=-1)  # (N//4, K, 4)

    # --- undo Step C ---
    # nibbles → (N//4, K//64, 64, 4) matches the flat-reinterpret output
    x = x.reshape(N // interleave, K // kstride, kstride, interleave)
    # reverse flat-reinterpret: (kstride, interleave) → (interleave, kstride)
    x = x.reshape(N // interleave, K // kstride, interleave, kstride)
    # reverse transpose(0,2,1,3)
    x = x.permute(0, 2, 1, 3).reshape(N, K)

    # --- undo Step B ---
    x = x.reshape(N, K // 32, 4, 2, 4)
    x = x.permute(0, 1, 2, 4, 3).reshape(N, K)

    # --- undo Step A ---
    x = x.reshape(N, K // 32, 4, 4, 2)
    x = x.permute(0, 1, 3, 2, 4).reshape(N, K)

    return x


# ────────────────────────────────────────────────────────────
# 2. AWQ int32 packing  (8 × int4 per int32)
# ────────────────────────────────────────────────────────────

_AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def pack_awq_int32(values: torch.Tensor) -> torch.Tensor:
    """Pack (rows, cols) int4 values into AWQ int32 format with interleave.

    Returns (rows, cols // 8), dtype int32.
    """
    rows, cols = values.shape
    assert cols % 8 == 0, f"cols={cols} must be divisible by 8"
    flat = values.to(torch.int32).reshape(rows, cols // 8, 8)
    reordered = flat[:, :, _AWQ_ORDER]
    packed = torch.zeros(rows, cols // 8, dtype=torch.int32, device=values.device)
    for i in range(8):
        packed |= (reordered[:, :, i] & 0xF) << (4 * i)
    return packed


# ────────────────────────────────────────────────────────────
# 3. WQLinear → AWQ format conversion
# ────────────────────────────────────────────────────────────

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
    """Convert WQLinear buffers to standard AWQ int32 format."""
    N, K = out_features, in_features
    num_groups = K // group_size

    # 1) unpack int4
    int4_NK = unpack_wqlinear_qweight(qweight_i16, N, K)

    # 2) AWQ qweight: transpose to (K, N) then pack
    awq_qweight = pack_awq_int32(int4_NK.t().contiguous())

    # 3) scales — trim padding rows if needed
    awq_scales = scales[:num_groups, :N].contiguous().to(torch.float16)

    # 4) recover zeros:  WQLinear stores  scaled_zeros = -(scales * zeros)
    safe_sc = awq_scales.float().clone()
    safe_sc[safe_sc.abs() < 1e-10] = 1.0
    sz = scaled_zeros[:num_groups, :N].float()
    zeros = (-sz / safe_sc).round().clamp(0, 15).to(torch.int32)

    # 5) pack zeros
    awq_qzeros = pack_awq_int32(zeros)

    return AWQBuffers(awq_qweight, awq_scales, awq_qzeros, int4_NK, zeros)


# ────────────────────────────────────────────────────────────
# 4. Dequantisation helpers (for verification)
# ────────────────────────────────────────────────────────────

def dequant_wqlinear(int4_NK, scales, scaled_zeros, group_size):
    """WQLinear dequant: w_fp = int4 * scale + scaled_zero."""
    N, K = int4_NK.shape
    num_groups = K // group_size
    sc = scales[:num_groups, :N].float()        # (G, N)
    sz = scaled_zeros[:num_groups, :N].float()  # (G, N)
    sc_exp = sc.repeat_interleave(group_size, dim=0).t()   # (N, K)
    sz_exp = sz.repeat_interleave(group_size, dim=0).t()   # (N, K)
    return int4_NK.float() * sc_exp + sz_exp


def dequant_awq(int4_NK, zeros, scales, group_size):
    """AWQ dequant: w_fp = (int4 - zero) * scale."""
    N, K = int4_NK.shape
    num_groups = K // group_size
    sc = scales[:num_groups, :N].float()
    zp = zeros[:num_groups, :N].float()
    sc_exp = sc.repeat_interleave(group_size, dim=0).t()
    zp_exp = zp.repeat_interleave(group_size, dim=0).t()
    return (int4_NK.float() - zp_exp) * sc_exp


# ────────────────────────────────────────────────────────────
# 5. Marlin W4A8 Linear Module
# ────────────────────────────────────────────────────────────

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

        # Placeholder buffers — filled by load_from_awq().
        # Registering as buffers ensures torch._dynamo / CUDAGraph tracks
        # their addresses as static, preventing segfaults on graph replay.
        self.register_buffer("qweight", torch.empty(0, dtype=torch.int32))
        self.register_buffer("scales", torch.empty(0, dtype=torch.float16))
        self.register_buffer("qzeros", torch.empty(0, dtype=torch.int32))
        self.register_buffer("workspace", torch.empty(0, dtype=torch.int32))
        self.register_buffer("g_idx", torch.empty(0, dtype=torch.int32))
        self.register_buffer("g_idx_sort_indices", torch.empty(0, dtype=torch.int32))
        self.register_buffer("input_global_scale", torch.ones(1, dtype=torch.float32))
        self.bias = None

    def load_from_awq(self, awq_qweight, awq_scales, awq_qzeros, bias=None):
        """AWQ int32 buffers → Marlin W4A8 format (on same device as inputs)."""

        dev = awq_qweight.device
        K, N = self.in_features, self.out_features

        self.workspace = marlin_make_workspace_new(dev)
        self.g_idx = marlin_make_empty_g_idx(dev)
        self.g_idx_sort_indices = marlin_make_empty_g_idx(dev)

        # repack weights with 8-bit-activation tile layout
        self.qweight = ops.awq_marlin_repack(
            awq_qweight, size_k=K, size_n=N, num_bits=4, is_a_8bit=True,
        )

        # permute scales for Marlin (uses scale_perm_single for 8-bit act)
        marlin_scales = marlin_permute_scales(
            awq_scales, size_k=K, size_n=N,
            group_size=self.group_size, is_a_8bit=True,
        )

        # quantise FP16 scales → INT16 + extract global factor
        num_groups = K // self.group_size
        if num_groups > 1:
            marlin_scales, global_s = marlin_act_int8_process_scales(marlin_scales)
        else:
            global_s = torch.ones(1, dtype=torch.float32, device=dev)
        self.scales = marlin_scales
        self.input_global_scale = global_s

        # convert zero points
        self.qzeros = awq_to_marlin_zero_points(
            awq_qzeros, size_k=num_groups, size_n=N, num_bits=4, is_a_8bit=True,
        )

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

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


# ────────────────────────────────────────────────────────────
# 6. Convenience: convert one WQLinear layer end-to-end
# ────────────────────────────────────────────────────────────

def convert_wqlinear_layer(qweight_i16, scales, scaled_zeros,
                           out_features, in_features, group_size,
                           bias=None, device="cuda"):
    """WQLinear buffers → MarlinW4A8Linear on *device*, ready for inference."""
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
    """Wrap all RotateLinearInt4 modules with BF16RotateLinearWrapper."""
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
    """Replace nn.Linear modules in *target* layers with empty RotateLinearInt4 shells.

    The shells are populated later by ``model.load_state_dict()``.
    """
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
    # Conservative limit — after scaling, the product with activations
    # (typically < 100) must stay within fp16 range.
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

    if n_fixed:
        logger.warning(f"Sanitised channel_scales in {n_fixed} RotateLinearInt4 modules")
    else:
        logger.info("All channel_scales are within safe fp16 range")


def load_paroquant_model(
    model_path: str,
    paro_checkpoint: str,
    *,
    mode: str = "streaming",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda"
):
    """Load AlpamayoR1 with ParoQuant quantization (no fusion).

    The model runs in bfloat16 for numerical stability. RotateLinearInt4
    modules are wrapped to cast bf16→fp16 internally (rotation kernel
    only supports float16).

    Args:
        model_path: Path to base AlpamayoR1 pretrained model.
        paro_checkpoint: Path to ParoQuant ``.pt`` state dict checkpoint.
        mode: Attention mode — ``"streaming"`` or ``"non-streaming"``.
        quantize_expert: If True, also replace expert linears with RotateLinearInt4.
        dtype: Model dtype (default bfloat16).
        use_w4a8: If True, convert WQLinear to QServeW4A8Linear
                  (INT4×INT8 GEMM with per-group weight quant + per-token act quant).

    Returns:
        AlpamayoR1 with ParoQuant INT4 weights on CUDA, eval mode.
    """
    from alpamayo_r1.models.alpamayo_r1_unified import AlpamayoR1FlashDrive

    model = AlpamayoR1FlashDrive.from_pretrained(model_path, dtype=dtype)
    model.setup_patch_for_torch_compile(
        torch_compile="max-autotune",
        mode=mode,
        fuse_qkv=False,
        fuse_gate_up=False,
    )

    replace_linears_with_rotate_linear(model, target="vlm", init_only=True)
    paro_sd = torch.load(paro_checkpoint, map_location="cpu", weights_only=True)

    # Diagnostic: compare keys
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(paro_sd.keys())
    logger.info(f"Model has {len(model_keys)} keys, checkpoint has {len(ckpt_keys)} keys")
    logger.info(f"Overlap: {len(model_keys & ckpt_keys)} keys")

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

    # Diagnostic: check for zero scales (indicates weights not loaded)
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
    logger.info(f"Converted {n} WQLinear -> MarlinW4A8Linear (Marlin W4A8)")
    return n