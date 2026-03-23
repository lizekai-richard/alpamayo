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
from typing import NamedTuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

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

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        """Override to allow loading Marlin buffers whose shapes differ from
        the empty(0) placeholders created in __init__."""
        # Directly assign buffers, bypassing shape checks.
        # Pop handled keys and temporarily hide the buffer names from
        # self._buffers so that super()'s default logic doesn't re-check
        # them and report false missing keys.
        handled = {}
        for name in list(self._buffers.keys()):
            key = prefix + name
            if key in state_dict:
                self._buffers[name] = state_dict.pop(key)
                handled[name] = self._buffers.pop(name)
            elif strict:
                missing_keys.append(key)
        # Let nn.Module handle any remaining (e.g. bias)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata,
            strict, missing_keys, unexpected_keys, error_msgs,
        )
        # Restore handled buffers
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


# ────────────────────────────────────────────────────────────
# 7. Save / Load pretrained W4A8 model
# ────────────────────────────────────────────────────────────

def save_w4a8_pretrained(model: nn.Module, save_path: str) -> None:
    """Save a fully converted W4A8 model via HuggingFace save_pretrained.

    Saves config.json + sharded safetensors containing Marlin-format
    weights, rotation parameters, and (optionally fused) expert weights.
    """
    import json as _json
    from pathlib import Path

    save_path = str(save_path)
    model.save_pretrained(save_path)

    # Write a small marker so load knows this is W4A8 Marlin format
    meta = {
        "format": "paroquant_marlin_w4a8",
        "num_quantized_layers": sum(
            1 for _, m in model.named_modules()
            if type(m).__name__ == "MarlinW4A8Linear"
        ),
    }
    Path(save_path, "w4a8_config.json").write_text(_json.dumps(meta, indent=2))
    log.info(f"Saved W4A8 model to {save_path}")


def load_w4a8_pretrained(
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

    Args:
        save_path: Directory written by ``save_w4a8_pretrained``.
        base_model_path: Path to the original AlpamayoR1 pretrained model.
            Only the *architecture* (config + module layout) is used;
            the base weights are discarded and replaced by the saved W4A8
            weights.
        mode: ``"streaming"`` or ``"non-streaming"``.
        device: Target device.
        dtype: Model dtype (default bfloat16).
    """
    import sys
    from pathlib import Path

    # ensure paroquant is importable
    _pq = str(Path(__file__).resolve().parents[4] / "paroquant")
    if _pq not in sys.path:
        sys.path.insert(0, _pq)

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.models.patches import patch_for_torch_compile
    from alpamayo_r1.models.paroquant_loading import (
        replace_linears_with_rotate_linear,
        _wrap_rotate_linears,
    )
    from inference_engine.model_executor.modules.rotation_linear import RotateLinearInt4
    from inference_engine.model_executor.modules.qmodule import WQLinear
    import paroquant_kernels as _pq_kernels  # noqa: F401 — registers rotation ops

    # ── 1. Create model architecture from base (weights will be discarded) ──
    log.info(f"Creating model architecture from {base_model_path}...")
    model = AlpamayoR1.from_pretrained(base_model_path, dtype=dtype)

    # ── 2. Patch for torch.compile (no fusion — VLM has rotation) ──
    patch_for_torch_compile(model, mode=mode, fuse_qkv=False, fuse_gate_up=False)

    # ── 3. Replace VLM linears with RotateLinearInt4 shells ──
    log.info("Creating RotateLinearInt4 shells...")
    replace_linears_with_rotate_linear(model, target="vlm", init_only=True)

    # ── 4. Replace WQLinear inside each RotateLinearInt4 with MarlinW4A8Linear shells ──
    n_marlin = 0
    for _, mod in model.named_modules():
        if isinstance(mod, RotateLinearInt4) and isinstance(mod.qlinear, WQLinear):
            wq = mod.qlinear
            mod.qlinear = MarlinW4A8Linear(wq.in_features, wq.out_features, wq.group_size)
            n_marlin += 1
    log.info(f"Created {n_marlin} MarlinW4A8Linear shells")

    # ── 5. Wrap RotateLinearInt4 with BF16 casting ──
    _wrap_rotate_linears(model, target="vlm")
    model._patched_for_compile = True

    # ── 6. Load saved state_dict ──
    log.info(f"Loading weights from {save_path}...")
    save_dir = Path(save_path)
    if not save_dir.is_dir():
        # save_path is a HuggingFace Hub repo ID — download to local cache
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
        # Filter out expected missing (e.g. visual model internals)
        quant_missing = [k for k in missing if "qlinear" in k or "rotation" in k]
        if quant_missing:
            log.warning(f"Missing quantization keys ({len(quant_missing)}): {quant_missing[:5]}")
        else:
            log.info(f"Missing keys ({len(missing)}, all non-quant — OK)")
    if unexpected:
        log.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    del sd

    log.info(f"Moving model to {device}...")
    model = model.to(device)
    model.eval()
    log.info("W4A8 model loaded successfully")
    return model

