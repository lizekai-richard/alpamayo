# Copyright (c) 2025, Haisheng Chen.

import torch
from . import _C


@torch.library.register_fake("rotation::rotate")
def _fake_kernel(x, idx_ij, theta, scales, group_size):
    return torch.empty_like(x)


@torch.library.register_fake("rotation::cute_rotate")
def _fake_cute_kernel(x, rotate_params, scales, krot):
    return torch.empty_like(x)


@torch.library.register_fake("rotation::rotate_and_quant")
def _fake_fused_kernel(x, idx_ij, theta, scales, sf_scale, group_size):
    # Calculate output dimensions for quantized data
    h = x.size(-1)
    seq_len = x.numel() // h
    quant_h = (h + 7) // 8  # 8 elements per uint32_t
    sf_h = (h + 63) // 64  # 64 elements per SF
    sf_s = ((seq_len + 127) // 128) * 128

    quant_out = torch.empty((seq_len, quant_h), dtype=torch.uint32, device=x.device)
    sf_out = torch.empty((sf_s, sf_h), dtype=torch.uint32, device=x.device)
    return quant_out, sf_out


from .interface import (
    RotateTensorFunc,
    scaled_pairwise_rotation,
)

__all__ = [
    "scaled_pairwise_rotation",
    "RotateTensorFunc",
]
