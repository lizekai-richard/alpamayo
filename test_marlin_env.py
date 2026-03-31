"""Environment & functional tests for vLLM Marlin W4A8 kernel on SM120 (RTX 5090).

Run:
    CUDA_VISIBLE_DEVICES=0 python tests/test_marlin_env.py

Each test prints PASS / FAIL and a one-line diagnostic.
No model checkpoint is needed — all tensors are synthetic.
"""

from __future__ import annotations

import sys
import traceback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_results: list[tuple[str, bool, str]] = []


def _run(name: str, fn):
    try:
        msg = fn()
        _results.append((name, True, msg or "ok"))
        print(f"  PASS  {name}: {msg or 'ok'}")
    except Exception as e:
        _results.append((name, False, str(e)))
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()


# ===================================================================
# 1. Basic environment
# ===================================================================
print("\n=== 1. Basic Environment ===")


def test_torch_import():
    import torch
    return f"torch {torch.__version__}, CUDA build {torch.version.cuda}"

_run("torch_import", test_torch_import)


def test_cuda_available():
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available() == False"
    return f"device_count={torch.cuda.device_count()}"

_run("cuda_available", test_cuda_available)


def test_gpu_arch():
    import torch
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    return f"{name}, sm_{cap[0]}{cap[1]}"

_run("gpu_arch", test_gpu_arch)


def test_cuda_basic_ops():
    import torch
    a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    c = a @ b
    assert c.shape == (64, 64)
    assert not torch.isnan(c).any()
    return "fp16 matmul ok"

_run("cuda_basic_ops", test_cuda_basic_ops)


def test_cuda_int8_ops():
    import torch
    a = torch.randint(-128, 127, (32, 64), device="cuda", dtype=torch.int8)
    b = torch.randint(-128, 127, (64, 32), device="cuda", dtype=torch.int8)
    c = torch._int_mm(a, b)
    assert c.shape == (32, 32)
    return "int8 matmul (_int_mm) ok"

_run("cuda_int8_ops", test_cuda_int8_ops)


# ===================================================================
# 2. vLLM imports
# ===================================================================
print("\n=== 2. vLLM Imports ===")


def test_vllm_import():
    import vllm
    return f"vllm {vllm.__version__}"

_run("vllm_import", test_vllm_import)


def test_vllm_custom_ops():
    from vllm import _custom_ops as ops
    for fn_name in ["awq_marlin_repack"]:
        assert hasattr(ops, fn_name), f"ops.{fn_name} not found"
    return "ops.awq_marlin_repack exists"

_run("vllm_custom_ops", test_vllm_custom_ops)


def test_marlin_utils_imports():
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        awq_to_marlin_zero_points,
        marlin_act_int8_process_scales,
        marlin_make_empty_g_idx,
        marlin_make_workspace_new,
        marlin_permute_scales,
        apply_awq_marlin_linear,
    )
    return "all 6 marlin_utils functions imported"

_run("marlin_utils_imports", test_marlin_utils_imports)


def test_scalar_types():
    from vllm.scalar_type import scalar_types
    qt = scalar_types.uint4
    return f"uint4 type = {qt}"

_run("scalar_types", test_scalar_types)


# ===================================================================
# 3. Marlin kernel functions — individual
# ===================================================================
print("\n=== 3. Marlin Kernel Functions (individual) ===")

# Realistic dimensions: hidden_size=2048, out_features=5504, group_size=128
K, N, GROUP_SIZE = 2048, 5504, 128
# Make N divisible by 128 for Marlin tile alignment
N = ((N + 127) // 128) * 128  # 5504 -> 5504 (already aligned? let's round)
# Actually safer to just pick clean dims
K, N, GROUP_SIZE = 2048, 5632, 128
NUM_GROUPS = K // GROUP_SIZE  # 16


def _make_synthetic_awq(K, N, group_size, device="cuda"):
    """Create synthetic AWQ-format buffers for testing."""
    import torch
    num_groups = K // group_size

    # AWQ qweight: (K, N//8) int32 — packed 8x int4
    qweight = torch.empty(K, N // 8, dtype=torch.int32, device=device).random_()
    # Scales: (num_groups, N) fp16
    scales = torch.randn(num_groups, N, dtype=torch.float16, device=device).abs() * 0.1 + 0.01
    # Zero-points: (num_groups, N//8) int32 — packed 8x int4
    qzeros = torch.randint(0, 2**16, (num_groups, N // 8),
                           dtype=torch.int32, device=device)
    return qweight, scales, qzeros


def test_marlin_make_workspace():
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )
    dev = torch.device("cuda")
    ws = marlin_make_workspace_new(dev)
    assert ws.device.type == "cuda"
    assert ws.dtype == torch.int32
    return f"workspace shape={ws.shape}, dtype={ws.dtype}"

_run("marlin_make_workspace", test_marlin_make_workspace)


def test_marlin_make_empty_g_idx():
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_empty_g_idx,
    )
    dev = torch.device("cuda")
    g_idx = marlin_make_empty_g_idx(dev)
    assert g_idx.device.type == "cuda"
    return f"g_idx shape={g_idx.shape}"

_run("marlin_make_empty_g_idx", test_marlin_make_empty_g_idx)


def test_awq_marlin_repack():
    """Test ops.awq_marlin_repack — repacks AWQ int32 layout to Marlin tile format."""
    import torch
    from vllm import _custom_ops as ops
    qweight, _, _ = _make_synthetic_awq(K, N, GROUP_SIZE)
    repacked = ops.awq_marlin_repack(qweight, size_k=K, size_n=N,
                                      num_bits=4, is_a_8bit=True)
    assert repacked.dtype == torch.int32
    assert repacked.device.type == "cuda"
    return f"input {qweight.shape} -> repacked {repacked.shape}"

_run("awq_marlin_repack", test_awq_marlin_repack)


def test_marlin_permute_scales():
    """Test marlin_permute_scales — rearranges scales for efficient Marlin access."""
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )
    scales = torch.randn(NUM_GROUPS, N, dtype=torch.float16, device="cuda").abs() * 0.1 + 0.01
    permuted = marlin_permute_scales(scales, size_k=K, size_n=N,
                                     group_size=GROUP_SIZE, is_a_8bit=True)
    assert permuted.dtype == torch.float16
    return f"input {scales.shape} -> permuted {permuted.shape}"

_run("marlin_permute_scales", test_marlin_permute_scales)


def test_marlin_act_int8_process_scales():
    """Test marlin_act_int8_process_scales — quantizes fp16 scales to int16 + global scale."""
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
        marlin_act_int8_process_scales,
    )
    scales = torch.randn(NUM_GROUPS, N, dtype=torch.float16, device="cuda").abs() * 0.1 + 0.01
    permuted = marlin_permute_scales(scales, size_k=K, size_n=N,
                                     group_size=GROUP_SIZE, is_a_8bit=True)
    int16_scales, global_s = marlin_act_int8_process_scales(permuted)
    assert int16_scales.dtype in (torch.int16, torch.float16), f"expected int16 or float16, got {int16_scales.dtype}"
    assert global_s.dtype == torch.float32, f"expected float32, got {global_s.dtype}"
    assert global_s.item() > 0, f"global_scale should be > 0, got {global_s.item()}"
    return f"scales {permuted.shape} -> int16 {int16_scales.shape}, global_s={global_s.item():.6f}"

_run("marlin_act_int8_process_scales", test_marlin_act_int8_process_scales)


def test_awq_to_marlin_zero_points():
    """Test awq_to_marlin_zero_points — converts AWQ packed zeros to Marlin format."""
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        awq_to_marlin_zero_points,
    )
    qzeros = torch.randint(0, 2**16, (NUM_GROUPS, N // 8),
                           dtype=torch.int32, device="cuda")
    marlin_zp = awq_to_marlin_zero_points(qzeros, size_k=NUM_GROUPS, size_n=N,
                                           num_bits=4, is_a_8bit=True)
    assert marlin_zp.dtype == torch.int32
    return f"input {qzeros.shape} -> marlin_zp {marlin_zp.shape}"

_run("awq_to_marlin_zero_points", test_awq_to_marlin_zero_points)


# ===================================================================
# 4. End-to-end Marlin W4A8 GEMM
# ===================================================================
print("\n=== 4. End-to-End Marlin W4A8 GEMM ===")


def _build_marlin_w4a8(K, N, group_size, device="cuda"):
    """Build a fully-initialized MarlinW4A8Linear from synthetic AWQ data."""
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        awq_to_marlin_zero_points,
        marlin_act_int8_process_scales,
        marlin_make_empty_g_idx,
        marlin_make_workspace_new,
        marlin_permute_scales,
    )
    from vllm.scalar_type import scalar_types

    num_groups = K // group_size
    qweight, scales, qzeros = _make_synthetic_awq(K, N, group_size, device)

    repacked_w = ops.awq_marlin_repack(qweight, size_k=K, size_n=N,
                                        num_bits=4, is_a_8bit=True)
    perm_scales = marlin_permute_scales(scales, size_k=K, size_n=N,
                                        group_size=group_size, is_a_8bit=True)
    int16_scales, global_s = marlin_act_int8_process_scales(perm_scales)
    marlin_zp = awq_to_marlin_zero_points(qzeros, size_k=num_groups, size_n=N,
                                           num_bits=4, is_a_8bit=True)
    workspace = marlin_make_workspace_new(torch.device(device))
    g_idx = marlin_make_empty_g_idx(torch.device(device))
    g_idx_sort = marlin_make_empty_g_idx(torch.device(device))

    return dict(
        qweight=repacked_w,
        scales=int16_scales,
        qzeros=marlin_zp,
        workspace=workspace,
        g_idx=g_idx,
        g_idx_sort_indices=g_idx_sort,
        input_global_scale=global_s,
        quant_type=scalar_types.uint4,
    )


def test_marlin_gemm_basic():
    """Full Marlin W4A8 GEMM: fp16 input -> int8 dynamic quant -> int4 weight matmul."""
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_awq_marlin_linear,
    )
    bufs = _build_marlin_w4a8(K, N, GROUP_SIZE)
    x = torch.randn(1, K, dtype=torch.float16, device="cuda")

    out = apply_awq_marlin_linear(
        input=x,
        weight=bufs["qweight"],
        weight_scale=bufs["scales"],
        weight_zp=bufs["qzeros"],
        g_idx=bufs["g_idx"],
        g_idx_sort_indices=bufs["g_idx_sort_indices"],
        workspace=bufs["workspace"],
        quant_type=bufs["quant_type"],
        output_size_per_partition=N,
        input_size_per_partition=K,
        input_global_scale=bufs["input_global_scale"],
        bias=None,
        input_dtype=torch.int8,
    )
    assert out.shape == (1, N), f"expected (1, {N}), got {out.shape}"
    assert not torch.isnan(out).any(), "output contains NaN"
    assert not torch.isinf(out).any(), "output contains Inf"
    return f"input (1, {K}) -> output {out.shape}, no NaN/Inf"

_run("marlin_gemm_basic", test_marlin_gemm_basic)


def test_marlin_gemm_batch():
    """Batched Marlin W4A8 GEMM with multiple tokens."""
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_awq_marlin_linear,
    )
    bufs = _build_marlin_w4a8(K, N, GROUP_SIZE)
    batch = 32
    x = torch.randn(batch, K, dtype=torch.float16, device="cuda")

    out = apply_awq_marlin_linear(
        input=x,
        weight=bufs["qweight"],
        weight_scale=bufs["scales"],
        weight_zp=bufs["qzeros"],
        g_idx=bufs["g_idx"],
        g_idx_sort_indices=bufs["g_idx_sort_indices"],
        workspace=bufs["workspace"],
        quant_type=bufs["quant_type"],
        output_size_per_partition=N,
        input_size_per_partition=K,
        input_global_scale=bufs["input_global_scale"],
        bias=None,
        input_dtype=torch.int8,
    )
    assert out.shape == (batch, N)
    assert not torch.isnan(out).any()
    return f"input ({batch}, {K}) -> output {out.shape}"

_run("marlin_gemm_batch", test_marlin_gemm_batch)


def test_marlin_gemm_3d():
    """3D input (batch, seq_len, hidden) — typical transformer shape."""
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_awq_marlin_linear,
    )
    bufs = _build_marlin_w4a8(K, N, GROUP_SIZE)
    B, S = 4, 16
    x = torch.randn(B, S, K, dtype=torch.float16, device="cuda")

    out = apply_awq_marlin_linear(
        input=x,
        weight=bufs["qweight"],
        weight_scale=bufs["scales"],
        weight_zp=bufs["qzeros"],
        g_idx=bufs["g_idx"],
        g_idx_sort_indices=bufs["g_idx_sort_indices"],
        workspace=bufs["workspace"],
        quant_type=bufs["quant_type"],
        output_size_per_partition=N,
        input_size_per_partition=K,
        input_global_scale=bufs["input_global_scale"],
        bias=None,
        input_dtype=torch.int8,
    )
    assert out.shape == (B, S, N), f"expected ({B}, {S}, {N}), got {out.shape}"
    assert not torch.isnan(out).any()
    return f"input ({B}, {S}, {K}) -> output {out.shape}"

_run("marlin_gemm_3d", test_marlin_gemm_3d)


def test_marlin_gemm_with_bias():
    """Marlin W4A8 GEMM with bias."""
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_awq_marlin_linear,
    )
    bufs = _build_marlin_w4a8(K, N, GROUP_SIZE)
    x = torch.randn(8, K, dtype=torch.float16, device="cuda")
    bias = torch.randn(N, dtype=torch.float16, device="cuda")

    out = apply_awq_marlin_linear(
        input=x,
        weight=bufs["qweight"],
        weight_scale=bufs["scales"],
        weight_zp=bufs["qzeros"],
        g_idx=bufs["g_idx"],
        g_idx_sort_indices=bufs["g_idx_sort_indices"],
        workspace=bufs["workspace"],
        quant_type=bufs["quant_type"],
        output_size_per_partition=N,
        input_size_per_partition=K,
        input_global_scale=bufs["input_global_scale"],
        bias=bias,
        input_dtype=torch.int8,
    )
    assert out.shape == (8, N)
    assert not torch.isnan(out).any()
    return f"with bias, output {out.shape}"

_run("marlin_gemm_with_bias", test_marlin_gemm_with_bias)


def test_marlin_gemm_bfloat16_input():
    """BF16 input — the actual dtype used in Alpamayo inference (cast to fp16 by wrapper)."""
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_awq_marlin_linear,
    )
    bufs = _build_marlin_w4a8(K, N, GROUP_SIZE)
    x_bf16 = torch.randn(8, K, dtype=torch.bfloat16, device="cuda")
    # The real code casts bf16 -> fp16 via BF16RotateLinearWrapper before Marlin
    x_fp16 = x_bf16.to(torch.float16)

    out = apply_awq_marlin_linear(
        input=x_fp16,
        weight=bufs["qweight"],
        weight_scale=bufs["scales"],
        weight_zp=bufs["qzeros"],
        g_idx=bufs["g_idx"],
        g_idx_sort_indices=bufs["g_idx_sort_indices"],
        workspace=bufs["workspace"],
        quant_type=bufs["quant_type"],
        output_size_per_partition=N,
        input_size_per_partition=K,
        input_global_scale=bufs["input_global_scale"],
        bias=None,
        input_dtype=torch.int8,
    )
    assert out.shape == (8, N)
    assert not torch.isnan(out).any()
    return f"bf16->fp16 input, output {out.shape}"

_run("marlin_gemm_bfloat16_input", test_marlin_gemm_bfloat16_input)


# ===================================================================
# 5. MarlinW4A8Linear module (from your codebase)
# ===================================================================
print("\n=== 5. MarlinW4A8Linear Module ===")


def test_marlin_w4a8_linear_module():
    """Test the full MarlinW4A8Linear class from paroquant_marlin_w4a8.py."""
    import torch
    sys.path.insert(0, "/root/alpamayo/src")
    from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import MarlinW4A8Linear

    mod = MarlinW4A8Linear(K, N, GROUP_SIZE).cuda()
    qweight, scales, qzeros = _make_synthetic_awq(K, N, GROUP_SIZE)
    mod.load_from_awq(qweight, scales, qzeros, bias=None)

    x = torch.randn(4, 16, K, dtype=torch.float16, device="cuda")
    out = mod(x)
    assert out.shape == (4, 16, N), f"expected (4, 16, {N}), got {out.shape}"
    assert not torch.isnan(out).any()
    return f"MarlinW4A8Linear forward ok, output {out.shape}"

_run("marlin_w4a8_linear_module", test_marlin_w4a8_linear_module)


def test_marlin_w4a8_linear_with_bias():
    """MarlinW4A8Linear with bias."""
    import torch
    sys.path.insert(0, "/root/alpamayo/src")
    from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import MarlinW4A8Linear

    mod = MarlinW4A8Linear(K, N, GROUP_SIZE).cuda()
    qweight, scales, qzeros = _make_synthetic_awq(K, N, GROUP_SIZE)
    bias = torch.randn(N, dtype=torch.float16, device="cuda")
    mod.load_from_awq(qweight, scales, qzeros, bias=bias)

    x = torch.randn(2, K, dtype=torch.float16, device="cuda")
    out = mod(x)
    assert out.shape == (2, N)
    assert not torch.isnan(out).any()
    return f"with bias, output {out.shape}"

_run("marlin_w4a8_linear_with_bias", test_marlin_w4a8_linear_with_bias)


# ===================================================================
# 6. WQLinear -> AWQ -> Marlin conversion pipeline
# ===================================================================
print("\n=== 6. WQLinear -> AWQ -> Marlin Conversion ===")


def test_unpack_wqlinear_qweight():
    """Test unpack_wqlinear_qweight: INT16 packed -> raw INT4."""
    import torch
    sys.path.insert(0, "/root/alpamayo/src")
    from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import unpack_wqlinear_qweight

    # WQLinear format: (N//4, K) int16
    qw = torch.randint(-32768, 32767, (N // 4, K), dtype=torch.int16, device="cuda")
    unpacked = unpack_wqlinear_qweight(qw, out_features=N, in_features=K)
    assert unpacked.shape == (N, K), f"expected ({N}, {K}), got {unpacked.shape}"
    # All values should be 0-15 (4-bit unsigned)
    assert (unpacked >= 0).all() and (unpacked <= 15).all(), "values outside [0, 15]"
    return f"unpacked {qw.shape} int16 -> {unpacked.shape} int4, range [{unpacked.min()}, {unpacked.max()}]"

_run("unpack_wqlinear_qweight", test_unpack_wqlinear_qweight)


def test_pack_awq_int32():
    """Test pack_awq_int32: raw INT4 -> AWQ packed INT32."""
    import torch
    sys.path.insert(0, "/root/alpamayo/src")
    from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import pack_awq_int32

    values = torch.randint(0, 16, (K, N), dtype=torch.int32, device="cuda")
    packed = pack_awq_int32(values)
    assert packed.shape == (K, N // 8), f"expected ({K}, {N // 8}), got {packed.shape}"
    assert packed.dtype == torch.int32
    return f"packed {values.shape} -> {packed.shape}"

_run("pack_awq_int32", test_pack_awq_int32)


def test_wqlinear_to_awq():
    """Test full WQLinear -> AWQ conversion."""
    import torch
    sys.path.insert(0, "/root/alpamayo/src")
    from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import wqlinear_to_awq

    qw_i16 = torch.randint(-32768, 32767, (N // 4, K), dtype=torch.int16, device="cuda")
    scales = torch.randn(NUM_GROUPS, N, dtype=torch.float16, device="cuda").abs() * 0.1 + 0.01
    scaled_zeros = torch.randn(NUM_GROUPS, N, dtype=torch.float16, device="cuda") * 0.01

    awq = wqlinear_to_awq(qw_i16, scales, scaled_zeros, N, K, GROUP_SIZE)
    assert awq.qweight.shape == (K, N // 8)
    assert awq.scales.shape == (NUM_GROUPS, N)
    assert awq.qzeros.shape == (NUM_GROUPS, N // 8)
    assert awq.int4_values.shape == (N, K)
    return f"qweight={awq.qweight.shape}, scales={awq.scales.shape}, qzeros={awq.qzeros.shape}"

_run("wqlinear_to_awq", test_wqlinear_to_awq)


def test_convert_wqlinear_layer():
    """Test the full convert_wqlinear_layer pipeline: WQLinear -> MarlinW4A8Linear."""
    import torch
    sys.path.insert(0, "/root/alpamayo/src")
    from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import convert_wqlinear_layer

    qw_i16 = torch.randint(-32768, 32767, (N // 4, K), dtype=torch.int16, device="cuda")
    scales = torch.randn(NUM_GROUPS, N, dtype=torch.float16, device="cuda").abs() * 0.1 + 0.01
    scaled_zeros = torch.randn(NUM_GROUPS, N, dtype=torch.float16, device="cuda") * 0.01

    mod, awq = convert_wqlinear_layer(
        qw_i16, scales, scaled_zeros, N, K, GROUP_SIZE, device="cuda"
    )

    x = torch.randn(4, K, dtype=torch.float16, device="cuda")
    out = mod(x)
    assert out.shape == (4, N), f"expected (4, {N}), got {out.shape}"
    assert not torch.isnan(out).any()
    return f"full pipeline ok, output {out.shape}"

_run("convert_wqlinear_layer", test_convert_wqlinear_layer)


# ===================================================================
# 7. Varying dimensions (stress test alignment)
# ===================================================================
print("\n=== 7. Varying Dimensions ===")

# Common Llama-style dimensions
_DIM_CASES = [
    (2048, 2048, 128, "square 2k"),
    (2048, 5632, 128, "hidden->mlp_up"),
    (5632, 2048, 128, "mlp_down->hidden"),
    (2048, 8192, 128, "hidden->large"),
    (4096, 4096, 128, "4k square"),
    (4096, 11008, 128, "llama7b mlp"),
    (256, 512, 128, "small"),
]

for k, n, gs, label in _DIM_CASES:
    def _test(k=k, n=n, gs=gs):
        import torch
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            apply_awq_marlin_linear,
        )
        bufs = _build_marlin_w4a8(k, n, gs)
        x = torch.randn(4, k, dtype=torch.float16, device="cuda")
        out = apply_awq_marlin_linear(
            input=x,
            weight=bufs["qweight"],
            weight_scale=bufs["scales"],
            weight_zp=bufs["qzeros"],
            g_idx=bufs["g_idx"],
            g_idx_sort_indices=bufs["g_idx_sort_indices"],
            workspace=bufs["workspace"],
            quant_type=bufs["quant_type"],
            output_size_per_partition=n,
            input_size_per_partition=k,
            input_global_scale=bufs["input_global_scale"],
            bias=None,
            input_dtype=torch.int8,
        )
        assert out.shape == (4, n)
        assert not torch.isnan(out).any()
        return f"({k}, {n}) gs={gs}"

    _run(f"dims_{label}", _test)


# ===================================================================
# 8. check_marlin_supported for SM120
# ===================================================================
print("\n=== 8. Marlin SM120 Compatibility Check ===")


def test_check_marlin_supported():
    """Check if vLLM thinks Marlin is supported on this GPU."""
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        check_marlin_supported,
    )
    from vllm.scalar_type import scalar_types

    cap = torch.cuda.get_device_capability(0)
    dev_cap = cap[0] * 10 + cap[1]

    supported = check_marlin_supported(
        quant_type=scalar_types.uint4, group_size=128, has_zp=True,
        device_capability=dev_cap,
    )
    return f"sm_{dev_cap}: marlin uint4 w/ zp supported = {supported}"

_run("check_marlin_supported", test_check_marlin_supported)


def test_query_supported_types():
    """List all Marlin-supported quant types on this device."""
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        query_marlin_supported_quant_types,
    )
    cap = torch.cuda.get_device_capability(0)
    dev_cap = cap[0] * 10 + cap[1]

    types_zp = query_marlin_supported_quant_types(has_zp=True, device_capability=dev_cap)
    types_no_zp = query_marlin_supported_quant_types(has_zp=False, device_capability=dev_cap)
    return f"sm_{dev_cap}: with_zp={[str(t) for t in types_zp]}, no_zp={[str(t) for t in types_no_zp]}"

_run("query_supported_types", test_query_supported_types)


# ===================================================================
# Summary
# ===================================================================
print("\n" + "=" * 60)
passed = sum(1 for _, ok, _ in _results if ok)
failed = sum(1 for _, ok, _ in _results if not ok)
total = len(_results)
print(f"RESULTS: {passed}/{total} passed, {failed} failed")

if failed:
    print("\nFailed tests:")
    for name, ok, msg in _results:
        if not ok:
            print(f"  FAIL  {name}: {msg}")
    sys.exit(1)
else:
    print("\nAll tests passed!")
    sys.exit(0)
