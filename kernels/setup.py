from setuptools import setup
from torch.utils import cpp_extension
import os
import warnings
import torch


def get_compute_capabilities():
    """Get compute capabilities of all available GPUs, with B200 => '10.0a'."""
    cc_list = set()
    device_count = torch.cuda.device_count()
    if device_count == 0:
        warnings.warn("No CUDA devices found. Building without GPU support.")
        return cc_list

    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 7:
            raise RuntimeError(
                "GPUs with compute capability below 7.0 are not supported."
            )
        cc_list.add(f"{major}.{minor}")
        cc_list.add(f"{major}.{minor}a")
    return cc_list


def generate_nvcc_gencode_flags(compute_caps):
    """Generate -gencode flags for nvcc from a set like {'8.0', '10.0a'}."""
    flags = []
    for cap in compute_caps:
        if cap.endswith("+PTX"):
            base = cap[:-5]
            arch = base.replace(".", "")
            flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
            flags.append(f"-gencode=arch=compute_{arch},code=compute_{arch}")
        else:
            arch = cap.replace(".", "")
            flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
    return flags


# Get compute capabilities of available GPUs
compute_capabilities = get_compute_capabilities()

# Fallback: if no GPU, you might want to error or skip; here we allow empty
if not compute_capabilities:
    raise RuntimeError("No supported CUDA GPU found for building kernels.")

# Generate gencode flags
gencode_flags = generate_nvcc_gencode_flags(compute_capabilities)


# Check if any GPU has capacity <= 100 (e.g., sm90)
def should_disable_fused_kernels():
    """Check if any available GPU has compute capacity <= 100."""
    device_count = torch.cuda.device_count()
    if device_count == 0:
        return True  # Disable fused kernels if no GPU found

    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        compute_capacity = major * 10 + minor
        if compute_capacity <= 100:
            return True
    return False


# Base NVCC flags
nvcc_flags = [
    "-O3",
    "-std=c++17",
    "-DENABLE_BF16",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "-w",
]

# Add define to disable fused kernels for GPUs with capacity <= 100
if should_disable_fused_kernels():
    nvcc_flags.append("-DDISABLE_FUSED_KERNELS")
    print("Disabling fused kernels due to GPU capacity <= 100")
else:
    print("Enabling fused kernels - all GPUs have capacity > 100")

nvcc_flags.extend(gencode_flags)

# Add CUTASS/CUTE include paths
import os

home_dir = os.path.expanduser("~")
include_dirs = [
    os.path.join(home_dir, "zhzhang/cutlass/include"),
    os.path.join(home_dir, "zhzhang/cutlass/tools/util/include"),
]

setup(
    name="paroquant_kernels",
    version="0.1.0",
    packages=["paroquant_kernels"],
    ext_modules=[
        cpp_extension.CUDAExtension(
            "paroquant_kernels._C",
            [
                "paroquant_kernels/src/rotation.cu",
            ],
            extra_compile_args={
                "cxx": ["-O2", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],
                "nvcc": nvcc_flags,
            },
            include_dirs=include_dirs,
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
