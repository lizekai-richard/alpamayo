/******************************************************************************
 * Copyright (c) 2025, Haisheng Chen.
 ******************************************************************************/

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <Python.h>
#include <math_functions.h> 
#include "rotation.cuh"


template<typename scalar_t, int CTA_M, int GROUP_SIZE, int KROT, bool USE_SCALE>
__global__ void rotate(
    const scalar_t* __restrict__ x,
          scalar_t* __restrict__ out,
    const int16_t* __restrict__ idx_ij,
    const scalar_t* __restrict__ theta,
    const scalar_t* __restrict__ scales,
    int s,
    int h
) {
  // shared buffer
  static_assert(CTA_M % 2 == 0, "CTA_M must be even");
  static_assert(GROUP_SIZE % 2 == 0, "GROUP_SIZE must be even");
  __shared__ float x_grp[CTA_M * GROUP_SIZE];

  int j = blockIdx.x;   // row‐block
  int g = blockIdx.y;   // group
  int t = threadIdx.x;  // pair within group

  // load & scale into shared memory
  RotateAccess<scalar_t>::template load_group<CTA_M, GROUP_SIZE, USE_SCALE>(
    x_grp, x, scales, s, h, j, g, t
  );

  // fetch sin/cos + idx into registers
  float reg_theta[KROT];
  int     reg_idx[KROT];
  RotateAccess<scalar_t>::template load_coeffs<KROT, GROUP_SIZE>(
    reg_theta, reg_idx, idx_ij, theta, h, g, t
  );
  __syncthreads();

  // apply all KROT rotations
  #pragma unroll
  for (int r = 0; r < KROT; r++) {
    RotateAccess<scalar_t>::template apply_one<CTA_M, GROUP_SIZE>(
      x_grp, reg_idx[r], reg_theta[r]
    );
    __syncthreads();
  }

  // write back
  RotateAccess<scalar_t>::template store_group<CTA_M, GROUP_SIZE>(
    out, x_grp, s, h, j, g, t
  );
}

#ifndef DISABLE_FUSED_KERNELS
// Fused rotation and quantization kernel
template<typename scalar_t, int CTA_M, int GROUP_SIZE, int KROT, bool USE_SCALE>
__global__ void rotate_and_quant(
    const scalar_t* __restrict__ x,
    const int16_t* __restrict__ idx_ij,
    const scalar_t* __restrict__ theta,
    const scalar_t* __restrict__ scales,
    const float* __restrict__ sf_scale,
    uint32_t* __restrict__ quant_out,
    uint32_t* __restrict__ sf_out,
    int s,
    int h
) {
  // shared buffer
  static_assert(CTA_M % 2 == 0, "CTA_M must be even");
  static_assert(GROUP_SIZE % 2 == 0, "GROUP_SIZE must be even");
  __shared__ float x_grp[CTA_M * GROUP_SIZE];

  int j = blockIdx.x;   // row‐block
  int g = blockIdx.y;   // group
  int t = threadIdx.x;  // pair within group

  // load & scale into shared memory
  RotateAccess<scalar_t>::template load_group<CTA_M, GROUP_SIZE, USE_SCALE>(
    x_grp, x, scales, s, h, j, g, t
  );

  // fetch sin/cos + idx into registers
  float reg_theta[KROT];
  int     reg_idx[KROT];
  RotateAccess<scalar_t>::template load_coeffs<KROT, GROUP_SIZE>(
    reg_theta, reg_idx, idx_ij, theta, h, g, t
  );
  __syncthreads();

  // apply all KROT rotations
  #pragma unroll
  for (int r = 0; r < KROT; r++) {
    RotateAccess<scalar_t>::template apply_one<CTA_M, GROUP_SIZE>(
      x_grp, reg_idx[r], reg_theta[r]
    );
    __syncthreads();
  }

  // fused store and quantization
  RotateAccess<scalar_t>::template fuzed_store_and_quant<CTA_M, GROUP_SIZE>(
    x_grp,  s, h, j, g, t, sf_scale, quant_out, sf_out
  );
}
#endif // DISABLE_FUSED_KERNELS


// C++ launcher
template <int KROT, int CTA_M, int GROUP_SIZE>
torch::Tensor rotate_launcher(
    at::Tensor x,
    at::Tensor idx_ij,
    at::Tensor theta,
    at::Tensor scales
) {
    int h = x.size(-1);
    TORCH_CHECK(h % GROUP_SIZE == 0, "h must be divisible by GROUP_SIZE");
    int groups_per_row = h / GROUP_SIZE;
    constexpr int pn = GROUP_SIZE / 2;
    int seq_len = x.numel() / x.size(-1);
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    at::Tensor out = torch::empty(x.sizes(), options);
    bool has_scale = scales.defined() && scales.numel() > 0;
    
    
    dim3 grid((seq_len + CTA_M - 1) / CTA_M, groups_per_row);
    dim3 block(pn);

    // Launch kernel
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto dtype = x.scalar_type();
    switch (dtype) {
      case at::kFloat: {
        if (has_scale) {
          rotate<float, CTA_M, GROUP_SIZE, KROT, true><<<grid, block, 0, stream>>>(
              x.data_ptr<float>(),
              out.data_ptr<float>(),
              idx_ij.data_ptr<int16_t>(),
              theta.data_ptr<float>(),
              scales.data_ptr<float>(),
              seq_len, h
          );
        } else {
          rotate<float, CTA_M, GROUP_SIZE, KROT, false><<<grid, block, 0, stream>>>(
              x.data_ptr<float>(),
              out.data_ptr<float>(),
              idx_ij.data_ptr<int16_t>(),
              theta.data_ptr<float>(),
              nullptr,
              seq_len, h
          );            
        }
        break;
      }
      case at::kHalf: {
        __half*  x_ptr      = reinterpret_cast<__half*>( x.data_ptr<c10::Half>() );
        __half*  out_ptr    = reinterpret_cast<__half*>( out.data_ptr<c10::Half>() );
        __half*  theta_ptr    = reinterpret_cast<__half*>( theta.data_ptr<c10::Half>() );
        __half*  scales_ptr = nullptr;          
        if (has_scale) {
          scales_ptr = reinterpret_cast<__half*>(scales.data_ptr<c10::Half>());
        }
        if (has_scale) {
          rotate<__half, CTA_M, GROUP_SIZE, KROT, true><<<grid, block, 0, stream>>>(
              x_ptr,
              out_ptr,
              idx_ij.data_ptr<int16_t>(),
              theta_ptr,
              scales_ptr,
              seq_len, h
          );
        } else {
          rotate<__half, CTA_M, GROUP_SIZE, KROT, false><<<grid, block, 0, stream>>>(
              x_ptr,
              out_ptr,
              idx_ij.data_ptr<int16_t>(),
              theta_ptr,
              nullptr,
              seq_len, h
          );            
        }
        break;
      }
      case at::kBFloat16: {
        __nv_bfloat16*  x_ptr      = reinterpret_cast<__nv_bfloat16*>( x.data_ptr<c10::BFloat16>() );
        __nv_bfloat16*  out_ptr    = reinterpret_cast<__nv_bfloat16*>( out.data_ptr<c10::BFloat16>() );
        __nv_bfloat16*  theta_ptr    = reinterpret_cast<__nv_bfloat16*>( theta.data_ptr<c10::BFloat16>() );
        __nv_bfloat16*  scales_ptr = nullptr;          
        if (has_scale) {
          scales_ptr = reinterpret_cast<__nv_bfloat16*>(scales.data_ptr<c10::BFloat16>());
        }
        if (has_scale) {
          rotate<__nv_bfloat16, CTA_M, GROUP_SIZE, KROT, true><<<grid, block, 0, stream>>>(
              x_ptr,
              out_ptr,
              idx_ij.data_ptr<int16_t>(),
              theta_ptr,
              scales_ptr,
              seq_len, h
          );
        } else {
          rotate<__nv_bfloat16, CTA_M, GROUP_SIZE, KROT, false><<<grid, block, 0, stream>>>(
              x_ptr,
              out_ptr,
              idx_ij.data_ptr<int16_t>(),
              theta_ptr,
              nullptr,
              seq_len, h
          );            
        }
        break;
      }
      default:
        TORCH_CHECK(false,
          "rotate only supports Float, Half, and BFloat16, but got ", dtype);
    }
    return out;
}

#ifndef DISABLE_FUSED_KERNELS
// C++ launcher for fused rotation and quantization
template <int KROT, int CTA_M, int GROUP_SIZE>
std::tuple<torch::Tensor, torch::Tensor> rotate_and_quant_launcher(
    at::Tensor x,
    at::Tensor idx_ij,
    at::Tensor theta,
    at::Tensor scales,
    at::Tensor sf_scale
) {
    int h = x.size(-1);
    TORCH_CHECK(h % GROUP_SIZE == 0, "h must be divisible by GROUP_SIZE");
    int groups_per_row = h / GROUP_SIZE;
    constexpr int pn = GROUP_SIZE / 2;
    int seq_len = x.numel() / x.size(-1);
    
    // Calculate output dimensions for quantized data
    int quant_h = h / 8;  // 8 elements per uint32_t
    int sf_h = h / 64;   // 64 elements per SF
    int sf_s = (seq_len + 127) & (~127);
    
    auto quant_options = torch::TensorOptions().dtype(torch::kUInt32).device(x.device());
    auto sf_options = torch::TensorOptions().dtype(torch::kUInt32).device(x.device());
    
    at::Tensor quant_out = torch::empty({seq_len, quant_h}, quant_options);
    at::Tensor sf_out = torch::empty({sf_s, sf_h}, sf_options);
    
    bool has_scale = scales.defined() && scales.numel() > 0;
    
    dim3 grid((seq_len + CTA_M - 1) / CTA_M, groups_per_row);
    dim3 block(pn);

    // Launch kernel
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto dtype = x.scalar_type();
    switch (dtype) {
      case at::kHalf: {
        __half*  x_ptr      = reinterpret_cast<__half*>( x.data_ptr<c10::Half>() );
        __half*  theta_ptr  = reinterpret_cast<__half*>( theta.data_ptr<c10::Half>() );
        __half*  scales_ptr = nullptr;          
        float*   sf_scale_ptr = nullptr;
        if (has_scale) {
          scales_ptr = reinterpret_cast<__half*>(scales.data_ptr<c10::Half>());
        }
        if (sf_scale.defined() && sf_scale.numel() > 0) {
          sf_scale_ptr = sf_scale.data_ptr<float>();
        }
        
        rotate_and_quant<__half, CTA_M, GROUP_SIZE, KROT, true><<<grid, block, 0, stream>>>(
            x_ptr,
            idx_ij.data_ptr<int16_t>(),
            theta_ptr,
            scales_ptr,
            sf_scale_ptr,
            quant_out.data_ptr<uint32_t>(),
            sf_out.data_ptr<uint32_t>(),
            seq_len, h
        );
        break;
      }
      case at::kBFloat16: {
        __nv_bfloat16*  x_ptr      = reinterpret_cast<__nv_bfloat16*>( x.data_ptr<c10::BFloat16>() );
        __nv_bfloat16*  theta_ptr  = reinterpret_cast<__nv_bfloat16*>( theta.data_ptr<c10::BFloat16>() );
        __nv_bfloat16*  scales_ptr = nullptr;          
        float*   sf_scale_ptr = nullptr;
        if (has_scale) {
          scales_ptr = reinterpret_cast<__nv_bfloat16*>(scales.data_ptr<c10::BFloat16>());
        }
        if (sf_scale.defined() && sf_scale.numel() > 0) {
          sf_scale_ptr = sf_scale.data_ptr<float>();
        }
        
        rotate_and_quant<__nv_bfloat16, CTA_M, GROUP_SIZE, KROT, true><<<grid, block, 0, stream>>>(
            x_ptr,
            idx_ij.data_ptr<int16_t>(),
            theta_ptr,
            scales_ptr,
            sf_scale_ptr,
            quant_out.data_ptr<uint32_t>(),
            sf_out.data_ptr<uint32_t>(),
            seq_len, h
        );
        break;
      }
      default:
        TORCH_CHECK(false,
          "rotate_and_quant only supports Half and BFloat16, but got ", dtype);
    }
    return std::make_tuple(quant_out, sf_out);
}
#endif

// Group size = 128
torch::Tensor rotate_k16g128(at::Tensor x,
                            at::Tensor idx,
                            at::Tensor th,
                            at::Tensor sc) {
    return rotate_launcher<16, 4, 128>(
             x, idx, th, sc);
}
torch::Tensor rotate_k8g128(at::Tensor x,
                            at::Tensor idx,
                            at::Tensor th,
                            at::Tensor sc) {
    return rotate_launcher<8, 4, 128>(
             x, idx, th, sc);
}
torch::Tensor rotate_k4g128(at::Tensor x,
                            at::Tensor idx,
                            at::Tensor th,
                            at::Tensor sc) {
    return rotate_launcher<4, 4, 128>(
             x, idx, th, sc);
}
torch::Tensor rotate_k2g128(at::Tensor x,
                            at::Tensor idx,
                            at::Tensor th,
                            at::Tensor sc) {
    return rotate_launcher<2, 4, 128>(
             x, idx, th, sc);
}
torch::Tensor rotate_k1g128(at::Tensor x,
                            at::Tensor idx,
                            at::Tensor th,
                            at::Tensor sc) {
    return rotate_launcher<1, 4, 128>(
             x, idx, th, sc);
}

// Group size = 64
torch::Tensor rotate_k16g64(at::Tensor x,
                            at::Tensor idx,
                            at::Tensor th,
                            at::Tensor sc) {
    return rotate_launcher<16, 4, 64>(
             x, idx, th, sc);
}
torch::Tensor rotate_k8g64(at::Tensor x,
                            at::Tensor idx,
                            at::Tensor th,
                            at::Tensor sc) {
    return rotate_launcher<8, 4, 64>(
             x, idx, th, sc);
}
torch::Tensor rotate_k4g64(at::Tensor x,
                            at::Tensor idx,
                            at::Tensor th,
                            at::Tensor sc) {
    return rotate_launcher<4, 4, 64>(
             x, idx, th, sc);
}
torch::Tensor rotate_k2g64(at::Tensor x,
                            at::Tensor idx,
                            at::Tensor th,
                            at::Tensor sc) {
    return rotate_launcher<2, 4, 64>(
             x, idx, th, sc);
}
torch::Tensor rotate_k1g64(at::Tensor x,
                            at::Tensor idx,
                            at::Tensor th,
                            at::Tensor sc) {
    return rotate_launcher<1, 4, 64>(
             x, idx, th, sc);
}

#ifndef DISABLE_FUSED_KERNELS
// Fused rotation and quantization functions for group size = 128
std::tuple<torch::Tensor, torch::Tensor> rotate_and_quant_k8g128(at::Tensor x,
                                                                at::Tensor idx,
                                                                at::Tensor th,
                                                                at::Tensor sc,
                                                                at::Tensor sf_scale) {
    return rotate_and_quant_launcher<8, 4, 128>(x, idx, th, sc, sf_scale);
}
std::tuple<torch::Tensor, torch::Tensor> rotate_and_quant_k4g128(at::Tensor x,
                                                                at::Tensor idx,
                                                                at::Tensor th,
                                                                at::Tensor sc,
                                                                at::Tensor sf_scale) {
    return rotate_and_quant_launcher<4, 4, 128>(x, idx, th, sc, sf_scale);
}
std::tuple<torch::Tensor, torch::Tensor> rotate_and_quant_k2g128(at::Tensor x,
                                                                at::Tensor idx,
                                                                at::Tensor th,
                                                                at::Tensor sc,
                                                                at::Tensor sf_scale) {
    return rotate_and_quant_launcher<2, 4, 128>(x, idx, th, sc, sf_scale);
}
std::tuple<torch::Tensor, torch::Tensor> rotate_and_quant_k1g128(at::Tensor x,
                                                                at::Tensor idx,
                                                                at::Tensor th,
                                                                at::Tensor sc,
                                                                at::Tensor sf_scale) {
    return rotate_and_quant_launcher<1, 4, 128>(x, idx, th, sc, sf_scale);
}

// Fused rotation and quantization functions for group size = 64
std::tuple<torch::Tensor, torch::Tensor> rotate_and_quant_k8g64(at::Tensor x,
                                                               at::Tensor idx,
                                                               at::Tensor th,
                                                               at::Tensor sc,
                                                               at::Tensor sf_scale) {
    return rotate_and_quant_launcher<8, 4, 64>(x, idx, th, sc, sf_scale);
}
std::tuple<torch::Tensor, torch::Tensor> rotate_and_quant_k4g64(at::Tensor x,
                                                               at::Tensor idx,
                                                               at::Tensor th,
                                                               at::Tensor sc,
                                                               at::Tensor sf_scale) {
    return rotate_and_quant_launcher<4, 4, 64>(x, idx, th, sc, sf_scale);
}
std::tuple<torch::Tensor, torch::Tensor> rotate_and_quant_k2g64(at::Tensor x,
                                                               at::Tensor idx,
                                                               at::Tensor th,
                                                               at::Tensor sc,
                                                               at::Tensor sf_scale) {
    return rotate_and_quant_launcher<2, 4, 64>(x, idx, th, sc, sf_scale);
}
std::tuple<torch::Tensor, torch::Tensor> rotate_and_quant_k1g64(at::Tensor x,
                                                               at::Tensor idx,
                                                               at::Tensor th,
                                                               at::Tensor sc,
                                                               at::Tensor sf_scale) {
    return rotate_and_quant_launcher<1, 4, 64>(x, idx, th, sc, sf_scale);
}
#endif // DISABLE_FUSED_KERNELS


torch::Tensor rotate_dynamic(at::Tensor x,
                             at::Tensor idx,
                             at::Tensor theta,
                             c10::optional<at::Tensor> scales_opt,
                             int64_t group_size) {
    int64_t krot = theta.size(0);  
    TORCH_CHECK(
      krot == idx.size(0),
      "theta.size(0) must equal idx_ij.size(0)"
    ); 
    at::Tensor scales = scales_opt.value_or(at::Tensor());

    if (group_size == 128) {
        switch (krot) {
            case 16: return rotate_k16g128(x, idx, theta, scales);
            case 8:  return rotate_k8g128(x, idx, theta, scales);
            case 4:  return rotate_k4g128(x, idx, theta, scales);
            case 2:  return rotate_k2g128(x, idx, theta, scales);
            case 1:  return rotate_k1g128(x, idx, theta, scales);
            default:
                TORCH_CHECK(false, "Unsupported KROT = ", krot,
                            "; compiled variants: 1/2/4/8");
        }
    } else if (group_size == 64) {
        switch (krot) {
            case 16: return rotate_k16g64(x, idx, theta, scales);
            case 8:  return rotate_k8g64(x, idx, theta, scales);
            case 4:  return rotate_k4g64(x, idx, theta, scales);
            case 2:  return rotate_k2g64(x, idx, theta, scales);
            case 1:  return rotate_k1g64(x, idx, theta, scales);
            default:
                TORCH_CHECK(false, "Unsupported KROT = ", krot,
                            "; compiled variants: 1/2/4/8");
        }
    } 
    
    TORCH_CHECK(false, "Unexpected group_size: ", group_size);
}

std::tuple<torch::Tensor, torch::Tensor> rotate_and_quant_dynamic(at::Tensor x,
                                                               at::Tensor idx,
                                                               at::Tensor theta,
                                                               c10::optional<at::Tensor> scales_opt,
                                                               at::Tensor sf_scale,
                                                               int64_t group_size) {
#ifdef DISABLE_FUSED_KERNELS
    // When fused kernels are disabled, fall back to regular rotation and return dummy tensors
    TORCH_CHECK(false, "Fused kernels are disabled. rotate_and_quant is not available.");
#else
    int64_t krot = theta.size(0);  
    TORCH_CHECK(
      krot == idx.size(0),
      "theta.size(0) must equal idx_ij.size(0)"
    ); 
    at::Tensor scales = scales_opt.value_or(at::Tensor());

    if (group_size == 128) {
        switch (krot) {
            case 8: return rotate_and_quant_k8g128(x, idx, theta, scales, sf_scale);
            case 4: return rotate_and_quant_k4g128(x, idx, theta, scales, sf_scale);
            case 2: return rotate_and_quant_k2g128(x, idx, theta, scales, sf_scale);
            case 1: return rotate_and_quant_k1g128(x, idx, theta, scales, sf_scale);
            default:
                TORCH_CHECK(false, "Unsupported KROT = ", krot,
                            "; compiled variants: 1/2/4/8");
        }
    } else if (group_size == 64) {
        switch (krot) {
            case 8: return rotate_and_quant_k8g64(x, idx, theta, scales, sf_scale);
            case 4: return rotate_and_quant_k4g64(x, idx, theta, scales, sf_scale);
            case 2: return rotate_and_quant_k2g64(x, idx, theta, scales, sf_scale);
            case 1: return rotate_and_quant_k1g64(x, idx, theta, scales, sf_scale);
            default:
                TORCH_CHECK(false, "Unsupported KROT = ", krot,
                            "; compiled variants: 1/2/4/8");
        }
    } 

    TORCH_CHECK(false, "Unexpected group_size: ", group_size);
#endif // DISABLE_FUSED_KERNELS
}

// C++ launcher
torch::Tensor rotate_k8g128half_launcher(
    at::Tensor x,
    at::Tensor idx_ij,
    at::Tensor theta,
    at::Tensor scales
) {
    int h = x.size(-1);
    const int KROT = 8;
    const int CTA_M = 4;
    const int GROUP_SIZE = 128;
    TORCH_CHECK(h % GROUP_SIZE == 0, "h must be divisible by GROUP_SIZE");
    int groups_per_row = h / GROUP_SIZE;
    constexpr int pn = GROUP_SIZE / 2;
    int seq_len = x.numel() / x.size(-1);
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    at::Tensor out = torch::empty(x.sizes(), options);
    dim3 grid((seq_len + CTA_M - 1) / CTA_M, groups_per_row);
    dim3 block(pn);

    // Launch kernel
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    __half*  x_ptr      = reinterpret_cast<__half*>( x.data_ptr<c10::Half>() );
    __half*  out_ptr    = reinterpret_cast<__half*>( out.data_ptr<c10::Half>() );
    __half*  theta_ptr  = reinterpret_cast<__half*>( theta.data_ptr<c10::Half>() );
    __half*  scales_ptr = reinterpret_cast<__half*>(scales.data_ptr<c10::Half>());      

    rotate<__half, CTA_M, GROUP_SIZE, KROT, true><<<grid, block, 0, stream>>>(
        x_ptr,
        out_ptr,
        idx_ij.data_ptr<int16_t>(),
        theta_ptr,
        scales_ptr,
        seq_len, h
    );
    return out;
}

TORCH_LIBRARY(rotation, m) {
    m.def("rotate(Tensor x, Tensor idx_ij, Tensor theta, Tensor? scales, int group_size) -> Tensor");
    m.def("rotate_and_quant(Tensor x, Tensor idx_ij, Tensor theta, Tensor? scales, Tensor sf_scale, int group_size) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(rotation, CUDA, m) {
    m.impl("rotate", &rotate_dynamic);
    m.impl("rotate_and_quant", &rotate_and_quant_dynamic);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { }
