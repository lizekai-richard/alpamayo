/******************************************************************************
 * Copyright (c) 2025, Haisheng Chen.
 ******************************************************************************/

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include "../../fused_quant/nvfp4_utils.cuh"


template<typename scalar_t>
struct RotateAccess;


template<>
struct RotateAccess<float> {
  // load & scale into shared buffer
  template<int CTA_M, int GROUP_SIZE, bool USE_SCALE>
  __device__ static void load_group(
      float*        __restrict__   x_grp,
      const float*  __restrict__       x,
      const float*  __restrict__  scales,
      const int           s,
      const int           h,
      const int           j,
      const int           g,
      const int           t
  ) {
    const int base0 = g * GROUP_SIZE + t;
    const int base1 = base0 + GROUP_SIZE/2;
    float scale0 = USE_SCALE ? scales[base0] : float(1);
    float scale1 = USE_SCALE ? scales[base1] : float(1);
    #pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        x_grp[ t                 + i  * GROUP_SIZE ] = x[row * h + base0] * scale0;
        x_grp[(t + GROUP_SIZE/2) + i  * GROUP_SIZE ] = x[row * h + base1] * scale1;
      }
    }
  }



  // load sin/cos and idx into regs
  template<int KROT, int GROUP_SIZE>
  __device__ static void load_coeffs(
      float        reg_theta[KROT],
      int            reg_idx[KROT],
      const int16_t*   __restrict__     idx_ij,
      const float*      __restrict__     theta,
      const int                  h,
      const int                  g,
      const int                  t
  ) {
    #pragma unroll
    for (int r = 0; r < KROT; r++) {
      reg_theta[r] = theta[r*h/2 + g*GROUP_SIZE/2 +  t    ];
      reg_idx[r] = *reinterpret_cast<const int*>(
        idx_ij + r*h + g*GROUP_SIZE + 2*t
      );
    }
  }

  // apply one Givens rotation
  template<int CTA_M, int GROUP_SIZE>
  __device__ static void apply_one(
      float*    __restrict__     x_grp,
      const int         ij,
      const float    theta
  ) {
    int16_t i = ij & 0xFFFF, j = ij >> 16;
    float s_, c_;
    __sincosf(theta, &s_, &c_);
    #pragma unroll
    for (int m = 0; m < CTA_M; m++) {
      float xi = x_grp[i + m * GROUP_SIZE];
      float xj = x_grp[j + m * GROUP_SIZE];
      x_grp[i + m * GROUP_SIZE] = xi * c_ + xj * s_;
      x_grp[j + m * GROUP_SIZE] = xi * (-s_) + xj * c_;
    }
  }

  template<int CTA_M, int GROUP_SIZE>
  __device__ static void store_group(
      float*      __restrict__      out,
      const float*  __restrict__  x_grp,
      const int           s,
      const int           h,
      const int           j,
      const int           g,
      const int           t
  ) {
    const int base0 = g * GROUP_SIZE + t;
    const int base1 = base0 + GROUP_SIZE/2;
    #pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        out[row * h + base0] = x_grp[ t + i * GROUP_SIZE];
        out[row * h + base1] = x_grp[(t + GROUP_SIZE/2) + i * GROUP_SIZE];
      }
    }
  }
};


template<>
struct RotateAccess<__nv_bfloat16> {
  using store_t = __nv_bfloat16;
  using accum_t = float;
  template<int CTA_M, int GROUP_SIZE, bool USE_SCALE>
  __device__ static void load_group(
      float*               __restrict__     x_grp,
      const __nv_bfloat16*   __restrict__    x,
      const __nv_bfloat16* __restrict__  scales,
      const int           s,
      const int           h,
      const int           j,
      const int           g,
      const int           t
  ) {
    const int base0 = g * GROUP_SIZE + t;
    const int base1 = base0 + GROUP_SIZE/2;
    float scale0 = 1.0f;
    float scale1 = 1.0f;
    if constexpr (USE_SCALE) {
      scale0 = __bfloat162float(scales[base0]);
      scale1 = __bfloat162float(scales[base1]);
    }

    #pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        x_grp[ t                 + i * GROUP_SIZE ] = __bfloat162float(x[row * h + base0]) * scale0;
        x_grp[(t + GROUP_SIZE/2) + i * GROUP_SIZE ] = __bfloat162float(x[row * h + base1]) * scale1;
      }
    }
  }

  template<int KROT, int GROUP_SIZE>
  __device__ static void load_coeffs(
      float    reg_theta[KROT],
      int        reg_idx[KROT],
      const int16_t* __restrict__  idx_ij,
      const __nv_bfloat16*  __restrict__    theta,
      const int             h,
      const int             g,
      const int             t
  ) {
    #pragma unroll
    for (int r = 0; r < KROT; r++) {
      reg_theta[r] = __bfloat162float(theta[r*h/2 + g*GROUP_SIZE/2 + t]);
      reg_idx[r] = *reinterpret_cast<const int*>(
        idx_ij + r*h + g*GROUP_SIZE + 2*t
      );
    }
  }

  template<int CTA_M, int GROUP_SIZE>
  __device__ static void apply_one(
      float*    __restrict__     x_grp,
      const int         ij,
      const float    theta
  ) {
    int16_t i = ij & 0xFFFF, j = ij >> 16;
    float s_, c_;
    __sincosf(theta, &s_, &c_);
    #pragma unroll
    for (int m = 0; m < CTA_M; m++) {
      float xi = x_grp[i + m * GROUP_SIZE];
      float xj = x_grp[j + m * GROUP_SIZE];
      x_grp[i + m * GROUP_SIZE] = xi * c_ + xj * s_;
      x_grp[j + m * GROUP_SIZE] = xi * (-s_) + xj * c_;
    }
  }

  template<int CTA_M, int GROUP_SIZE>
  __device__ static void store_group(
      __nv_bfloat16*   __restrict__    out,
      const float*  __restrict__  x_grp,
      const int           s,
      const int           h,
      const int           j,
      const int           g,
      const int           t
  ) {
    const int base0 = g * GROUP_SIZE + t;
    const int base1 = base0 + GROUP_SIZE/2;
    #pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        out[row * h + base0] = __float2bfloat16_rn(x_grp[ t + i * GROUP_SIZE]);
        out[row * h + base1] = __float2bfloat16_rn(x_grp[(t + GROUP_SIZE/2) + i * GROUP_SIZE]);
      }
    }
  }

#ifndef DISABLE_FUSED_KERNELS
  // Fused store and quantization function for __nv_bfloat16
  template<int CTA_M, int GROUP_SIZE>
  __device__ static void fuzed_store_and_quant(
      const float* __restrict__ x_grp,
      const int           s,
      const int           h,
      const int           j,
      const int           g, 
      const int           t,
      float const* SFScale, uint32_t* out, uint32_t* SFout
  ) {
    using namespace vllm;
    const int base = GROUP_SIZE * g;
    constexpr int CVT_FP4_NUM_THREADS_PER_SF = CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;

    int row = j * CTA_M + t;

    if (row < s) {
      using PackedVec = PackedVec<__nv_bfloat16>;
      PackedVec in_vec;
      #pragma unroll
      for (int l = 0; l < CTA_M; l++) {
        __nv_bfloat162 out2 = __floats2bfloat162_rn(
          x_grp[(2*t)     + l * GROUP_SIZE],
          x_grp[(2*t + 1) + l * GROUP_SIZE]
        );
        in_vec.elts[l] = out2;
      }

      int colIdx = (base + 2 * t) / CVT_FP4_ELTS_PER_THREAD;
      auto& out_pos = out[row * h / CVT_FP4_ELTS_PER_THREAD + colIdx];
      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
              row, colIdx, h, SFout);
      const float SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
      out_pos = cvt_warp_fp16_to_fp4<__nv_bfloat16, false>(in_vec, SFScaleVal, sf_out);
    }
  }
#endif // DISABLE_FUSED_KERNELS
};


template<>
struct RotateAccess<__half> {
  using accum_t = float;
  template<int CTA_M, int GROUP_SIZE, bool USE_SCALE>
  __device__ static void load_group(
      float*    __restrict__     x_grp,
      const __half*   __restrict__    x,
      const __half* __restrict__  scales,
      const int           s,
      const int           h,
      const int           j,
      const int           g,
      const int           t
  ) {
    const int base0 = g * GROUP_SIZE + t;
    const int base1 = base0 + GROUP_SIZE/2;
    float scale0 = 1.0f;
    float scale1 = 1.0f;
    if constexpr (USE_SCALE) {
      scale0 = __half2float(scales[base0]);
      scale1 = __half2float(scales[base1]);
    }

    #pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        x_grp[ t                 + i * GROUP_SIZE ] = __half2float(x[row * h + base0]) * scale0;
        x_grp[(t + GROUP_SIZE/2) + i * GROUP_SIZE ] = __half2float(x[row * h + base1]) * scale1;
      }
    }
  }

  template<int KROT, int GROUP_SIZE>
  __device__ static void load_coeffs(
      float    reg_theta[KROT],
      int        reg_idx[KROT],
      const int16_t* __restrict__  idx_ij,
      const __half*  __restrict__    theta,
      const int             h,
      const int             g,
      const int             t
  ) {
    #pragma unroll
    for (int r = 0; r < KROT; r++) {
      reg_theta[r] = __half2float(theta[r*h/2 + g*GROUP_SIZE/2 + t]);
      reg_idx[r] = *reinterpret_cast<const int*>(
        idx_ij + r*h + g*GROUP_SIZE + 2*t
      );
    }
  }

  template<int CTA_M, int GROUP_SIZE>
  __device__ static void apply_one(
      float*    __restrict__     x_grp,
      const int         ij,
      const float    theta
  ) {
    int16_t i = ij & 0xFFFF, j = ij >> 16;
    float s_, c_;
    __sincosf(theta, &s_, &c_);
    #pragma unroll
    for (int m = 0; m < CTA_M; m++) {
      float xi = x_grp[i + m * GROUP_SIZE];
      float xj = x_grp[j + m * GROUP_SIZE];
      x_grp[i + m * GROUP_SIZE] = xi * c_ + xj * s_;
      x_grp[j + m * GROUP_SIZE] = xi * (-s_) + xj * c_;
    }
  }

  template<int CTA_M, int GROUP_SIZE>
  __device__ static void store_group(
      __half*   __restrict__    out,
      const float* __restrict__ x_grp,
      const int           s,
      const int           h,
      const int           j,
      const int           g, 
      const int           t
  ) {
    const int base0 = g * GROUP_SIZE + t;
    const int base1 = base0 + GROUP_SIZE/2;
    #pragma unroll
    for (int i = 0; i < CTA_M; i++) {
      int row = j * CTA_M + i;
      if (row < s) {
        out[row * h + base0] = __float2half_rn(x_grp[ t + i * GROUP_SIZE]);
        out[row * h + base1] = __float2half_rn(x_grp[(t + GROUP_SIZE/2) + i * GROUP_SIZE]);
      }
    }
  }

#ifndef DISABLE_FUSED_KERNELS
  // Fused store and quantization function for __half
  template<int CTA_M, int GROUP_SIZE>
  __device__ static void fuzed_store_and_quant(
      const float* __restrict__ x_grp,
      const int           s,
      const int           h,
      const int           j,
      const int           g, 
      const int           t,
      float const* SFScale, uint32_t* out, uint32_t* SFout
  ) {
    using namespace vllm;
    const int base = GROUP_SIZE * g;
    constexpr int CVT_FP4_NUM_THREADS_PER_SF = CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;

    int row = j * CTA_M + t;

    if (row < s) {
      using PackedVec = PackedVec<__half>;
      PackedVec in_vec;
      #pragma unroll
      for (int l = 0; l < CTA_M; l++) {
        __half2 out2 = __floats2half2_rn(
          x_grp[(2*t)     + l * GROUP_SIZE],
          x_grp[(2*t + 1) + l * GROUP_SIZE]
        );
        in_vec.elts[l] = out2;
      }

      int colIdx = (base + 2 * t) / CVT_FP4_ELTS_PER_THREAD;
      auto& out_pos = out[row * h / CVT_FP4_ELTS_PER_THREAD + colIdx];
      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t, CVT_FP4_NUM_THREADS_PER_SF>(
              row, colIdx, h, SFout);
      const float SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
      out_pos = cvt_warp_fp16_to_fp4<__half, false>(in_vec, SFScaleVal, sf_out);
    }
  }
#endif // DISABLE_FUSED_KERNELS
};
