/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#define VLLM_MAX_THREADS_PER_SM 2048
#ifndef VLLM_LAUNCH_BLOCKS_CAP
#define VLLM_LAUNCH_BLOCKS_CAP 4
#endif

#define VLLM_BLOCKS_DIV(VAL) (VLLM_MAX_THREADS_PER_SM / (VAL))
#define VLLM_CLAMP_BLOCKS_PER_SM(VAL) \
  (((VAL) <= 0)                       \
       ? 1                            \
       : (((VAL) < VLLM_LAUNCH_BLOCKS_CAP) ? (VAL) : VLLM_LAUNCH_BLOCKS_CAP))
#define VLLM_BLOCKS_PER_SM(BLOCK_THREADS) \
  VLLM_CLAMP_BLOCKS_PER_SM(VLLM_BLOCKS_DIV(BLOCK_THREADS))

static inline int vllm_runtime_blocks_per_sm(int block_threads) {
  int device = -1;
  cudaGetDevice(&device);
  int max_threads_per_sm = VLLM_MAX_THREADS_PER_SM;
  cudaDeviceGetAttribute(&max_threads_per_sm,
                         cudaDevAttrMaxThreadsPerMultiProcessor, device);
  int blocks = (block_threads > 0) ? (max_threads_per_sm / block_threads) : 1;
  return VLLM_CLAMP_BLOCKS_PER_SM(blocks);
}
