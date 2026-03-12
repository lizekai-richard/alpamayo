#pragma once
#include <torch/extension.h>

torch::Tensor rotate_dynamic(at::Tensor x,
                             at::Tensor idx,
                             at::Tensor theta,
                             c10::optional<at::Tensor> scales_opt);

std::tuple<torch::Tensor, torch::Tensor> rotate_and_quant_dynamic(at::Tensor x,
                                                               at::Tensor idx,
                                                               at::Tensor theta,
                                                               c10::optional<at::Tensor> scales_opt,
                                                               at::Tensor sf_scale,
                                                               int64_t group_size);
