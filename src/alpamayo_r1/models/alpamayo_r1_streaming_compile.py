# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
from typing import Any

import einops
import hydra.utils as hyu
import numpy as np
import torch
from transformers import AutoConfig, AutoModel
from transformers import StaticCache
from transformers.generation.logits_process import (
    LogitsProcessor,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from alpamayo_r1.action_space import ActionSpace
from alpamayo_r1.models.streaming_model import ReasoningVLA
from alpamayo_r1.config import AlpamayoR1Config
from alpamayo_r1.diffusion.base import BaseDiffusion
from alpamayo_r1.models.patches import patch_for_torch_compile
from alpamayo_r1.models.patches import StaticCache
from alpamayo_r1.models.token_utils import (
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)
from alpamayo_r1.models.streaming_masking_utils import (
    create_streaming_attention_mask_sdpa_optimized,
)

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class ExpertLogitsProcessor(LogitsProcessor):
    """Masks out the logits for discrete trajectory tokens."""

    def __init__(self, traj_token_offset: int, traj_vocab_size: int):
        super().__init__()
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size] = float("-inf")
        return scores


class StreamingAlpamayoR1(ReasoningVLA):
    """Streaming Expert model for reasoning VLA with torch.compile support."""

    config_class: type[AlpamayoR1Config] = AlpamayoR1Config
    base_model_prefix = "vlm"

    def __init__(
        self,
        config: AlpamayoR1Config,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
    ):
        super().__init__(config, pretrained_modules, original_vocab_size, print_param_count=False)

        # Expert model setup
        expert_config = copy.deepcopy(self.vlm.config.text_config)
        if config.expert_cfg is not None:
            for key, value in config.expert_cfg.items():
                setattr(expert_config, key, value)
        self.expert = AutoModel.from_config(expert_config)

        # Action space and diffusion setup
        self.action_space: ActionSpace = hyu.instantiate(config.action_space_cfg)
        self.diffusion: BaseDiffusion = hyu.instantiate(
            config.diffusion_cfg,
            x_dims=self.action_space.get_action_space_dims(),
        )

        self.action_in_proj = hyu.instantiate(
            config.action_in_proj_cfg,
            in_dims=self.action_space.get_action_space_dims(),
            out_dim=expert_config.hidden_size,
        )
        self.action_out_proj = hyu.instantiate(
            config.action_out_proj_cfg,
            in_features=expert_config.hidden_size,
            out_features=self.action_space.get_action_space_dims()[-1],
        )

        # Convert action-related modules to the same dtype as expert
        expert_dtype = self.expert.dtype
        if self.config.keep_same_dtype:
            self.diffusion = self.diffusion.to(dtype=expert_dtype)
            self.action_in_proj = self.action_in_proj.to(dtype=expert_dtype)
            self.action_out_proj = self.action_out_proj.to(dtype=expert_dtype)

        self.post_init()

        # Streaming-specific parameters
        self.prefill_seq_length = 3006
        self.max_cache_len = self.prefill_seq_length + 256
        self.num_views = 4
        self.num_frames_per_view = 4
        self.num_image_tokens_per_frame = 180

        # Streaming state (will be initialized on first call)
        self._past_key_values = None
        self._cached_position_ids = None
        self._cached_attention_mask = None
        self._cached_streaming_attention_mask = None
        self._cached_rope_deltas = None
        self.vision_start_end_ids_ranges = None
        self.image_token_ids_ranges = None
        self.traj_and_text_ids_range = None
        self.is_first_prefill = True

    # ==================== Properties ====================

    @property
    def traj_start_token_id(self) -> int:
        """Token ID for <traj_future_start>."""
        if not hasattr(self, "_traj_start_token_id"):
            self._traj_start_token_id = self.tokenizer.convert_tokens_to_ids(
                to_special_token("traj_future_start")
            )
        return self._traj_start_token_id

    @property
    def num_action_tokens(self) -> int:
        """Number of action tokens (trajectory length)."""
        return self.action_space.get_action_space_dims()[0]

    # ==================== Streaming Helpers ====================

    def _retrieve_streaming_related_inputs(self, input_ids):
        """
        Returns vision and trajectory token ranges for streaming.

        Returns:
            vision_start_end_ids_ranges: Per-view list of frame ranges [num_views][num_frames_per_view]
            image_token_ids_ranges: Per-view list of image token ranges
            traj_and_text_ids_range: (start, end) tuple for trajectory and text tokens
        """
        vision_start_end_ids_ranges = [[] for _ in range(self.num_views)]
        image_token_ids_ranges = [[] for _ in range(self.num_views)]

        vision_start_token = "<|vision_start|>"
        vision_end_token = "<|vision_end|>"
        vision_start_token_id = self.tokenizer.encode(vision_start_token)[0]
        vision_end_token_id = self.tokenizer.encode(vision_end_token)[0]

        vision_start_token_mask = (input_ids == vision_start_token_id)
        vision_end_token_mask = (input_ids == vision_end_token_id)

        all_vision_start_token_ids = torch.where(vision_start_token_mask)[1]
        all_vision_end_token_ids = torch.where(vision_end_token_mask)[1]

        for frame_idx, (vision_start, vision_end) in enumerate(
            zip(all_vision_start_token_ids, all_vision_end_token_ids)
        ):
            view_idx = frame_idx // self.num_frames_per_view
            vision_start_end_ids_ranges[view_idx].append((vision_start.item(), vision_end.item() + 1))
            image_token_ids_ranges[view_idx].append((vision_start.item() + 1, vision_end.item()))

        last_vision_end_id = all_vision_end_token_ids[-1]
        traj_and_text_ids_range = (last_vision_end_id.item() + 1, self.prefill_seq_length)

        return vision_start_end_ids_ranges, image_token_ids_ranges, traj_and_text_ids_range

    def _update_past_key_values(self):
        """Shift KV cache: move frames 2-4 to positions 1-3 for each view."""
        for layer in self._past_key_values.layers:
            key_cache = layer.keys
            value_cache = layer.values

            for i in range(self.num_views):
                new_kv_start = self.vision_start_end_ids_ranges[i][0][0]
                new_kv_end = self.vision_start_end_ids_ranges[i][-2][1]
                old_kv_start = self.vision_start_end_ids_ranges[i][1][0]
                old_kv_end = self.vision_start_end_ids_ranges[i][-1][1]

                key_cache[:, :, new_kv_start:new_kv_end, :].copy_(
                    key_cache[:, :, old_kv_start:old_kv_end, :].clone()
                )
                value_cache[:, :, new_kv_start:new_kv_end, :].copy_(
                    value_cache[:, :, old_kv_start:old_kv_end, :].clone()
                )

    def _create_cache_position(self) -> torch.Tensor:
        """Create cache positions for streaming prefill."""
        if self.is_first_prefill:
            return torch.arange(0, self.prefill_seq_length)
        else:
            cache_position = []
            for i in range(self.num_views):
                start, end = self.vision_start_end_ids_ranges[i][-1]
                cache_position.append(torch.arange(start, end))
            cache_position.append(
                torch.arange(self.traj_and_text_ids_range[0], self.traj_and_text_ids_range[1])
            )
            return torch.cat(cache_position, dim=0)

    def _get_streaming_attention_mask(
        self,
        cache_position: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Create streaming attention mask for non-first prefill."""
        return create_streaming_attention_mask_sdpa_optimized(
            batch_size=1,
            cache_position=cache_position,
            kv_length=self.max_cache_len,
            vision_start_end_ids_ranges=self.vision_start_end_ids_ranges,
            traj_and_text_ids_range=self.traj_and_text_ids_range,
            valid_length=self.prefill_seq_length,
            device=device,
            dtype=dtype,
        )

    def _crop_static_cache(self, valid_length: int):
        """Zero out cache positions beyond valid_length."""
        for layer in self._past_key_values.layers:
            if layer.is_initialized:
                layer.keys[:, :, valid_length:, :].zero_()
                layer.values[:, :, valid_length:, :].zero_()

    # ==================== Compiled Functions ====================

    def _encode(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode images using the visual encoder."""
        if not hasattr(self, "_encode_fn"):
            self._encode_pixel_values = torch.empty_like(pixel_values)
            self._encode_image_grid_thw = torch.empty_like(image_grid_thw)

            def encode_fn():
                pixels = self._encode_pixel_values.type(self.vlm.model.visual.dtype)
                return self.vlm.model.visual(pixels, grid_thw=self._encode_image_grid_thw)

            self._encode_fn = encode_fn

        self._encode_pixel_values.copy_(pixel_values)
        self._encode_image_grid_thw.copy_(image_grid_thw)

        if not hasattr(self, "_compiled_encode_fn"):
            if self._torch_compile:
                self._encode_fn()  # Warmup
                self._compiled_encode_fn = torch.compile(
                    self._encode_fn, mode=self._torch_compile, fullgraph=True
                )
            else:
                self._compiled_encode_fn = self._encode_fn

        if self._torch_compile:
            torch.compiler.cudagraph_mark_step_begin()
        return self._compiled_encode_fn()

    def _prefill(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        deepstack_image_embeds: list[torch.Tensor],
        streaming_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run prefill forward pass."""
        if not hasattr(self, "_prefill_fn"):
            self._prefill_inputs_embeds = torch.empty_like(inputs_embeds)
            self._prefill_position_ids = torch.empty_like(position_ids)
            self._prefill_cache_position = torch.empty_like(cache_position)
            self._prefill_visual_pos_masks = torch.empty_like(visual_pos_masks)
            self._prefill_deepstack_embeds = [torch.empty_like(e) for e in deepstack_image_embeds]
            self._prefill_streaming_attention_mask = torch.empty_like(streaming_attention_mask)

            def prefill_fn():
                hidden = self.vlm.model.language_model(
                    inputs_embeds=self._prefill_inputs_embeds,
                    position_ids=self._prefill_position_ids,
                    past_key_values=self._past_key_values,
                    cache_position=self._prefill_cache_position,
                    visual_pos_masks=self._prefill_visual_pos_masks,
                    deepstack_visual_embeds=self._prefill_deepstack_embeds,
                    streaming_attention_mask=self._prefill_streaming_attention_mask,
                    use_cache=True,
                ).last_hidden_state[:, -1]
                return self.vlm.lm_head(hidden)

            self._prefill_fn = prefill_fn

        self._prefill_inputs_embeds.copy_(inputs_embeds)
        self._prefill_position_ids.copy_(position_ids)
        self._prefill_cache_position.copy_(cache_position)
        self._prefill_visual_pos_masks.copy_(visual_pos_masks)
        self._prefill_streaming_attention_mask.copy_(streaming_attention_mask)
        for buf, emb in zip(self._prefill_deepstack_embeds, deepstack_image_embeds):
            buf.copy_(emb)
        
        if not hasattr(self, "_compiled_prefill_fn"):
            if self._torch_compile:
                self._prefill_fn()
                self._compiled_prefill_fn = torch.compile(
                    self._prefill_fn, mode=self._torch_compile, fullgraph=True
                )
            else:
                self._compiled_prefill_fn = self._prefill_fn

        if self._torch_compile:
            torch.compiler.cudagraph_mark_step_begin()
        return self._compiled_prefill_fn()

    def _decode(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        """Run decode forward pass for a single token."""
        if not hasattr(self, "_decode_fn"):
            self._decode_input_ids = torch.empty_like(input_ids)
            self._decode_position_ids = torch.empty_like(position_ids)
            self._decode_cache_position = torch.empty_like(cache_position)

            def decode_fn():
                hidden = self.vlm.model.language_model(
                    input_ids=self._decode_input_ids,
                    position_ids=self._decode_position_ids,
                    past_key_values=self._past_key_values,
                    cache_position=self._decode_cache_position,
                    use_cache=True,
                ).last_hidden_state[:, -1]
                return self.vlm.lm_head(hidden)

            self._decode_fn = decode_fn

        self._decode_input_ids.copy_(input_ids)
        self._decode_position_ids.copy_(position_ids)
        self._decode_cache_position.copy_(cache_position)

        if not hasattr(self, "_compiled_decode_fn"):
            if self._torch_compile:
                self._decode_fn()
                self._compiled_decode_fn = torch.compile(
                    self._decode_fn, mode=self._torch_compile, fullgraph=True
                )
            else:
                self._compiled_decode_fn = self._decode_fn

        if self._torch_compile:
            torch.compiler.cudagraph_mark_step_begin()
        return self._compiled_decode_fn()

    def _action(
        self,
        num_action_tokens: int,
        total_samples: int,
        device: torch.device,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        diffusion_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if not hasattr(self, "_action_fn"):
            # Initialize static buffers
            self._action_attention_mask = torch.empty_like(attention_mask)
            self._action_cache_position = torch.empty_like(cache_position)
            self._action_noise = torch.empty(
                total_samples, *self.action_space.get_action_space_dims(), device=device, dtype=torch.bfloat16
            )
            expert_kwargs = {"is_causal": False} if self.config.expert_non_causal_attention else {}
            action_dims = self.action_space.get_action_space_dims()

            def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                action_embeds = self.action_in_proj(x, t)
                if action_embeds.dim() == 2:
                    action_embeds = action_embeds.view(x.shape[0], num_action_tokens, -1)

                hidden = self.expert(
                    inputs_embeds=action_embeds,
                    position_ids=self._cached_position_ids,
                    past_key_values=self._past_key_values,
                    attention_mask=self._action_attention_mask,
                    cache_position=self._action_cache_position,
                    use_cache=True,
                    **expert_kwargs,
                ).last_hidden_state[:, -num_action_tokens:]
                return self.action_out_proj(hidden).view(-1, *action_dims)

            def action_fn():
                return self.diffusion.sample(
                    noise=self._action_noise,
                    batch_size=total_samples,
                    step_fn=step_fn,
                    device=device,
                    return_all_steps=False,
                    **(diffusion_kwargs or {}),
                )

            self._action_fn = action_fn

        # Copy inputs to static buffers
        # self._action_position_ids.copy_(position_ids)
        self._action_attention_mask.copy_(attention_mask)
        self._action_cache_position.copy_(cache_position)

        # Generate noise outside compiled graph for deterministic RNG
        self._action_noise.normal_()

        # Warmup and compile on first call (if enabled)
        if not hasattr(self, "_compiled_action_fn"):
            if self._torch_compile:
                self._action_fn()  # Warmup for compile
                self._compiled_action_fn = torch.compile(
                    self._action_fn, mode=self._torch_compile, fullgraph=True
                )
            else:
                self._compiled_action_fn = self._action_fn

        if self._torch_compile:
            torch.compiler.cudagraph_mark_step_begin()
        return self._compiled_action_fn()

    # ==================== Logits Processor ====================

    def _build_logits_processor(
        self,
        temperature: float,
        top_k: int | None,
        top_p: float,
    ) -> LogitsProcessorList:
        """Build logits processor for generation."""
        processors = [
            ExpertLogitsProcessor(
                traj_token_offset=self.config.traj_token_start_idx,
                traj_vocab_size=self.config.traj_vocab_size,
            )
        ]
        if temperature > 0 and temperature != 1.0:
            processors.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k > 0:
            processors.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
        if top_p < 1.0:
            processors.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))
        return LogitsProcessorList(processors)

    def _find_traj_start_positions(self, output_ids: torch.Tensor) -> torch.Tensor:
        """Find <traj_future_start> token position for each sequence."""
        traj_start_mask = output_ids == self.traj_start_token_id
        has_traj_start = traj_start_mask.any(dim=1)

        if not has_traj_start.all():
            missing = (~has_traj_start).nonzero(as_tuple=True)[0].tolist()
            logger.warning(f"No <traj_future_start> token found in sequences: {missing}")

        return torch.where(
            has_traj_start,
            traj_start_mask.int().argmax(dim=1),
            output_ids.shape[1] - 1,
        )

    def _find_first_truncation_positions(
        self,
        output_ids: torch.Tensor,
        prompt_length: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Find first truncation position (sentence-ending '.' or <|im_end|>) after prompt tokens."""
        if not hasattr(self, "_truncation_token_mask"):
            vocab_size = len(self.tokenizer)
            token_strings = self.tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
            im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            self._truncation_token_mask = torch.tensor(
                [(token is not None and token.endswith(".")) or tid == im_end_token_id
                 for tid, token in enumerate(token_strings)],
                dtype=torch.bool,
            )

        period_token_mask = self._truncation_token_mask.to(output_ids.device)[output_ids]
        if prompt_length is None:
            prompt_length = getattr(self, "_cached_prompt_length", 0)

        seq_len = output_ids.shape[1]
        if isinstance(prompt_length, int):
            position_indices = torch.arange(seq_len, device=output_ids.device)
            mask_after_prompt = position_indices >= prompt_length
        else:
            prompt_length = torch.as_tensor(prompt_length, device=output_ids.device).long()
            position_indices = torch.arange(seq_len, device=output_ids.device).unsqueeze(0)
            if prompt_length.dim() == 0:
                mask_after_prompt = position_indices >= prompt_length.item()
            else:
                mask_after_prompt = position_indices >= prompt_length[:, None]

        period_token_mask = period_token_mask & mask_after_prompt
        has_period = period_token_mask.any(dim=1)

        period_positions = period_token_mask.int().argmax(dim=1)
        if has_period.all():
            return period_positions

        return -1

    def _find_first_period_positions(
        self,
        output_ids: torch.Tensor,
        start_pos: torch.Tensor,
        end_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Find first token containing '.' between start_pos and end_pos (inclusive)."""
        output_ids_cpu = output_ids.tolist()
        start_pos_cpu = start_pos.tolist()
        end_pos_cpu = end_pos.tolist()
        first_period_positions = []
        for seq_ids, seq_start, seq_end in zip(output_ids_cpu, start_pos_cpu, end_pos_cpu):
            if seq_end < seq_start:
                first_period_positions.append(seq_end)
                continue
            tokens = self.tokenizer.convert_ids_to_tokens(seq_ids[seq_start : seq_end + 1])
            found = None
            for idx, token in enumerate(tokens):
                if "." in token:
                    found = seq_start + idx
                    break
            if found is None:
                found = seq_end
            first_period_positions.append(found)
        return torch.tensor(
            first_period_positions, device=output_ids.device, dtype=end_pos.dtype
        )

    # ==================== Main Inference ====================

    # We don't need any output from the first prefill. Only need to cache key/values, position_ids, and attention_mask.
    def _first_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        num_samples: int,
        device: torch.device,
    ):
        # Cache all streaming related inputs
        (
            vision_start_end_ids_ranges,
            image_token_ids_ranges,
            traj_and_text_ids_range,
        ) = self._retrieve_streaming_related_inputs(input_ids[:1])

        cache_position = self._create_cache_position().to(device)

        position_ids, rope_deltas = self.vlm.model.get_rope_index(input_ids, image_grid_thw)
        # Pad position_ids to max_cache_len
        padding_length = self.max_cache_len - self.prefill_seq_length
        if padding_length > 0:
            last_pos = position_ids[:, :, -1:]
            padding_pos = last_pos + torch.arange(1, padding_length + 1, device=device)
            position_ids = torch.cat([position_ids, padding_pos], dim=-1)

        self._cached_position_ids = position_ids
        self._cached_rope_deltas = rope_deltas
        self._cached_attention_mask = attention_mask
        self.vision_start_end_ids_ranges = vision_start_end_ids_ranges
        self.image_token_ids_ranges = image_token_ids_ranges
        self.traj_and_text_ids_range = traj_and_text_ids_range

        # Launch the first non-streaming prefill
        inputs_embeds = self.vlm.model.get_input_embeddings()(input_ids)
        image_embeds, deepstack_image_embeds = self.vlm.model.visual(pixel_values, grid_thw=image_grid_thw)
        image_mask = (input_ids == self.vlm.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        _ = self.vlm.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=self._past_key_values,
            cache_position=cache_position,
            visual_pos_masks=image_mask[..., 0],
            deepstack_visual_embeds=deepstack_image_embeds,
            streaming_attention_mask=None,
            use_cache=True,
        ).last_hidden_state[:, -1]

        # Delete the cached_pos_embeds in QwenVLVisionModel
        if hasattr(self.vlm.model.visual, "_cached_pos_embeds"):
            delattr(self.vlm.model.visual, "_cached_pos_embeds")
        logger.info("Deleted _cached_pos_embeds in Qwen3VLVisionModel")
        
        for block in self.vlm.model.visual.blocks:
            if hasattr(block.attn, "_num_chunks"):
                delattr(block.attn, "_num_chunks")
        logger.info("Deleted _num_chunks in Qwen3VLVisionModel")
        
        if hasattr(self.vlm.model.language_model, "_cached_deepstack_indices"):
            delattr(self.vlm.model.language_model, "_cached_deepstack_indices")
        logger.info("Deleted _cached_deepstack_indices in Qwen3VLTextModel")

    @torch.inference_mode()
    def sample_trajectories_from_data_with_streaming_vlm_rollout(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        diffusion_kwargs: dict[str, Any] | None = None,
        torch_compile: str = "max-autotune",
        fuse_qkv: bool = False,
        fuse_gate_up: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample trajectories with streaming VLM rollout.

        Args:
            data: Input data containing tokenized_data, ego_history_xyz, ego_history_rot.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            temperature: Sampling temperature.
            num_traj_samples: Number of trajectory samples.
            num_traj_sets: Number of trajectory sets.
            diffusion_kwargs: Additional kwargs for diffusion sampling.
            use_compile: Whether to use torch.compile for optimization.
            fuse_qkv: Whether to fuse q/k/v projections into a single QKVLinear.
            fuse_gate_up: Whether to fuse gate/up projections into a single MergedColumnLinear.
        Returns:
            pred_xyz: Predicted xyz trajectories.
            pred_rot: Predicted rotation trajectories.
            extra: (optional) Extra information including CoC text.
        """
        self._torch_compile = torch_compile
        if self._torch_compile and not hasattr(self, "_patched_for_compile"):
            patch_for_torch_compile(self, mode="streaming", fuse_qkv=fuse_qkv, fuse_gate_up=fuse_gate_up)
            self._patched_for_compile = True

        # Extract inputs
        tokenized = data["tokenized_data"]
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        pixel_values = tokenized["pixel_values"]
        image_grid_thw = tokenized["image_grid_thw"]
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]

        batch_size, num_traj_groups, _, _ = ego_history_xyz.shape
        num_samples = num_traj_samples * num_traj_sets
        assert num_traj_groups == 1, "Only one trajectory group is supported."
        device = input_ids.device

        # Fuse trajectory tokens
        input_ids = self.fuse_traj_tokens(
            input_ids, {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot}
        )

        # Setup generation
        max_new_tokens = kwargs.get("max_generation_length", self.config.tokens_per_future_traj)
        logits_processor = self._build_logits_processor(temperature, top_k, top_p)
        self.max_cache_len = self.prefill_seq_length + max_new_tokens + self.num_action_tokens

        # Initialize KV cache on first call
        if self._past_key_values is None:
            self._past_key_values = StaticCache(
                config=self.vlm.config,
                max_cache_len=self.max_cache_len,
                max_batch_size=num_samples * batch_size,
                offloading=False,
            )
        
        if self.is_first_prefill:
            logger.info("First prefill: caching KV and returning (no streaming logs yet).")
            self._first_prefill(input_ids, attention_mask, pixel_values, image_grid_thw, num_samples, device)
            self._update_past_key_values()
            self.is_first_prefill = False
            return

        # Prepare input_ids for streaming (slice off system prompt for non-first prefill)
        vision_start_token_id = self.tokenizer.encode("<|vision_start|>")[0]
        first_vision_start = torch.where(input_ids == vision_start_token_id)[1][0].item()
        input_ids = input_ids[:, first_vision_start:]

        # Create cache position and position IDs
        cache_position = self._create_cache_position().to(device)
        seq_len = input_ids.shape[1]
        self._cached_prompt_length = seq_len

        # Get embeddings
        inputs_embeds = self.vlm.model.get_input_embeddings()(input_ids)

        # ===== Encode =====
        image_embeds, deepstack_image_embeds = self._encode(pixel_values, image_grid_thw)
        image_mask = (input_ids == self.vlm.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # Create streaming attention mask for non-first prefill
        if self._cached_streaming_attention_mask is None:
            streaming_attention_mask = self._get_streaming_attention_mask(
                cache_position=cache_position,
                device=device,
            )
            self._cached_streaming_attention_mask = streaming_attention_mask

        # ===== Prefill =====
        logits = self._prefill(
            inputs_embeds=inputs_embeds,
            position_ids=self._cached_position_ids,
            cache_position=cache_position,
            visual_pos_masks=image_mask[..., 0],
            deepstack_image_embeds=deepstack_image_embeds,
            streaming_attention_mask=self._cached_streaming_attention_mask,
        )

        # ===== Decode =====
        output_ids = input_ids.clone()
        if num_samples > 1:
            self._past_key_values.expand_batch()
            logits = logits.expand(num_samples, -1).contiguous()
            output_ids = output_ids.expand(num_samples, -1).contiguous()
        unfinished = torch.ones(batch_size * num_samples, dtype=torch.bool, device=device)
        cur_pos = cache_position[-1].item() + 1

        for _ in range(max_new_tokens):
            logits = logits_processor(output_ids, logits)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            next_token = torch.where(unfinished, next_token, self.tokenizer.pad_token_id)
            output_ids = torch.cat([output_ids, next_token.unsqueeze(-1)], dim=-1)

            unfinished = unfinished & (next_token != self.traj_start_token_id)
            if not unfinished.any():
                break

            logits = self._decode(
                input_ids=next_token.unsqueeze(-1),
                position_ids=self._cached_position_ids,
                cache_position=torch.tensor([cur_pos], device=device),
            )
            cur_pos += 1

        output_ids = replace_padding_after_eos(
            token_ids=output_ids,
            eos_token_id=self.traj_start_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Find <traj_future_start> position
        traj_start_pos = self._find_traj_start_positions(output_ids)
        truncation_pos = self._find_first_truncation_positions(output_ids, prompt_length=seq_len)
        if truncation_pos != -1:
            traj_start_pos = torch.min(traj_start_pos, truncation_pos)
            logger.info(f"traj_start_pos: {traj_start_pos.item()}, truncation_pos: {truncation_pos.item()}")
        streaming_input_len = seq_len
        # Clamp prompt conditioning to the first sentence in decoded text.
        first_period_pos = self._find_first_period_positions(
            output_ids=output_ids,
            start_pos=torch.full_like(traj_start_pos, streaming_input_len),
            end_pos=traj_start_pos,
        )
        traj_start_pos = first_period_pos

        # ===== Action (Diffusion) =====
        # Note: Action only attends to prompt tokens, NOT reasoning tokens (they are masked out).
        # This is by design - the expert model conditions only on the original prompt.

        # MODIFIED: Calculate offset for action tokens. In streaming setting, the offset is wrong without the modification due to truncated input length.
        # But the length of kv cache is always the same.
        if not self.is_first_prefill:
            action_start_pos = self.prefill_seq_length + (traj_start_pos - streaming_input_len) + 1
        else:
            action_start_pos = traj_start_pos + 1
        
        # Build position_ids for action tokens
        # position_ids = torch.arange(self.num_action_tokens, device=device)
        # position_ids = einops.repeat(position_ids, "t -> 3 b t", b=batch_size).clone()
        # position_ids += (self._cached_rope_deltas + action_start_pos[:, None]).to(device)

        # Build attention mask: attend to prompt only, mask out reasoning tokens
        indices = torch.arange(self._past_key_values.max_cache_len, device=device).expand(num_samples, -1)
        is_prompt = indices < action_start_pos[:, None]
        is_action = (indices >= cur_pos) & (indices < cur_pos + self.num_action_tokens)
        attention_mask = torch.where(
            (is_prompt | is_action)[:, None, None, :], 0.0, torch.finfo(torch.float32).min
        )

        # Cache positions for action tokens
        cache_position = torch.arange(
            cur_pos, cur_pos + self.num_action_tokens, device=device
        )

        sampled_action = self._action(
            num_action_tokens=self.num_action_tokens,
            total_samples=batch_size * num_samples,
            device=device,
            cache_position=cache_position,
            attention_mask=attention_mask,
            diffusion_kwargs=diffusion_kwargs,
        )

        # Convert actions to trajectories
        hist_xyz = einops.repeat(ego_history_xyz[:, -1], "b ... -> (b n) ...", n=num_samples)
        hist_rot = einops.repeat(ego_history_rot[:, -1], "b ... -> (b n) ...", n=num_samples)
        pred_xyz, pred_rot = self.action_space.action_to_traj(sampled_action, hist_xyz, hist_rot)
        pred_xyz = einops.rearrange(
            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )
        pred_rot = einops.rearrange(
            pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )

        # Update streaming state
        # self._crop_static_cache(self.prefill_seq_length)
        self._update_past_key_values()

        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, output_ids)
            for key in extra:
                extra[key] = np.array(extra[key]).reshape(
                    [batch_size, num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot


AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
AutoModel.register(AlpamayoR1Config, StreamingAlpamayoR1)
