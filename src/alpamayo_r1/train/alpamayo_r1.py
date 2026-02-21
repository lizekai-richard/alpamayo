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
from transformers import AutoConfig, AutoModel, StoppingCriteriaList
from transformers.generation.logits_process import (
    LogitsProcessor,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from alpamayo_r1.action_space import ActionSpace
from alpamayo_r1.models.base_model import ReasoningVLA
from alpamayo_r1.config import AlpamayoR1Config
from alpamayo_r1.diffusion.base import BaseDiffusion
from alpamayo_r1.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)
from alpamayo_r1.models.streaming_masking_utils import (
    create_streaming_attention_mask_sdpa,
    create_streaming_attention_mask_sdpa_training,
)
from alpamayo_r1.models.patches import StaticCache, patch_for_torch_compile
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast

logger = logging.getLogger(__name__)


class ExpertLogitsProcessor(LogitsProcessor):
    """Masks out the logits for discrete trajectory tokens."""

    def __init__(self, traj_token_offset: int, traj_vocab_size: int):
        """Initialize the ExpertLogitsProcessor.

        Args:
            traj_token_offset: The offset of the trajectory tokens.
            traj_vocab_size: The vocabulary size of the trajectory tokens.
        """
        super().__init__()
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Call the ExpertLogitsProcessor to mask out the logits for discrete trajectory tokens.

        The discrete trajectory tokens are not used for the expert model thus masking them out for
        better CoC generation.

        Args:
            input_ids: The input IDs.
            scores: The scores.

        Returns:
            torch.FloatTensor: The modified scores tensor with trajectory tokens masked out (set to -inf).
        """
        # Directly assign -inf to the trajectory token positions in the scores tensor
        scores[:, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size] = float('-inf')
        return scores


class AlpamayoR1(ReasoningVLA):
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
        self.num_views = 4
        self.num_frames_per_view = 4
        self.num_image_tokens_per_frame = 180

        # Streaming state (will be initialized on first call)
        self._past_key_values = None
        self._cached_position_ids = None
        self._cached_attention_mask = None
        self._cached_streaming_attention_mask = None
        self._cached_rope_deltas = None
        self._cached_keep_mask = None
        self.vision_start_end_ids_ranges = None
        self.image_token_ids_ranges = None
        self.traj_and_text_ids_range = None
        self.is_first_prefill = True

    def reset_streaming_state(self):
        """Reset all streaming state between batches."""
        self._past_key_values = None
        self._cached_position_ids = None
        self._cached_attention_mask = None
        self._cached_streaming_attention_mask = None
        self._cached_rope_deltas = None
        self._cached_keep_mask = None
        self.vision_start_end_ids_ranges = None
        self.image_token_ids_ranges = None
        self.traj_and_text_ids_range = None
        self.is_first_prefill = True

    def set_training_stage(self, stage: str):
        """Configure which modules are trainable based on training stage.

        Args:
            stage: "vlm" to fine-tune VLM (freeze expert/action/diffusion),
                   "expert" to fine-tune expert/action/diffusion (freeze VLM).
        """
        if stage == "vlm":
            for module in [self.expert, self.action_in_proj, self.action_out_proj, self.diffusion]:
                for param in module.parameters():
                    param.requires_grad = False
            for param in self.vlm.parameters():
                param.requires_grad = True
        elif stage == "expert":
            for param in self.vlm.parameters():
                param.requires_grad = False
            for module in [self.expert, self.action_in_proj, self.action_out_proj, self.diffusion]:
                for param in module.parameters():
                    param.requires_grad = True
        else:
            raise ValueError(f"Unknown training stage: {stage!r}. Expected 'vlm' or 'expert'.")

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
    
    def _create_cache_position_training(self, output_ids_range) -> torch.Tensor:
        """Create cache positions for training."""
        input_cache_position = self._create_cache_position()
        output_cache_position = torch.arange(output_ids_range[0], output_ids_range[1])
        return torch.cat([input_cache_position, output_cache_position], dim=0)

    def _get_streaming_attention_mask(
        self,
        cache_position: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Create streaming attention mask for non-first prefill."""
        return create_streaming_attention_mask_sdpa(
            batch_size=1,
            cache_position=cache_position,
            kv_length=self.max_cache_len,
            vision_start_end_ids_ranges=self.vision_start_end_ids_ranges,
            traj_and_text_ids_range=self.traj_and_text_ids_range,
            valid_length=self.prefill_seq_length,
            device=device,
            dtype=dtype,
        )
    
    def _get_streaming_attention_mask_training(
        self,
        cache_position: torch.Tensor,
        output_ids_range: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Create streaming attention mask for training."""
        return create_streaming_attention_mask_sdpa_training(
            batch_size=1,
            cache_position=cache_position,
            kv_length=self.max_cache_len,
            vision_start_end_ids_ranges=self.vision_start_end_ids_ranges,
            traj_and_text_ids_range=self.traj_and_text_ids_range,
            output_ids_range=output_ids_range,
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
        pixels = pixel_values.type(self.vlm.model.visual.dtype)
        return self.vlm.model.visual(pixels, grid_thw=image_grid_thw)

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

        hidden = self.vlm.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=self._past_key_values,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_image_embeds,
            streaming_attention_mask=streaming_attention_mask,
            use_cache=True,
        ).last_hidden_state[:, -1]
        return self.vlm.lm_head(hidden)

    def _decode(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        """Run decode forward pass for a single token."""
        hidden = self.vlm.model.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=self._past_key_values,
            cache_position=cache_position,
            use_cache=True,
        ).last_hidden_state[:, -1]
        return self.vlm.lm_head(hidden)

    def _action(
        self,
        num_action_tokens: int,
        total_samples: int,
        device: torch.device,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        diffusion_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:

        action_noise = torch.randn(
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
                attention_mask=attention_mask,
                cache_position=cache_position,
                use_cache=True,
                **expert_kwargs,
            ).last_hidden_state[:, -num_action_tokens:]
            return self.action_out_proj(hidden).view(-1, *action_dims)

        return self.diffusion.sample(
            noise=action_noise,
            batch_size=total_samples,
            step_fn=step_fn,
            device=device,
            return_all_steps=False,
            **(diffusion_kwargs or {}),
        )

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

    # ==================== Main Inference ====================

    # We don't need any output from the first prefill. Only need to cache key/values, position_ids, and attention_mask.
    def _first_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        device: torch.device,
    ):

        # Launch the first non-streaming prefill
        # inputs_embeds = self.vlm.model.get_input_embeddings()(input_ids)
        image_embeds, deepstack_image_embeds, _ = self._encode(pixel_values, image_grid_thw)

        inputs_embeds = self.vlm.model.get_input_embeddings()(input_ids)
        image_mask = (input_ids == self.vlm.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        position_ids, rope_deltas = self.vlm.model.get_rope_index(
            input_ids, image_grid_thw, None, None
        )
        padding_length = self.max_cache_len - input_ids.shape[1]
        if padding_length > 0:
            last_pos = position_ids[:, :, -1:]
            padding_pos = last_pos + torch.arange(1, padding_length + 1, device=device)
            position_ids = torch.cat([position_ids, padding_pos], dim=-1)
        
        # Cache all streaming related inputs
        (
            vision_start_end_ids_ranges, 
            image_token_ids_ranges, 
            traj_and_text_ids_range
        ) = self._retrieve_streaming_related_inputs(input_ids[:1])
        cache_position = self._create_cache_position().to(device)

        self._cached_position_ids = position_ids
        self._cached_rope_deltas = rope_deltas
        self._cached_attention_mask = attention_mask
        self.vision_start_end_ids_ranges = vision_start_end_ids_ranges
        self.image_token_ids_ranges = image_token_ids_ranges
        self.traj_and_text_ids_range = traj_and_text_ids_range

        self._prefill(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            cache_position=cache_position,
            visual_pos_masks=image_mask[..., 0],
            deepstack_image_embeds=deepstack_image_embeds,
            streaming_attention_mask=None,
        )

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
        sparsity_ratio: float = 0.5,
        rope_mode: str = "contiguous",
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
            rope_mode: mRoPE mode for pruned tokens ("contiguous", "direct", "reshape").
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

        if self.is_first_prefill:
            self.prefill_seq_length = input_ids.shape[1]
            self.max_cache_len = input_ids.shape[1] + max_new_tokens + self.num_action_tokens

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
            logits = self._first_prefill(input_ids, attention_mask, pixel_values, image_grid_thw, device)
            self.is_first_prefill = False
        
        else:
            # ===== Encode =====
            image_embeds, deepstack_image_embeds, _ = self._encode(pixel_values, image_grid_thw)
            
            inputs_embeds = self.vlm.model.get_input_embeddings()(input_ids)

            # Create cache position and position IDs
            cache_position = self._create_cache_position().to(device)
            seq_len = input_ids.shape[1]
            self._cached_prompt_length = seq_len

            # Get embeddings
            # inputs_embeds = self.vlm.model.get_input_embeddings()(input_ids)
            image_mask = (input_ids == self.vlm.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # Create streaming attention mask for non-first prefill
            if self._cached_streaming_attention_mask is None:
                self._cached_streaming_attention_mask = self._get_streaming_attention_mask(
                    cache_position=cache_position,
                    device=device,
                )

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
        streaming_input_len = seq_len

        # ===== Action (Diffusion) =====
        # Note: Action only attends to prompt tokens, NOT reasoning tokens (they are masked out).
        # This is by design - the expert model conditions only on the original prompt.

        # MODIFIED: Calculate offset for action tokens. In streaming setting, the offset is wrong without the modification due to truncated input length.
        # But the length of kv cache is always the same.
        if not self.is_first_prefill:
            action_start_pos = self.prefill_seq_length + (traj_start_pos - streaming_input_len) + 1
        else:
            action_start_pos = traj_start_pos + 1

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
        self._update_past_key_values()

        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, output_ids)
            for key in extra:
                extra[key] = np.array(extra[key]).reshape(
                    [batch_size, num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot
    
    def _forward_vlm_training(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        pixel_values: torch.Tensor, 
        image_grid_thw: torch.Tensor,
        ego_history_xyz: torch.Tensor,
        ego_history_rot: torch.Tensor,
        output_ids_range: tuple[int, int],
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for the model.

        Args:
            input_ids: The input IDs.
            attention_mask: The attention mask.
            pixel_values: The pixel values.
            image_grid_thw: The image grid.
            ego_history_xyz: The ego history xyz.
            ego_history_rot: The ego history rot.
            labels: The labels.
        """
        input_ids = self.fuse_traj_tokens(input_ids, {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot})
        device = input_ids.device
        cache_position = self._create_cache_position_training(output_ids_range).to(device)
        streaming_attention_mask = self._get_streaming_attention_mask_training(
            cache_position, output_ids_range, device=device,
        )
        vlm_outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            past_key_values=self._past_key_values,
            cache_position=cache_position,
            labels=labels,
            streaming_attention_mask=streaming_attention_mask,
        )
        
        return vlm_outputs
    
    def _forward_expert_training(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        top_p: float,
        ego_history_xyz: torch.Tensor,
        ego_history_rot: torch.Tensor,
        ego_future_xyz: torch.Tensor,
        ego_future_rot: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = input_ids.shape[0]
        device = input_ids.device
        logits_processor = self._build_logits_processor(temperature, top_k, top_p)

        # Convert ground-truth trajectory to action space: (B, 64, 2)
        action_labels = self.action_space.traj_to_action(
            ego_history_xyz, ego_history_rot, ego_future_xyz, ego_future_rot,
        )

        num_action_tokens = self.num_action_tokens
        action_dims = self.action_space.get_action_space_dims()
        expert_kwargs = {"is_causal": False} if self.config.expert_non_causal_attention else {}

        # ===== VLM Encode + Prefill + Decode (no grad, only building KV cache) =====
        with torch.no_grad():
            image_embeds, deepstack_image_embeds, _ = self._encode(pixel_values, image_grid_thw)
            inputs_embeds = self.vlm.model.get_input_embeddings()(input_ids)

            cache_position = self._create_cache_position().to(device)
            seq_len = input_ids.shape[1]

            image_mask = (input_ids == self.vlm.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if self._cached_streaming_attention_mask is None:
                self._cached_streaming_attention_mask = self._get_streaming_attention_mask(
                    cache_position=cache_position,
                    device=device,
                )

            logits = self._prefill(
                inputs_embeds=inputs_embeds,
                position_ids=self._cached_position_ids,
                cache_position=cache_position,
                visual_pos_masks=image_mask[..., 0],
                deepstack_image_embeds=deepstack_image_embeds,
                streaming_attention_mask=self._cached_streaming_attention_mask,
            )

            # Autoregressive decode to generate CoT and populate KV cache
            logits_processor = self._build_logits_processor(temperature, top_k, top_p)
            output_ids = input_ids.clone()
            unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
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

            # Build expert attention mask: attend to prompt only + action tokens
            traj_start_pos = self._find_traj_start_positions(output_ids)
            action_start_pos = self.prefill_seq_length + (traj_start_pos - seq_len) + 1

            indices = torch.arange(self._past_key_values.max_cache_len, device=device).expand(batch_size, -1)
            is_prompt = indices < action_start_pos[:, None]
            is_action = (indices >= cur_pos) & (indices < cur_pos + num_action_tokens)
            expert_attention_mask = torch.where(
                (is_prompt | is_action)[:, None, None, :], 0.0, torch.finfo(torch.float32).min
            )

            expert_cache_position = torch.arange(
                cur_pos, cur_pos + num_action_tokens, device=device
            )

        # ===== Flow Matching Training (with grad) =====
        # Sample random timestep t ~ U(0, 1)
        t = torch.rand(batch_size, 1, 1, device=device, dtype=action_labels.dtype)

        # Sample noise
        noise = torch.randn_like(action_labels)

        # Interpolate: x_t = (1 - t) * noise + t * data
        x_t = (1 - t) * noise + t * action_labels

        # Target velocity: v = data - noise
        v_target = action_labels - noise

        # Expert forward: predict velocity
        action_embeds = self.action_in_proj(x_t, t)
        if action_embeds.dim() == 2:
            action_embeds = action_embeds.view(batch_size, num_action_tokens, -1)

        hidden = self.expert(
            inputs_embeds=action_embeds,
            position_ids=self._cached_position_ids,
            past_key_values=self._past_key_values,
            attention_mask=expert_attention_mask,
            cache_position=expert_cache_position,
            use_cache=True,
            **expert_kwargs,
        ).last_hidden_state[:, -num_action_tokens:]

        v_pred = self.action_out_proj(hidden).view(batch_size, *action_dims)

        # Flow matching loss: MSE(predicted velocity, target velocity)
        loss = torch.nn.functional.mse_loss(v_pred, v_target)
        return loss


AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
AutoModel.register(AlpamayoR1Config, AlpamayoR1)
