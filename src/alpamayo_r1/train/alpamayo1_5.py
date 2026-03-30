# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from functools import partial
import logging
from typing import Any

import einops
import hydra.utils as hyu
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteriaList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from alpamayo_r1.action_space import ActionSpace
from alpamayo_r1.models.base_model import ReasoningVLA
from alpamayo_r1.config import Alpamayo1_5Config
from alpamayo_r1.diffusion.base import BaseDiffusion
from alpamayo_r1.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)
from alpamayo_r1.utils.streaming.streaming_masking_utils import (
    create_streaming_attention_mask_sdpa_v1p5,
    create_streaming_attention_mask_sdpa_training,
)
from alpamayo_r1.train.patches import StaticCache
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast
from alpamayo_r1.nav_utils import remove_nav_text

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
        scores[:, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size] = float(
            "-inf"
        )
        return scores


class Alpamayo1_5(ReasoningVLA):
    """Expert model for reasoning VLA."""

    config_class: type[Alpamayo1_5Config] = Alpamayo1_5Config
    base_model_prefix = "vlm"

    def __init__(
        self,
        config: Alpamayo1_5Config,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
    ):
        super().__init__(config, pretrained_modules, original_vocab_size, print_param_count=False)

        # we only need the text config for the expert model
        expert_config = copy.deepcopy(self.vlm.config.text_config)
        if config.expert_cfg is not None:
            for key, value in config.expert_cfg.items():
                setattr(expert_config, key, value)
        self.expert = AutoModel.from_config(expert_config)
        # we don't need the embed_tokens of the expert model
        del self.expert.embed_tokens

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
        self.vision_start_end_ids_ranges = None
        self.image_token_ids_ranges = None
        self.traj_and_text_ids_range = None
        self.is_first_prefill = True
        self.keep_frame_labels = False
        self.kv_shift_mode = "vision_only"  # "block" or "vision_only"
    
    def reset_streaming_state(self, kv_shift_mode: str = "vision_only", keep_frame_labels: bool = False):
        """Reset all streaming state between batches."""
        self._past_key_values = None
        self._cached_position_ids = None
        self._cached_attention_mask = None
        self._cached_streaming_attention_mask = None
        self._cached_rope_deltas = None
        self.vision_start_end_ids_ranges = None
        self.image_token_ids_ranges = None
        self.traj_and_text_ids_range = None
        self.is_first_prefill = True
        self.keep_frame_labels = keep_frame_labels
        self.kv_shift_mode = kv_shift_mode
        assert self.kv_shift_mode in ["block", "vision_only"], "Invalid kv_shift_mode"
    
    
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

    @staticmethod
    def _find_eos_offset(
        sequences: torch.Tensor,
        eos_token_id: int,
        device: torch.device,
        warn: bool = False,
    ) -> torch.Tensor:
        """Find the first eos_token_id position in each sequence and return offset = pos + 1.

        Falls back to the last token position when eos_token_id is not found.
        The returned offset marks the boundary between VLM-generated tokens and
        the region where expert diffusion tokens will be appended.
        """
        b_star = sequences.shape[0]
        mask = sequences == eos_token_id
        has_eos = mask.any(dim=1)  # [b_star]
        if warn:
            for i in range(b_star):
                if not has_eos[i]:
                    logger.warning(
                        f"No <traj_future_start> token found in generated sequences"
                        f" for sequence {i}"
                    )
        eos_positions = mask.int().argmax(dim=1)  # [b_star], first occurrence
        last_positions = torch.full((b_star,), sequences.shape[1] - 1, device=device)
        return torch.where(has_eos, eos_positions, last_positions) + 1

    @staticmethod
    def _build_expert_pos_ids_and_attn_mask(
        offset: torch.Tensor,
        rope_deltas: torch.Tensor,
        kv_cache_seq_len: int,
        n_diffusion_tokens: int,
        b_star: int,
        device: torch.device,
        prefix_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build position IDs and 4D attention mask for the expert denoiser.

        Args:
            offset: [b_star] — token position right after <traj_future_start>.
            rope_deltas: [b_star, 1] — RoPE delta from the VLM.
            kv_cache_seq_len: sequence length already in the KV cache.
            n_diffusion_tokens: number of expert diffusion tokens to append.
            b_star: batch size (B * num_return_sequences).
            device: torch device.
            prefix_mask: [b_star, L] optional 1D attention mask (already repeated
                to match b_star); zeros mark padding positions that should be
                masked in the expert's cross-attention to the KV cache.

        Returns:
            position_ids: [3, b_star, n_diffusion_tokens] — Qwen2.5-VL RoPE ids.
            attention_mask: [b_star, 1, n_diffusion_tokens, KV] — 4D float mask
                (0 = attend, -inf = masked).
        """
        # Qwen2.5-VL uses 3-component (temporal, height, width) RoPE
        position_ids = torch.arange(n_diffusion_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
        position_ids += (rope_deltas + offset[:, None]).to(position_ids.device)

        # [b_star, H, Q, KV] — mask the gap between offset and diffusion tokens
        attention_mask = torch.zeros(
            (b_star, 1, n_diffusion_tokens, kv_cache_seq_len + n_diffusion_tokens),
            dtype=torch.float32,
            device=device,
        )
        for i in range(b_star):
            attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = torch.finfo(
                attention_mask.dtype
            ).min

        # Propagate input padding mask (left-padding) into the KV prefix region
        if prefix_mask is not None:
            # [b_star, H, Q, KV]
            input_mask = prefix_mask[:, None, None, :]
            attention_mask[:, :, :, : input_mask.shape[-1]] = torch.where(
                input_mask == 0,
                torch.finfo(attention_mask.dtype).min,
                attention_mask[:, :, :, : input_mask.shape[-1]],
            )

        return position_ids, attention_mask
    
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

        # Optionally expand last frame's range to start from sec-last VE+1
        # (includes frame label text in v1.5). In v1 sec_last_VE+1 == last_VS,
        # so this is a no-op. When keep_frame_labels=False, skip expansion.
        if self.keep_frame_labels:
            for view_idx in range(self.num_views):
                sec_last_end = vision_start_end_ids_ranges[view_idx][-2][1]
                last_end = vision_start_end_ids_ranges[view_idx][-1][1]
                vision_start_end_ids_ranges[view_idx][-1] = (sec_last_end, last_end)

        last_vision_end_id = all_vision_end_token_ids[-1]
        traj_and_text_ids_range = (last_vision_end_id.item() + 1, self.prefill_seq_length)

        return vision_start_end_ids_ranges, image_token_ids_ranges, traj_and_text_ids_range

    def _update_past_key_values(self):
        """Shift KV cache: move frames 1-3 to positions 0-2.

        Controlled by self.kv_shift_mode:
        - "block": copy [frame1_start..frame3_end) → [frame0_start..frame2_end),
          shifting frame labels together with vision tokens as a unit.
        - "vision_only": per-frame copy of [VS..VE+1] only, skipping text
          tokens between frames so they stay in place.
        """
        for layer in self._past_key_values.layers:
            key_cache = layer.keys
            value_cache = layer.values

            if self.kv_shift_mode == "block":
                for i in range(self.num_views):
                    # Compute frame label length from gap between frame 0 and frame 1
                    frame0_vs = self.vision_start_end_ids_ranges[i][0][0]
                    frame0_ve_plus1 = self.vision_start_end_ids_ranges[i][0][1]
                    label_len = self.vision_start_end_ids_ranges[i][1][0] - frame0_ve_plus1

                    # Extend range to include frame 0's label
                    new_kv_start = frame0_vs - label_len
                    new_kv_end = self.vision_start_end_ids_ranges[i][-2][1]
                    old_kv_start = frame0_ve_plus1  # = frame 1's label start
                    old_kv_end = self.vision_start_end_ids_ranges[i][-1][1]

                    key_cache[:, :, new_kv_start:new_kv_end, :].copy_(
                        key_cache[:, :, old_kv_start:old_kv_end, :].clone()
                    )
                    value_cache[:, :, new_kv_start:new_kv_end, :].copy_(
                        value_cache[:, :, old_kv_start:old_kv_end, :].clone()
                    )
            else:  # vision_only
                for i in range(self.num_views):
                    for k in range(self.num_frames_per_view - 1):
                        dst_start, dst_end = self.image_token_ids_ranges[i][k]
                        src_start, src_end = self.image_token_ids_ranges[i][k + 1]
                        dst_start -= 1
                        dst_end += 1
                        src_start -= 1
                        src_end += 1

                        key_cache[:, :, dst_start:dst_end, :].copy_(
                            key_cache[:, :, src_start:src_end, :].clone()
                        )
                        value_cache[:, :, dst_start:dst_end, :].copy_(
                            value_cache[:, :, src_start:src_end, :].clone()
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
        """Create cache positions for training.

        output_ids_range is the relative range within the window's input_ids.
        The absolute cache positions for output tokens start at prefill_seq_length.
        """
        input_cache_position = self._create_cache_position()
        output_len = output_ids_range[1] - output_ids_range[0]
        output_cache_position = torch.arange(
            self.prefill_seq_length, self.prefill_seq_length + output_len
        )
        return torch.cat([input_cache_position, output_cache_position], dim=0)

    def _get_streaming_attention_mask(
        self,
        cache_position: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Create streaming attention mask for non-first prefill."""
        return create_streaming_attention_mask_sdpa_v1p5(
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
        image_embeds, deepstack_image_embeds = self._encode(pixel_values, image_grid_thw)

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

        logits = self._prefill(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            cache_position=cache_position,
            visual_pos_masks=image_mask[..., 0],
            deepstack_image_embeds=deepstack_image_embeds,
            streaming_attention_mask=None,
        )

        return logits

    # ==================== Unified Training Forward ====================

    def forward(
        self,
        batch: list[dict[str, Any]],
        device: torch.device | None = None,
        training_stage: str = "vlm",
        max_generation_length: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.98,
    ) -> dict[str, torch.Tensor]:
        """Unified training forward: rollout (no grad) + loss on last window (with grad).

        This is the standard nn.Module.forward() so DDP/FSDP gradient hooks work.

        Args:
            batch: List of window dicts from the dataloader.
            device: Target device. If None, inferred from model parameters.
            training_stage: "vlm" or "expert".
            max_generation_length: Max CoT tokens to generate during rollout.
            temperature: Sampling temperature for rollout generation.
            top_p: Top-p sampling parameter for rollout generation.

        Returns:
            Dict with "loss" key (and optionally other outputs).
        """
        if device is None:
            device = next(self.parameters()).device

        self.reset_streaming_state()

        # ===== Rollout: populate streaming KV cache (no grad) =====
        with torch.inference_mode():
            for window in batch[:-1]:
                self._forward_rollout_window(window, device, max_generation_length, temperature, top_p)

        # ===== Training: compute loss on last window (with grad) =====
        if training_stage == "vlm":
            vlm_output = self._forward_vlm_last_window(batch[-1], device)
            return {"loss": vlm_output.loss, "rollout_steps": len(batch) - 1}
        else:  # expert
            loss = self._forward_expert_last_window(batch[-1], device, max_generation_length, temperature, top_p)
            return {"loss": loss, "rollout_steps": len(batch) - 1}

    def _forward_rollout_window(
        self,
        window: dict[str, Any],
        device: torch.device,
        max_generation_length: int,
        temperature: float,
        top_p: float,
    ):
        """Run a single rollout window to populate streaming KV cache."""
        input_ids = window["input_ids"].to(device)
        attention_mask = window["attention_mask"].to(device)
        pixel_values = window["pixel_values"].to(device)
        image_grid_thw = window["image_grid_thw"].to(device)
        ego_history_xyz = window["ego_history_xyz"].to(device)
        ego_history_rot = window["ego_history_rot"].to(device)

        batch_size = input_ids.shape[0]

        input_ids = self.fuse_traj_tokens(
            input_ids, {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot}
        )

        if self.is_first_prefill:
            self.prefill_seq_length = input_ids.shape[1]
            self.max_cache_len = input_ids.shape[1] + max_generation_length + self.num_action_tokens

        # Initialize KV cache on first call
        if self._past_key_values is None:
            self._past_key_values = StaticCache(
                config=self.vlm.config,
                max_cache_len=self.max_cache_len,
                max_batch_size=batch_size,
                offloading=False,
            )

        if self.is_first_prefill:
            self._first_prefill(input_ids, attention_mask, pixel_values, image_grid_thw, device)
            self._update_past_key_values()
            self.is_first_prefill = False
            return

        # ===== Streaming prefill =====
        image_embeds, deepstack_image_embeds = self._encode(pixel_values, image_grid_thw)
        inputs_embeds = self.vlm.model.get_input_embeddings()(input_ids)
        cache_position = self._create_cache_position().to(device)

        image_mask = (input_ids == self.vlm.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if self._cached_streaming_attention_mask is None:
            self._cached_streaming_attention_mask = self._get_streaming_attention_mask(
                cache_position=cache_position, device=device,
            )

        logits = self._prefill(
            inputs_embeds=inputs_embeds,
            position_ids=self._cached_position_ids,
            cache_position=cache_position,
            visual_pos_masks=image_mask[..., 0],
            deepstack_image_embeds=deepstack_image_embeds,
            streaming_attention_mask=self._cached_streaming_attention_mask,
        )

        # Update streaming state (shift KV cache frames)
        self._update_past_key_values()

    def _forward_vlm_last_window(
        self,
        window: dict[str, Any],
        device: torch.device,
    ) -> Qwen3VLCausalLMOutputWithPast:
        """Run VLM training on the last window."""
        input_ids = window["input_ids"].to(device)
        attention_mask = window["attention_mask"].to(device)
        pixel_values = window["pixel_values"].to(device)
        image_grid_thw = window["image_grid_thw"].to(device)
        ego_history_xyz = window["ego_history_xyz"].to(device)
        ego_history_rot = window["ego_history_rot"].to(device)
        output_ids_range = window["output_ids_range"]
        labels = window["labels"].to(device)

        input_ids = self.fuse_traj_tokens(
            input_ids, {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot}
        )

        # Clone StaticCache keys/values from inference tensors to normal tensors
        # so they can be updated in-place during the training forward pass.
        for layer in self._past_key_values.layers:
            if layer.is_initialized:
                layer.keys = layer.keys.clone()
                layer.values = layer.values.clone()

        cache_position = self._create_cache_position_training(output_ids_range).to(device)
        # Convert output_ids_range from input_ids coordinates to KV cache coordinates
        output_len = output_ids_range[1] - output_ids_range[0]
        output_kv_range = (self.prefill_seq_length, self.prefill_seq_length + output_len)
        streaming_attention_mask = self._get_streaming_attention_mask_training(
            cache_position, output_kv_range, device=device,
        )

        return self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=self._cached_position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            past_key_values=self._past_key_values,
            cache_position=cache_position,
            labels=labels,
            streaming_attention_mask=streaming_attention_mask,
        )
    
    def _forward_expert_last_window(
        self,
        window: dict[str, Any],
        device: torch.device,
        max_generation_length: int,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        """Expert training on the last window: VLM prefill+decode (no grad) + flow matching (with grad)."""
        input_ids = window["input_ids"].to(device)
        pixel_values = window["pixel_values"].to(device)
        image_grid_thw = window["image_grid_thw"].to(device)
        ego_history_xyz = window["ego_history_xyz"].to(device)
        ego_history_rot = window["ego_history_rot"].to(device)
        ego_future_xyz = window["ego_future_xyz"].to(device)
        ego_future_rot = window["ego_future_rot"].to(device)

        batch_size = input_ids.shape[0]

        input_ids = self.fuse_traj_tokens(
            input_ids, {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot}
        )

        # ===== VLM prefill + decode (no grad) to build KV cache =====
        with torch.inference_mode():
            image_embeds, deepstack_image_embeds = self._encode(pixel_values, image_grid_thw)
            inputs_embeds = self.vlm.model.get_input_embeddings()(input_ids)
            cache_position = self._create_cache_position().to(device)
            seq_len = input_ids.shape[1]

            image_mask = (input_ids == self.vlm.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if self._cached_streaming_attention_mask is None:
                self._cached_streaming_attention_mask = self._get_streaming_attention_mask(
                    cache_position=cache_position, device=device,
                )

            logits = self._prefill(
                inputs_embeds=inputs_embeds,
                position_ids=self._cached_position_ids,
                cache_position=cache_position,
                visual_pos_masks=image_mask[..., 0],
                deepstack_image_embeds=deepstack_image_embeds,
                streaming_attention_mask=self._cached_streaming_attention_mask,
            )

            # Decode CoT
            logits_processor = self._build_logits_processor(temperature, None, top_p)
            output_ids = input_ids.clone()
            unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
            cur_pos = cache_position[-1].item() + 1

            for _ in range(max_generation_length):
                logits = logits_processor(output_ids, logits)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                next_token = torch.where(unfinished, next_token, self.tokenizer.pad_token_id)
                output_ids = torch.cat([output_ids, next_token.unsqueeze(-1)], dim=-1)
                unfinished = unfinished & (next_token != self.traj_start_token_id)
                if not unfinished.any():
                    self._decode(
                        input_ids=next_token.unsqueeze(-1),
                        position_ids=self._cached_position_ids,
                        cache_position=torch.tensor([cur_pos], device=device),
                    )
                    cur_pos += 1
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

            # Compute scalar values needed for expert mask construction
            # traj_start_pos = self._find_traj_start_positions(output_ids)
            # _find_eos_offset returns pos+1, subtract 1 to get actual position
            traj_start_pos = self._find_eos_offset(output_ids, self.traj_start_token_id, device) - 1
            action_start_pos = self.prefill_seq_length + (traj_start_pos - seq_len) + 1
            # Save as python ints so they can be used outside inference_mode
            action_start_pos_val = action_start_pos.cpu().tolist()
            cur_pos_val = cur_pos

        # Build expert attention mask and cache position (outside inference_mode)
        num_action_tokens = self.num_action_tokens
        action_start_pos = torch.tensor(action_start_pos_val, device=device)
        indices = torch.arange(self._past_key_values.max_cache_len, device=device).expand(batch_size, -1)
        is_prompt = indices < action_start_pos[:, None]
        is_action = (indices >= cur_pos_val) & (indices < cur_pos_val + num_action_tokens)
        expert_attention_mask = torch.where(
            (is_prompt | is_action)[:, None, None, :], 0.0, torch.finfo(torch.float32).min
        )
        expert_cache_position = torch.arange(cur_pos_val, cur_pos_val + num_action_tokens, device=device)

        # Re-create StaticCache keys/values as normal tensors (were created under inference_mode)
        for layer in self._past_key_values.layers:
            if layer.is_initialized:
                layer.keys = layer.keys.clone()
                layer.values = layer.values.clone()

        # Squeeze num_traj_groups dim (always 1) so shapes are (B, T, ...)
        action_labels = self.action_space.traj_to_action(
            ego_history_xyz.squeeze(1), ego_history_rot.squeeze(1),
            ego_future_xyz.squeeze(1), ego_future_rot.squeeze(1),
        )
        action_dims = self.action_space.get_action_space_dims()
        expert_kwargs = {"is_causal": False} if self.config.expert_non_causal_attention else {}

        t = torch.rand(batch_size, 1, 1, device=device, dtype=action_labels.dtype)
        noise = torch.randn_like(action_labels)
        x_t = (1 - t) * noise + t * action_labels
        v_target = action_labels - noise

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
        return torch.nn.functional.mse_loss(v_pred, v_target)
    
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
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample trajectories from the data with VLM rollout.

        Args:
            data: The input data.
            top_p: The top-p value for sampling.
            top_k: The top-k value for sampling.
            temperature: The temperature for sampling.
            num_traj_samples: The number of trajectory samples.
            num_traj_sets: The number of trajectory sets.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pred_xyz: The predicted xyz.
            pred_rot: The predicted rotation.
            logprob: The log probability.
        """
        data = copy.deepcopy(data)
        n_samples_total = num_traj_samples * num_traj_sets
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        tokenized_data = data["tokenized_data"]
        pixel_values = data.get("pixel_values", tokenized_data.get("pixel_values"))
        image_grid_thw = data.get("image_grid_thw", tokenized_data.get("image_grid_thw"))
        batch_size, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."
        input_ids = tokenized_data.pop("input_ids")
        attention_mask = tokenized_data.pop("attention_mask")
        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
        device = input_ids.device

        # 1) run autoregressive generation for the VLM
        max_generation_length = kwargs.get(
            "max_generation_length", self.config.tokens_per_future_traj
        )
        logits_processor = self._build_logits_processor(temperature, top_k, top_p)

        if self.is_first_prefill:
            self.prefill_seq_length = input_ids.shape[1]
            self.max_cache_len = self.prefill_seq_length + max_generation_length + self.num_action_tokens
        
        # Initialize KV cache on first call
        if self._past_key_values is None:
            self._past_key_values = StaticCache(
                config=self.vlm.config,
                max_cache_len=self.max_cache_len,
                max_batch_size=n_samples_total * batch_size,
                offloading=False,
            )
        
        if self.is_first_prefill:
            logger.info("First prefill: caching KV and returning (no streaming logs yet).")
            logits = self._first_prefill(input_ids, attention_mask, pixel_values, image_grid_thw, device)
            self._update_past_key_values()
            self.is_first_prefill = False
            return

        image_embeds, deepstack_image_embeds = self._encode(pixel_values, image_grid_thw)
        inputs_embeds = self.vlm.model.get_input_embeddings()(input_ids)
        cache_position = self._create_cache_position().to(device)

        image_mask = (input_ids == self.vlm.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if self._cached_streaming_attention_mask is None:
            self._cached_streaming_attention_mask = self._get_streaming_attention_mask(
                cache_position=cache_position, device=device,
            )
        
        logits = self._prefill(
            inputs_embeds=inputs_embeds,
            position_ids=self._cached_position_ids,
            cache_position=cache_position,
            visual_pos_masks=image_mask[..., 0],
            deepstack_image_embeds=deepstack_image_embeds,
            streaming_attention_mask=self._cached_streaming_attention_mask,
        )

        output_ids = input_ids.clone()
        if n_samples_total > 1:
            self._past_key_values.expand_batch()
            logits = logits.expand(n_samples_total, -1).contiguous()
            output_ids = output_ids.expand(n_samples_total, -1).contiguous()
        unfinished = torch.ones(batch_size * n_samples_total, dtype=torch.bool, device=device)
        cur_pos = cache_position[-1].item() + 1

        for _ in range(max_generation_length):
            logits = logits_processor(output_ids, logits)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            next_token = torch.where(unfinished, next_token, self.tokenizer.pad_token_id)
            output_ids = torch.cat([output_ids, next_token.unsqueeze(-1)], dim=-1)

            unfinished = unfinished & (next_token != self.traj_start_token_id)
            if not unfinished.any():
                self._decode(
                    input_ids=next_token.unsqueeze(-1),
                    position_ids=self._cached_position_ids,
                    cache_position=torch.tensor([cur_pos], device=device),
                )
                cur_pos += 1
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
        # _find_eos_offset returns pos+1, so subtract 1 to get the actual position
        traj_start_pos = self._find_eos_offset(output_ids, self.traj_start_token_id, device) - 1

        # MODIFIED: Calculate offset for action tokens. In streaming setting, the offset is wrong without the modification due to truncated input length.
        # But the length of kv cache is always the same.
        action_start_pos = self.prefill_seq_length + (traj_start_pos - input_ids.shape[1]) + 1

        # Build position_ids for action tokens
        # Build attention mask: attend to prompt only, mask out reasoning tokens
        indices = torch.arange(self._past_key_values.max_cache_len, device=device).expand(n_samples_total, -1)
        is_prompt = indices < action_start_pos[:, None]
        is_action = (indices >= cur_pos) & (indices < cur_pos + self.num_action_tokens)
        attention_mask = torch.where(
            (is_prompt | is_action)[:, None, None, :], 0.0, torch.finfo(torch.bfloat16).min
        ).to(torch.bfloat16)

        # Cache positions for action tokens
        cache_position = torch.arange(
            cur_pos, cur_pos + self.num_action_tokens, device=device
        )

        sampled_action = self._action(
            num_action_tokens=self.num_action_tokens,
            total_samples=batch_size * n_samples_total,
            device=device,
            cache_position=cache_position,
            attention_mask=attention_mask,
            diffusion_kwargs=diffusion_kwargs,
        )

        # Convert actions to trajectories
        hist_xyz = einops.repeat(ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total)
        hist_rot = einops.repeat(ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total)
        pred_xyz, pred_rot = self.action_space.action_to_traj(sampled_action, hist_xyz, hist_rot)
        pred_xyz = einops.rearrange(
            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )
        pred_rot = einops.rearrange(
            pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )

        # Update streaming state: zero out decode+action tokens before shifting
        self._crop_static_cache(self.prefill_seq_length)
        self._update_past_key_values()

        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, output_ids)
            for key in extra:
                extra[key] = np.array(extra[key]).reshape(
                    [batch_size, num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot


AutoConfig.register("alpamayo1_5", Alpamayo1_5Config)
AutoModel.register(Alpamayo1_5Config, Alpamayo1_5)
