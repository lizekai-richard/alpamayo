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

"""
Unified compiled model supporting both streaming and non-streaming inference modes.

This module provides a single model class that can operate in two modes:
- Streaming mode: Reuses KV cache across frames, only processes new frames after initial prefill
- Non-streaming mode: Resets KV cache each call, processes all frames every time

Both modes use torch.compile with CUDA graphs for optimized inference.

Usage:
    model = AlpamayoR1.from_pretrained("./Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")

    # Streaming mode
    result = model.sample_trajectories_from_data_with_streaming_vlm_rollout(data)

    # Non-streaming mode
    result = model.sample_trajectories_from_data_with_vlm_rollout(data)
"""

import copy
import logging
from typing import Any

import einops
import hydra.utils as hyu
import numpy as np
import torch
from transformers import AutoConfig, AutoModel
from alpamayo_r1.models.patches import StaticCache
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
from alpamayo_r1.models.patches import patch_for_torch_compile
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
logger.setLevel(logging.INFO)


class ExpertLogitsProcessor(LogitsProcessor):
    """Masks out the logits for discrete trajectory tokens."""

    def __init__(self, traj_token_offset: int, traj_vocab_size: int):
        super().__init__()
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size] = float("-inf")
        return scores


class AlpamayoR1(ReasoningVLA):
    """
    Unified compiled model supporting both streaming and non-streaming modes.

    This class provides a single interface for inference with two modes:
    - streaming=True: Efficient streaming inference with KV cache reuse
    - streaming=False: Standard inference with fresh KV cache each call

    Both modes use torch.compile with CUDA graphs for optimal performance.
    """

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
        self.traj_and_text_ids_range = None
        self.is_first_prefill = True

        # Compile mode
        self._torch_compile = "max-autotune"

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
            traj_and_text_ids_range: (start, end) tuple for trajectory and text tokens
        """
        vision_start_end_ids_ranges = [[] for _ in range(self.num_views)]

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

        last_vision_end_id = all_vision_end_token_ids[-1]
        traj_and_text_ids_range = (last_vision_end_id.item() + 1, self.prefill_seq_length)

        return vision_start_end_ids_ranges, traj_and_text_ids_range

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
            self._encode_fn()  # Warmup
            self._compiled_encode_fn = torch.compile(
                self._encode_fn, mode=self._torch_compile, fullgraph=True
            )

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
        """
        Run prefill forward pass.

        Args:
            streaming_attention_mask: If None, use standard causal mask.
                                      If provided, use streaming attention pattern.
        """
        if not hasattr(self, "_prefill_fn"):
            self._prefill_inputs_embeds = torch.empty_like(inputs_embeds)
            self._prefill_position_ids = torch.empty_like(position_ids)
            self._prefill_cache_position = torch.empty_like(cache_position)
            self._prefill_visual_pos_masks = torch.empty_like(visual_pos_masks)
            self._prefill_deepstack_embeds = [torch.empty_like(e) for e in deepstack_image_embeds]

            # Handle streaming attention mask
            if streaming_attention_mask is not None:
                self._prefill_streaming_attention_mask = torch.empty_like(streaming_attention_mask)
            else:
                self._prefill_streaming_attention_mask = None

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
        for buf, emb in zip(self._prefill_deepstack_embeds, deepstack_image_embeds):
            buf.copy_(emb)

        if streaming_attention_mask is not None and self._prefill_streaming_attention_mask is not None:
            self._prefill_streaming_attention_mask.copy_(streaming_attention_mask)

        if not hasattr(self, "_compiled_prefill_fn"):
            self._prefill_fn()  # Warmup
            self._compiled_prefill_fn = torch.compile(
                self._prefill_fn, mode=self._torch_compile, fullgraph=True
            )

        torch.compiler.cudagraph_mark_step_begin()
        return self._compiled_prefill_fn()

    def _decode(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        mode: str = "streaming",
    ) -> torch.Tensor:
        """Run decode forward pass for a single token.

        Args:
            mode: "streaming" or "non_streaming" - determines which set of compiled
                  functions and static buffers to use.
        """
        # Use mode-specific attribute names
        fn_attr = f"_decode_fn_{mode}"
        compiled_attr = f"_compiled_decode_fn_{mode}"
        input_ids_attr = f"_decode_input_ids_{mode}"
        pos_ids_attr = f"_decode_position_ids_{mode}"
        cache_pos_attr = f"_decode_cache_position_{mode}"

        if not hasattr(self, fn_attr):
            # Create mode-specific static buffers
            setattr(self, input_ids_attr, torch.empty_like(input_ids))
            setattr(self, pos_ids_attr, torch.empty_like(position_ids))
            setattr(self, cache_pos_attr, torch.empty_like(cache_position))

            # Capture buffer references for closure
            decode_input_ids = getattr(self, input_ids_attr)
            decode_position_ids = getattr(self, pos_ids_attr)
            decode_cache_position = getattr(self, cache_pos_attr)

            def decode_fn():
                hidden = self.vlm.model.language_model(
                    input_ids=decode_input_ids,
                    position_ids=decode_position_ids,
                    past_key_values=self._past_key_values,
                    cache_position=decode_cache_position,
                    use_cache=True,
                ).last_hidden_state[:, -1]
                return self.vlm.lm_head(hidden)

            setattr(self, fn_attr, decode_fn)

        # Copy input tensors to static buffers
        getattr(self, input_ids_attr).copy_(input_ids)
        getattr(self, pos_ids_attr).copy_(position_ids)
        getattr(self, cache_pos_attr).copy_(cache_position)

        if not hasattr(self, compiled_attr):
            getattr(self, fn_attr)()  # Warmup
            setattr(self, compiled_attr, torch.compile(
                getattr(self, fn_attr), mode=self._torch_compile, fullgraph=True
            ))

        torch.compiler.cudagraph_mark_step_begin()
        return getattr(self, compiled_attr)()

    def _action(
        self,
        num_action_tokens: int,
        total_samples: int,
        device: torch.device,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        mode: str = "streaming",
        diffusion_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Run diffusion sampling for action prediction.

        Args:
            mode: "streaming" or "non_streaming" - determines which set of compiled
                  functions and static buffers to use.
        """
        # Use mode-specific attribute names to separate streaming vs non-streaming buffers
        fn_attr = f"_action_fn_{mode}"
        compiled_attr = f"_compiled_action_fn_{mode}"
        pos_ids_attr = f"_action_position_ids_{mode}"
        attn_mask_attr = f"_action_attention_mask_{mode}"
        cache_pos_attr = f"_action_cache_position_{mode}"
        noise_attr = f"_action_noise_{mode}"

        if not hasattr(self, fn_attr):
            # Create mode-specific static buffers
            setattr(self, pos_ids_attr, torch.empty_like(position_ids))
            setattr(self, attn_mask_attr, torch.empty_like(attention_mask))
            setattr(self, cache_pos_attr, torch.empty_like(cache_position))
            setattr(self, noise_attr, torch.empty(
                total_samples, *self.action_space.get_action_space_dims(), device=device, dtype=torch.bfloat16
            ))

            expert_kwargs = {"is_causal": False} if self.config.expert_non_causal_attention else {}
            action_dims = self.action_space.get_action_space_dims()

            # Capture buffer references for closure
            action_position_ids = getattr(self, pos_ids_attr)
            action_attention_mask = getattr(self, attn_mask_attr)
            action_cache_position = getattr(self, cache_pos_attr)
            action_noise = getattr(self, noise_attr)

            def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                action_embeds = self.action_in_proj(x, t)
                if action_embeds.dim() == 2:
                    action_embeds = action_embeds.view(x.shape[0], num_action_tokens, -1)

                hidden = self.expert(
                    inputs_embeds=action_embeds,
                    position_ids=action_position_ids,
                    past_key_values=self._past_key_values,
                    attention_mask=action_attention_mask,
                    cache_position=action_cache_position,
                    use_cache=True,
                    **expert_kwargs,
                ).last_hidden_state[:, -num_action_tokens:]
                return self.action_out_proj(hidden).view(-1, *action_dims)

            def action_fn():
                return self.diffusion.sample(
                    noise=action_noise,
                    batch_size=total_samples,
                    step_fn=step_fn,
                    device=device,
                    return_all_steps=False,
                    **(diffusion_kwargs or {}),
                )

            setattr(self, fn_attr, action_fn)

        # Copy input tensors to static buffers
        getattr(self, pos_ids_attr).copy_(position_ids)
        getattr(self, attn_mask_attr).copy_(attention_mask)
        getattr(self, cache_pos_attr).copy_(cache_position)

        # Generate noise outside compiled graph for deterministic RNG
        getattr(self, noise_attr).normal_()

        if not hasattr(self, compiled_attr):
            getattr(self, fn_attr)()  # Warmup
            setattr(self, compiled_attr, torch.compile(
                getattr(self, fn_attr), mode=self._torch_compile, fullgraph=True
            ))

        torch.compiler.cudagraph_mark_step_begin()
        return getattr(self, compiled_attr)()

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

    # ==================== Main Inference Methods ====================

    def _first_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        device: torch.device,
    ):
        """Run first prefill for streaming mode (caches KV and metadata)."""
        # Cache all streaming related inputs
        (
            vision_start_end_ids_ranges,
            traj_and_text_ids_range,
        ) = self._retrieve_streaming_related_inputs(input_ids)

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

        # Delete cached embeddings for next calls
        if hasattr(self.vlm.model.visual, "_cached_pos_embeds"):
            delattr(self.vlm.model.visual, "_cached_pos_embeds")
        
        for block in self.vlm.model.visual.blocks:
            if hasattr(block.attn, "_num_chunks"):
                delattr(block.attn, "_num_chunks")

        if hasattr(self.vlm.model.language_model, "_cached_deepstack_indices"):
            delattr(self.vlm.model.language_model, "_cached_deepstack_indices")

    @torch.inference_mode()
    def _streaming_rollout(
        self,
        data: dict[str, Any],
        torch_compile: str,
        top_p: float,
        top_k: int | None,
        temperature: float,
        num_traj_samples: int,
        num_traj_sets: int,
        diffusion_kwargs: dict[str, Any] | None,
        fuse_qkv: bool = False,
        fuse_gate_up: bool = False,
        **kwargs: Any,
    ):
        """Streaming mode: reuses KV cache, first call is prefill only."""
        self._torch_compile = torch_compile
        if not hasattr(self, "_patched_for_compile"):
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

        cache_len = self.prefill_seq_length + max_new_tokens + self.num_action_tokens
        self.max_cache_len = cache_len

        # Initialize KV cache on first call
        if self._past_key_values is None:
            self._past_key_values = StaticCache(
                config=self.vlm.config,
                max_cache_len=cache_len,
                max_batch_size=num_samples,
                offloading=False,
            )

        if self.is_first_prefill:
            logger.info("Streaming: First prefill - caching KV, position_ids, and attention_mask.")
            self._first_prefill(input_ids, attention_mask, pixel_values, image_grid_thw, device)
            self._update_past_key_values()
            self.is_first_prefill = False
            if kwargs.get("return_extra", False):
                return None, None, None
            return None, None

        # Prepare input_ids for streaming (slice off system prompt)
        vision_start_token_id = self.tokenizer.encode("<|vision_start|>")[0]
        first_vision_start = torch.where(input_ids == vision_start_token_id)[1][0].item()
        input_ids = input_ids[:, first_vision_start:]

        # Create cache position and position IDs
        cache_position = self._create_cache_position().to(device)
        seq_len = input_ids.shape[1]

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
                mode="streaming",
            )
            cur_pos += 1

        output_ids = replace_padding_after_eos(
            token_ids=output_ids,
            eos_token_id=self.traj_start_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Find <traj_future_start> position
        traj_start_pos = self._find_traj_start_positions(output_ids)

        # ===== Action (Diffusion) =====
        num_samples = num_traj_samples * num_traj_sets
        streaming_input_len = seq_len

        # Note: we must add back the truncated input length in the streaming step.
        action_start_pos = self.prefill_seq_length + (traj_start_pos - streaming_input_len) + 1

        # Build attention mask
        indices = torch.arange(self._past_key_values.max_cache_len, device=device).expand(num_samples, -1)
        is_prompt = indices < action_start_pos[:, None]
        is_action = (indices >= cur_pos) & (indices < cur_pos + self.num_action_tokens)
        attention_mask = torch.where(
            (is_prompt | is_action)[:, None, None, :], 0.0, torch.finfo(torch.float32).min
        )

        cache_position = torch.arange(
            cur_pos, cur_pos + self.num_action_tokens, device=device
        )

        sampled_action = self._action(
            num_action_tokens=self.num_action_tokens,
            total_samples=batch_size * num_samples,
            device=device,
            position_ids=self._cached_position_ids,
            cache_position=cache_position,
            attention_mask=attention_mask,
            mode="streaming",
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

    @torch.inference_mode()
    def _non_streaming_rollout(
        self,
        data: dict[str, Any],
        torch_compile: str,
        top_p: float,
        top_k: int | None,
        temperature: float,
        num_traj_samples: int,
        num_traj_sets: int,
        diffusion_kwargs: dict[str, Any] | None,
        fuse_qkv: bool = False,
        fuse_gate_up: bool = False,
        **kwargs: Any,
    ):
        """Non-streaming mode: resets KV cache each call, processes all frames."""
        self._torch_compile = torch_compile
        if not hasattr(self, "_patched_for_compile"):
            patch_for_torch_compile(self, mode="non_streaming", fuse_qkv=fuse_qkv, fuse_gate_up=fuse_gate_up)
            self._patched_for_compile = True

        # Extract inputs
        tokenized = data["tokenized_data"]
        input_ids = tokenized["input_ids"]
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
        seq_len = input_ids.shape[1]

        # Initialize or reset KV cache
        cache_len = seq_len + max_new_tokens + self.num_action_tokens
        self.max_cache_len = cache_len
        if self._past_key_values is None:
            self._past_key_values = StaticCache(
                config=self.vlm.config,
                max_cache_len=cache_len,
                max_batch_size=num_samples,
                offloading=False,
            )
        self._past_key_values.reset()

        # Compute position_ids
        position_ids, rope_deltas = self.vlm.model.get_rope_index(input_ids, image_grid_thw)

        # Get embeddings
        inputs_embeds = self.vlm.model.get_input_embeddings()(input_ids)

        # ===== Encode =====
        image_embeds, deepstack_image_embeds = self._encode(pixel_values, image_grid_thw)
        image_mask = (input_ids == self.vlm.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # ===== Prefill (no streaming attention mask) =====
        cache_position = torch.arange(seq_len, device=device)
        logits = self._prefill(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            cache_position=cache_position,
            visual_pos_masks=image_mask[..., 0],
            deepstack_image_embeds=deepstack_image_embeds,
            streaming_attention_mask=None,  # Standard causal mask
        )

        # ===== Decode =====
        output_ids = input_ids.clone()
        if num_samples > 1:
            self._past_key_values.expand_batch()
            logits = logits.expand(num_samples, -1).contiguous()
            output_ids = output_ids.expand(num_samples, -1).contiguous()
        unfinished = torch.ones(batch_size * num_samples, dtype=torch.bool, device=device)
        cur_pos = seq_len

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
                position_ids=(cur_pos + rope_deltas).unsqueeze(0).expand(3, -1, -1),
                cache_position=torch.tensor([cur_pos], device=device),
                mode="non_streaming",
            )
            cur_pos += 1

        output_ids = replace_padding_after_eos(
            token_ids=output_ids,
            eos_token_id=self.traj_start_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Find <traj_future_start> position
        traj_start_pos = self._find_traj_start_positions(output_ids)

        # ===== Action (Diffusion) =====
        num_samples = num_traj_samples * num_traj_sets
        action_start_pos = traj_start_pos + 1

        # Build position_ids for action tokens
        position_ids = torch.arange(self.num_action_tokens, device=device)
        position_ids = einops.repeat(position_ids, "t -> 3 b t", b=batch_size * num_samples).clone()
        position_ids += (rope_deltas + action_start_pos[None, :, None]).to(device)

        # Build attention mask
        indices = torch.arange(self._past_key_values.max_cache_len, device=device).expand(num_samples, -1)
        is_prompt = indices < action_start_pos[:, None]
        is_action = (indices >= cur_pos) & (indices < cur_pos + self.num_action_tokens)
        attention_mask = torch.where(
            (is_prompt | is_action)[:, None, None, :], 0.0, torch.finfo(torch.float32).min
        )

        cache_position = torch.arange(
            cur_pos, cur_pos + self.num_action_tokens, device=device
        )

        sampled_action = self._action(
            num_action_tokens=self.num_action_tokens,
            total_samples=batch_size * num_samples,
            device=device,
            position_ids=position_ids,
            cache_position=cache_position,
            attention_mask=attention_mask,
            mode="non_streaming",
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

        # No KV cache update in non-streaming mode

        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, output_ids)
            for key in extra:
                extra[key] = np.array(extra[key]).reshape(
                    [batch_size, num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot

    # ==================== Public API Methods ====================

    def sample_trajectories_from_data_with_streaming_vlm_rollout(
        self,
        data: dict[str, Any],
        torch_compile: str = "max-autotune",
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        diffusion_kwargs: dict[str, Any] | None = None,
        fuse_qkv: bool = False,
        fuse_gate_up: bool = False,
        **kwargs: Any,
    ):
        """
        Streaming inference: reuses KV cache across frames.

        First call caches KV state (returns None), subsequent calls reuse cached state.
        """
        return self._streaming_rollout(
            data=data,
            torch_compile=torch_compile,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_traj_samples=num_traj_samples,
            num_traj_sets=num_traj_sets,
            diffusion_kwargs=diffusion_kwargs,
            fuse_qkv=fuse_qkv,
            fuse_gate_up=fuse_gate_up,
            **kwargs,
        )

    def sample_trajectories_from_data_with_vlm_rollout(
        self,
        data: dict[str, Any],
        torch_compile: str = "max-autotune",
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        diffusion_kwargs: dict[str, Any] | None = None,
        fuse_qkv: bool = False,
        fuse_gate_up: bool = False,
        **kwargs: Any,
    ):
        """
        Non-streaming inference: resets KV cache each call, processes all frames.
        """
        return self._non_streaming_rollout(
            data=data,
            torch_compile=torch_compile,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_traj_samples=num_traj_samples,
            num_traj_sets=num_traj_sets,
            diffusion_kwargs=diffusion_kwargs,
            fuse_qkv=fuse_qkv,
            fuse_gate_up=fuse_gate_up,
            **kwargs,
        )

    def reset_streaming_state(self):
        """Reset streaming state to allow fresh prefill."""
        self.is_first_prefill = True
        self._cached_position_ids = None
        self._cached_attention_mask = None
        self._cached_streaming_attention_mask = None
        self._cached_rope_deltas = None
        self.vision_start_end_ids_ranges = None
        self.traj_and_text_ids_range = None
        # if self._past_key_values is not None:
        #     self._past_key_values.reset()

        for layer in self._past_key_values.layers:
            layer.keys.zero_()
            layer.values.zero_()

        # Reset VLM internal caches
        for block in self.vlm.model.visual.blocks:
            if hasattr(block.attn, "_num_chunks"):
                delattr(block.attn, "_num_chunks")
        if hasattr(self.vlm.model.visual, "_cached_pos_embeds"):
            delattr(self.vlm.model.visual, "_cached_pos_embeds")
        if hasattr(self.vlm.model.language_model, "_cached_deepstack_indices"):
            delattr(self.vlm.model.language_model, "_cached_deepstack_indices")


AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
AutoModel.register(AlpamayoR1Config, AlpamayoR1)
