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
Compiled model with DFlash speculative decoding and streaming inference.

Uses torch.compile with CUDA graphs for optimized inference.
Requires setup_dflash() to be called before sample_trajectories().

Usage:
    model = UnifiedAlpamayoR1.from_pretrained("./Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
    model.setup_dflash(draft_model_path="./draft_model")
    result = model.sample_trajectories(data)
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
from alpamayo_r1.models.streaming_masking_utils import create_streaming_attention_mask_sdpa
from alpamayo_r1.dflash_integration import (
    build_target_layer_ids,
    sample_tokens,
    _TrajectoryTokenMask,
    GenerationStats,
)

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
logger.setLevel(logging.INFO)


class AlpamayoR1(ReasoningVLA):
    """Compiled model with DFlash speculative decoding and streaming KV cache reuse.

    Uses torch.compile with CUDA graphs for optimal performance.
    Requires setup_dflash() before calling sample_trajectories().
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

    # ==================== DFlash Setup ====================

    def setup_dflash(
        self,
        draft_model: torch.nn.Module,
        target_layer_ids: list[int] | None = None,
        mask_token_id: int | None = None,
        block_size: int = 8,
    ):
        """Configure DFlash speculative decoding.

        After calling this, use ``sample_trajectories(data, dflash=True)``
        to run inference with speculative decoding.

        Args:
            draft_model: The DFlash draft model (DFlashDraftModel).
            target_layer_ids: Layer indices for context feature extraction.
                Auto-computed from model configs if None.
            mask_token_id: Token ID used as mask in DFlash blocks.
                Extracted from draft model config if None.
            block_size: Number of tokens per DFlash block.
        """
        self._draft_model = draft_model
        self._dflash_block_size = getattr(draft_model, "block_size", block_size)

        # Determine target layer IDs
        if target_layer_ids is not None:
            self._dflash_target_layer_ids = target_layer_ids
        elif hasattr(draft_model, "target_layer_ids"):
            self._dflash_target_layer_ids = draft_model.target_layer_ids
        else:
            self._dflash_target_layer_ids = build_target_layer_ids(
                self.vlm.config.text_config.num_hidden_layers,
                draft_model.config.num_hidden_layers,
            )

        # Determine mask token ID
        if mask_token_id is not None:
            self._dflash_mask_token_id = mask_token_id
        else:
            mid = getattr(draft_model, "mask_token_id", None)
            if mid is None:
                mid = getattr(draft_model.config, "mask_token_id", None)
            self._dflash_mask_token_id = mid

        # Logits processor to mask trajectory tokens during CoC generation
        self._dflash_logits_processor = _TrajectoryTokenMask(
            traj_token_offset=self.config.traj_token_start_idx,
            traj_vocab_size=self.config.traj_vocab_size,
        )

        # Module references are set lazily after patch_for_torch_compile()
        # replaces the language_model with the patched Qwen3VLTextModel that
        # has set_capture_layer_ids.  See _ensure_dflash_refs().
        self._dflash_refs_initialized = False

        logger.info(
            f"DFlash configured: block_size={self._dflash_block_size}, "
            f"target_layers={self._dflash_target_layer_ids}, "
            f"mask_token_id={self._dflash_mask_token_id}"
        )

    def _ensure_dflash_refs(self):
        """Initialize DFlash module references after patching.

        Must be called after ``patch_for_torch_compile()`` so that the
        language_model is the patched ``Qwen3VLTextModel`` with
        ``set_capture_layer_ids``.
        """
        if self._dflash_refs_initialized:
            return
        self.vlm.model.language_model.set_capture_layer_ids(self._dflash_target_layer_ids)
        self._dflash_embed_tokens = self.vlm.model.language_model.embed_tokens
        self._dflash_lm_head = self.vlm.lm_head
        self._dflash_language_model = self.vlm.model.language_model
        self._dflash_refs_initialized = True

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
        needs_reinit = not hasattr(self, "_encode_fn")
        if not needs_reinit and (
            self._encode_pixel_values.shape != pixel_values.shape
            or self._encode_image_grid_thw.shape != image_grid_thw.shape
        ):
            logger.warning(
                f"_encode: shape changed (pixels {self._encode_pixel_values.shape} -> {pixel_values.shape}, "
                f"grid {self._encode_image_grid_thw.shape} -> {image_grid_thw.shape}), reinitializing buffers"
            )
            for attr in ("_encode_fn", "_compiled_encode_fn"):
                if hasattr(self, attr):
                    delattr(self, attr)
            needs_reinit = True

        if needs_reinit:
            self._encode_pixel_values = torch.empty_like(pixel_values)
            self._encode_image_grid_thw = torch.empty_like(image_grid_thw)

            def encode_fn():
                pixels = self._encode_pixel_values.type(self.vlm.model.visual.dtype)
                return self.vlm.model.visual(pixels, grid_thw=self._encode_image_grid_thw)

            self._encode_fn = encode_fn

        self._encode_pixel_values.copy_(pixel_values)
        self._encode_image_grid_thw.copy_(image_grid_thw)

        if self._torch_compile is None:
            return self._encode_fn()

        if not hasattr(self, "_compiled_encode_fn"):
            self._encode_fn()  # Warmup
            self._compiled_encode_fn = torch.compile(
                self._encode_fn, mode=self._torch_compile, fullgraph=True
            )

        torch.compiler.cudagraph_mark_step_begin()
        return self._compiled_encode_fn()

    def _dflash_prefill(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        deepstack_image_embeds: list[torch.Tensor],
        streaming_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run prefill forward pass and return both logits and captured hidden states.

        Used by DFlash rollouts to capture hidden states during prefill,
        avoiding the extra forward pass in _dflash_extract_initial_hidden.
        Requires set_capture_layer_ids() to have been called beforehand.

        Returns:
            (logits, context) where logits is (vocab_size,) and context is
            (1, 1, hidden_dim * num_capture_layers).
        """
        needs_reinit = not hasattr(self, "_dflash_prefill_fn")
        if not needs_reinit and (
            self._dp_inputs_embeds.shape != inputs_embeds.shape
            or self._dp_cache_position.shape != cache_position.shape
            or any(
                b.shape != e.shape
                for b, e in zip(self._dp_deepstack_embeds, deepstack_image_embeds)
            )
        ):
            logger.warning(
                f"_dflash_prefill: shape changed (embeds {self._dp_inputs_embeds.shape} -> {inputs_embeds.shape}, "
                f"cache_pos {self._dp_cache_position.shape} -> {cache_position.shape}), reinitializing buffers"
            )
            for attr in ("_dflash_prefill_fn", "_compiled_dflash_prefill_fn"):
                if hasattr(self, attr):
                    delattr(self, attr)
            needs_reinit = True

        if needs_reinit:
            self._dp_inputs_embeds = torch.empty_like(inputs_embeds)
            self._dp_position_ids = torch.empty_like(position_ids)
            self._dp_cache_position = torch.empty_like(cache_position)
            self._dp_visual_pos_masks = torch.empty_like(visual_pos_masks)
            self._dp_deepstack_embeds = [torch.empty_like(e) for e in deepstack_image_embeds]

            if streaming_attention_mask is not None:
                self._dp_streaming_attention_mask = torch.empty_like(streaming_attention_mask)
            else:
                self._dp_streaming_attention_mask = None

            def dflash_prefill_fn():
                output = self.vlm.model.language_model(
                    inputs_embeds=self._dp_inputs_embeds,
                    position_ids=self._dp_position_ids,
                    past_key_values=self._past_key_values,
                    cache_position=self._dp_cache_position,
                    visual_pos_masks=self._dp_visual_pos_masks,
                    deepstack_visual_embeds=self._dp_deepstack_embeds,
                    streaming_attention_mask=self._dp_streaming_attention_mask,
                    use_cache=True,
                )
                logits = self.vlm.lm_head(output.last_hidden_state[:, -1])
                # hidden_states is tuple of captured layers (via set_capture_layer_ids)
                context = torch.cat(output.hidden_states, dim=-1)[:, -1:, :]
                return logits, context

            self._dflash_prefill_fn = dflash_prefill_fn

        self._dp_inputs_embeds.copy_(inputs_embeds)
        self._dp_position_ids.copy_(position_ids)
        self._dp_cache_position.copy_(cache_position)
        self._dp_visual_pos_masks.copy_(visual_pos_masks)
        for buf, emb in zip(self._dp_deepstack_embeds, deepstack_image_embeds):
            buf.copy_(emb)

        if streaming_attention_mask is not None and self._dp_streaming_attention_mask is not None:
            self._dp_streaming_attention_mask.copy_(streaming_attention_mask)

        if self._torch_compile is None:
            return self._dflash_prefill_fn()

        if not hasattr(self, "_compiled_dflash_prefill_fn"):
            self._dflash_prefill_fn()  # Warmup
            self._compiled_dflash_prefill_fn = torch.compile(
                self._dflash_prefill_fn, mode=self._torch_compile, fullgraph=True
            )

        torch.compiler.cudagraph_mark_step_begin()
        return self._compiled_dflash_prefill_fn()

    def _action(
        self,
        num_action_tokens: int,
        total_samples: int,
        device: torch.device,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        diffusion_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Run diffusion sampling for action prediction."""
        if not hasattr(self, "_action_fn"):
            self._action_position_ids = torch.empty_like(position_ids)
            self._action_attention_mask = torch.empty_like(attention_mask)
            self._action_cache_position = torch.empty_like(cache_position)
            self._action_noise = torch.empty(
                total_samples, *self.action_space.get_action_space_dims(), device=device, dtype=torch.bfloat16
            )

            expert_kwargs = {"is_causal": False} if self.config.expert_non_causal_attention else {}
            action_dims = self.action_space.get_action_space_dims()

            # Capture buffer references for closure
            action_position_ids = self._action_position_ids
            action_attention_mask = self._action_attention_mask
            action_cache_position = self._action_cache_position
            action_noise = self._action_noise

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

            self._action_fn = action_fn

        # Copy input tensors to static buffers
        self._action_position_ids.copy_(position_ids)
        self._action_attention_mask.copy_(attention_mask)
        self._action_cache_position.copy_(cache_position)

        # Generate noise outside compiled graph for deterministic RNG
        self._action_noise.normal_()

        if self._torch_compile is None:
            return self._action_fn()

        if not hasattr(self, "_compiled_action_fn"):
            self._action_fn()  # Warmup
            self._compiled_action_fn = torch.compile(
                self._action_fn, mode=self._torch_compile, fullgraph=True
            )

        torch.compiler.cudagraph_mark_step_begin()
        return self._compiled_action_fn()

    # ==================== DFlash Compiled Functions ====================

    def _dflash_verify(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compiled verify step: language_model on a block -> (logits, context).

        Processes a block of tokens through the target language model and returns
        both logits (for acceptance checking) and captured hidden states (for the
        next draft iteration).

        Args:
            input_ids: Token IDs to verify, shape (1, block_size).
            position_ids: MROPE position IDs, shape (3, 1, max_cache_len).
            cache_position: Absolute cache positions, shape (block_size,).

        Returns:
            (logits, context) where logits is (1, block_size, vocab_size) and
            context is (1, block_size, hidden_dim * num_capture_layers).
        """
        if not hasattr(self, "_dflash_verify_fn"):
            self._dflash_verify_input_ids = torch.empty_like(input_ids)
            self._dflash_verify_position_ids = torch.empty_like(position_ids)
            self._dflash_verify_cache_position = torch.empty_like(cache_position)

            verify_input_ids = self._dflash_verify_input_ids
            verify_position_ids = self._dflash_verify_position_ids
            verify_cache_position = self._dflash_verify_cache_position

            def verify_fn():
                output = self._dflash_language_model(
                    input_ids=verify_input_ids,
                    position_ids=verify_position_ids,
                    past_key_values=self._past_key_values,
                    cache_position=verify_cache_position,
                    use_cache=True,
                )
                logits = self._dflash_lm_head(output.last_hidden_state)
                context = torch.cat(output.hidden_states, dim=-1)
                return logits, context

            self._dflash_verify_fn = verify_fn

        self._dflash_verify_input_ids.copy_(input_ids)
        self._dflash_verify_position_ids.copy_(position_ids)
        self._dflash_verify_cache_position.copy_(cache_position)

        if self._torch_compile is None:
            return self._dflash_verify_fn()

        if not hasattr(self, "_compiled_dflash_verify_fn"):
            self._dflash_verify_fn()  # Warmup
            self._compiled_dflash_verify_fn = torch.compile(
                self._dflash_verify_fn, mode=self._torch_compile, fullgraph=True
            )

        torch.compiler.cudagraph_mark_step_begin()
        return self._compiled_dflash_verify_fn()

    def _dflash_draft(
        self,
        block_output_ids: torch.Tensor,
        target_context: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compiled draft step: embed -> draft_model -> lm_head.

        Runs the DFlash draft model to predict the next block of tokens.
        The draft model is stateless (no KV cache).

        Args:
            block_output_ids: Current block token IDs, shape (1, block_size).
            target_context: Context from target model, shape (1, 1, context_dim).
            position_ids: Relative positions, shape (1, 1+block_size).

        Returns:
            (draft_hidden, draft_logits) where draft_hidden is (1, 1+block_size, hidden)
            and draft_logits is (1, block_size, vocab_size) (positions 1: onward).
        """
        if not hasattr(self, "_dflash_draft_fn"):
            # Static buffers
            self._dflash_draft_block_ids = torch.empty_like(block_output_ids)
            self._dflash_draft_context = torch.empty_like(target_context)
            self._dflash_draft_pos_ids = torch.empty_like(position_ids)

            # Capture buffer references for closure
            draft_block_ids = self._dflash_draft_block_ids
            draft_context = self._dflash_draft_context
            draft_pos_ids = self._dflash_draft_pos_ids

            def draft_fn():
                noise = self._dflash_embed_tokens(draft_block_ids)
                hidden = self._draft_model(
                    target_hidden=draft_context,
                    noise_embedding=noise,
                    position_ids=draft_pos_ids,
                    past_key_values=None,
                    use_cache=False,
                    is_causal=False,
                )
                logits = self._dflash_lm_head(hidden[:, 1:, :])
                return hidden, logits

            self._dflash_draft_fn = draft_fn

        # Copy to static buffers
        self._dflash_draft_block_ids.copy_(block_output_ids)
        self._dflash_draft_context.copy_(target_context)
        self._dflash_draft_pos_ids.copy_(position_ids)

        # Warmup + compile (first call only)
        if self._torch_compile is None:
            return self._dflash_draft_fn()

        if not hasattr(self, "_compiled_dflash_draft_fn"):
            self._dflash_draft_fn()  # Warmup
            self._compiled_dflash_draft_fn = torch.compile(
                self._dflash_draft_fn, mode=self._torch_compile, fullgraph=True
            )

        torch.compiler.cudagraph_mark_step_begin()
        return self._compiled_dflash_draft_fn()

    def _dflash_traj_forward(
        self,
        traj_token_id: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> None:
        """Compiled single-token forward for <traj_future_start>.

        Populates the StaticCache KV entry at the traj position so the action
        decoder doesn't attend to zeros.

        Args:
            traj_token_id: (1, 1) — the <traj_future_start> token ID.
            position_ids: (3, 1, max_cache_len) — MROPE positions.
            cache_position: (1,) — absolute cache position.
        """
        if not hasattr(self, "_traj_fwd_fn"):
            self._traj_fwd_ids = torch.empty_like(traj_token_id)
            self._traj_fwd_pos = torch.empty_like(position_ids)
            self._traj_fwd_cache_pos = torch.empty_like(cache_position)

            fwd_ids = self._traj_fwd_ids
            fwd_pos = self._traj_fwd_pos
            fwd_cache_pos = self._traj_fwd_cache_pos

            def traj_fwd_fn():
                self._dflash_language_model(
                    input_ids=fwd_ids,
                    position_ids=fwd_pos,
                    past_key_values=self._past_key_values,
                    cache_position=fwd_cache_pos,
                    use_cache=True,
                )

            self._traj_fwd_fn = traj_fwd_fn

        self._traj_fwd_ids.copy_(traj_token_id)
        self._traj_fwd_pos.copy_(position_ids)
        self._traj_fwd_cache_pos.copy_(cache_position)

        if self._torch_compile is None:
            self._traj_fwd_fn()
            return

        if not hasattr(self, "_compiled_traj_fwd"):
            self._traj_fwd_fn()  # Warmup
            self._compiled_traj_fwd = torch.compile(
                self._traj_fwd_fn, mode=self._torch_compile, fullgraph=True
            )

        torch.compiler.cudagraph_mark_step_begin()
        self._compiled_traj_fwd()

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

    # ==================== DFlash Decode Loop ====================

    def _dflash_decode_loop(
        self,
        output_ids: torch.Tensor,
        num_input_tokens: int,
        cur_seq_len: int,
        target_hidden: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        stop_token_ids: list[int] | None = None,
        full_position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, GenerationStats, int]:
        """Run DFlash speculative decode loop.

        Performs block-parallel speculative decoding: draft model proposes a
        block of tokens, target model verifies them, accepted tokens advance
        the sequence.

        The first generated token must already be placed at
        ``output_ids[:, num_input_tokens]`` before calling this method.  The
        decode loop uses that token as the first "known" token of the first
        block.

        Args:
            output_ids: (1, max_length + block_size) buffer with input tokens
                pre-filled and first generated token at position num_input_tokens.
            num_input_tokens: Number of prompt tokens.  The first generated
                token is at position num_input_tokens in output_ids.
            cur_seq_len: Current valid length in the StaticCache (positions
                0..cur_seq_len-1 are valid).
            target_hidden: (1, 1, D) initial hidden state from prefill.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
            stop_token_ids: Token IDs that signal end of generation.
                If None, loop runs until max_new_tokens.
            full_position_ids: Full padded position IDs covering the entire
                cache range.  The patched attention indexes rotary embeddings
                by ``cache_position`` into this tensor.

        Returns:
            (output_ids, cur_pos, stats, current_seq_len) where output_ids is
            trimmed, cur_pos is the position after the last generated token,
            and current_seq_len is the valid length in the KV cache.
        """
        device = output_ids.device
        bsz = output_ids.shape[0]
        block_size = self._dflash_block_size
        max_length = num_input_tokens + max_new_tokens
        stats = GenerationStats(block_size=block_size)

        # Precompute constant tensors
        reset_position_ids = torch.arange(
            0, 1 + block_size, device=device
        ).unsqueeze(0).expand(bsz, -1)
        start = num_input_tokens  # First known token is at output_ids[:, num_input_tokens]
        current_seq_len = cur_seq_len

        while start < max_length:
            # 1. Prepare block (first token known, rest are masks)
            block_output_ids = output_ids[:, start : start + block_size].clone()

            # 2. Draft: embed → draft_model → lm_head
            current_context = target_hidden[:, -1:, :]
            _draft_hidden, draft_logits = self._dflash_draft(
                block_output_ids, current_context, reset_position_ids,
            )

            # 4. Sample draft tokens
            block_output_ids[:, 1:] = sample_tokens(
                draft_logits, temperature, self._dflash_logits_processor
            )

            # 5. Verify: run target model on drafted block
            verify_cache_position = torch.arange(
                current_seq_len, current_seq_len + block_size, device=device
            )
            verify_logits, verify_context = self._dflash_verify(
                block_output_ids,
                full_position_ids,
                verify_cache_position,
            )

            # 6. Sample from target logits
            posterior = sample_tokens(
                verify_logits, temperature, self._dflash_logits_processor
            )

            # 7. Compute acceptance
            matches = block_output_ids[:, 1:] == posterior[:, :-1]
            acceptance_length = matches.cumprod(dim=1).sum(dim=1)[0].item()

            # Accept matched tokens + target's next token
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

            # 8. Check for stop token
            hit_stop = False
            stop_position = None
            tokens_to_advance = acceptance_length + 1
            period_token_id = 13  # "." token triggers stop heuristic

            if stop_token_ids is not None:
                accepted_tokens = output_ids[0, start : start + acceptance_length + 2]
                stop_token = stop_token_ids[0]

                is_stop = (accepted_tokens == stop_token)
                if is_stop.any():
                    stop_position = is_stop.nonzero(as_tuple=True)[0][0].item()
                    hit_stop = True
                    tokens_to_advance = stop_position
                    if stop_position < acceptance_length + 1:
                        output_ids[:, start + stop_position + 1 :] = self._dflash_mask_token_id
                elif posterior[0, acceptance_length].item() == period_token_id:
                    # Period heuristic: treat "." as stop
                    hit_stop = True
                    stop_position = acceptance_length + 1

            # 9. Update position and stats
            start += tokens_to_advance

            if hit_stop and stop_position is not None:
                stats.acceptance_lengths.append(stop_position)
                draft_matched = min(stop_position - 1, block_size - 1)
                stats.draft_matches.append(draft_matched)
                stats.tokens_verified.append(draft_matched)
                stats.hit_stop.append(True)
            else:
                stats.acceptance_lengths.append(acceptance_length + 1)
                stats.draft_matches.append(acceptance_length)
                stats.tokens_verified.append(block_size - 1)
                stats.hit_stop.append(False)
            stats.drafting_iterations += 1
            stats.total_iterations += 1

            if hit_stop:
                # Update cache state before breaking: verify wrote the full
                # block to cache, crop to only the valid entries.
                current_seq_len += min(stop_position + 1, block_size)
                self._crop_static_cache(current_seq_len)
                break

            # 10. Update cache and extract hidden state for next iteration
            current_seq_len += acceptance_length + 1
            if acceptance_length < block_size - 1:
                self._crop_static_cache(current_seq_len)

            target_hidden = verify_context[:, acceptance_length : acceptance_length + 1, :]

        # Cleanup output: remove mask tokens, trim to max_length
        output_ids = output_ids[:, :max_length]
        mask = output_ids[0] != self._dflash_mask_token_id
        output_ids = output_ids[:, mask]

        # Truncate at stop token if found
        if stop_token_ids is not None:
            stop_token_ids_tensor = torch.tensor(stop_token_ids, device=device)
            generated_tokens = output_ids[0, num_input_tokens:]
            stop_indices = torch.isin(generated_tokens, stop_token_ids_tensor).nonzero(as_tuple=True)[0]
            if stop_indices.numel() > 0:
                output_ids = output_ids[:, : num_input_tokens + stop_indices[0] + 1]

        stats.total_tokens = output_ids.shape[1] - num_input_tokens
        return output_ids, start, stats, current_seq_len

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
        image_embeds, deepstack_image_embeds, _, _ = self.vlm.model.visual(pixel_values, grid_thw=image_grid_thw)
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

        # Delete cached embeddings so streaming steps reinitialize with correct sizes
        for attr in ("_cached_pos_embeds", "_cached_position_embeddings", "_cached_cu_seqlens"):
            if hasattr(self.vlm.model.visual, attr):
                delattr(self.vlm.model.visual, attr)

        # Clear _num_chunks from visual attention blocks (from dev/zekail)
        for block in self.vlm.model.visual.blocks:
            if hasattr(block.attn, "_num_chunks"):
                delattr(block.attn, "_num_chunks")

        if hasattr(self.vlm.model.language_model, "_cached_deepstack_indices"):
            delattr(self.vlm.model.language_model, "_cached_deepstack_indices")

    # ==================== DFlash Rollout Methods ====================

    def sample_trajectories_with_dflash_streaming_vlm_rollout(
        self,
        data: dict[str, Any],
        torch_compile: str,
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
        """DFlash streaming mode: speculative decode with KV cache reuse.

        Mirrors _streaming_rollout() but replaces the token-by-token decode
        loop with DFlash block-parallel speculative decoding.
        """
        self._torch_compile = torch_compile
        if not hasattr(self, "_patched_for_compile"):
            patch_for_torch_compile(self, mode="streaming", fuse_qkv=fuse_qkv, fuse_gate_up=fuse_gate_up)
            self._patched_for_compile = True
        self._ensure_dflash_refs()

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
        block_size = self._dflash_block_size

        # Set prefill_seq_length from actual input on first call
        if self.is_first_prefill:
            self.prefill_seq_length = input_ids.shape[1]
            logger.info(f"DFlash Streaming: prefill_seq_length set to {self.prefill_seq_length}")

        cache_len = self.prefill_seq_length + max_new_tokens + block_size + self.num_action_tokens
        self.max_cache_len = cache_len

        # Initialize KV cache on first call
        if self._past_key_values is None:
            self._past_key_values = StaticCache(
                config=self.vlm.config.text_config,
                max_cache_len=cache_len,
                max_batch_size=num_samples,
                offloading=False,
            )

        if self.is_first_prefill:
            logger.info("DFlash Streaming: First prefill - caching KV, position_ids, and attention_mask.")
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
        image_embeds, deepstack_image_embeds, _, _ = self._encode(pixel_values, image_grid_thw)
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
        logits, target_hidden = self._dflash_prefill(
            inputs_embeds=inputs_embeds,
            position_ids=self._cached_position_ids,
            cache_position=cache_position,
            visual_pos_masks=image_mask[..., 0],
            deepstack_image_embeds=deepstack_image_embeds,
            streaming_attention_mask=self._cached_streaming_attention_mask,
        )

        # ===== DFlash Decode =====
        num_input_tokens = self.prefill_seq_length

        # Initialize output buffer with mask tokens
        max_length = num_input_tokens + max_new_tokens
        dflash_output_ids = torch.full(
            (batch_size, max_length + block_size),
            self._dflash_mask_token_id,
            dtype=torch.long,
            device=device,
        )
        # Copy a placeholder for input region (we only need generated tokens)
        dflash_output_ids[:, :num_input_tokens] = self.tokenizer.pad_token_id

        # Sample first token from prefill logits
        first_token_logits = logits.unsqueeze(1)  # (1, 1, vocab)
        first_token = sample_tokens(first_token_logits, 0.0, self._dflash_logits_processor)
        dflash_output_ids[:, num_input_tokens : num_input_tokens + 1] = first_token

        # Run speculative decode loop (stop at <cot_end>, matching external accelerator)
        cot_end_token_id = self.tokenizer.convert_tokens_to_ids(to_special_token("cot_end"))
        dflash_output_ids, dflash_end_pos, dflash_stats, cur_seq_len = self._dflash_decode_loop(
            output_ids=dflash_output_ids,
            num_input_tokens=num_input_tokens,
            cur_seq_len=num_input_tokens,
            target_hidden=target_hidden,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            stop_token_ids=[cot_end_token_id],
            full_position_ids=self._cached_position_ids,
        )

        # Forward <traj_future_start> through LLM to populate KV cache entry.
        # Without this, the StaticCache at traj position has zeros — the action
        # decoder attends to these zeros, degrading trajectory quality.
        # Uses compiled path (_dflash_traj_forward) when torch.compile is active.
        # NOTE: In streaming mode, the patched attention indexes rotary embeddings
        # via cache_position into position_ids, so we must pass the full-range
        # position_ids (not a local (3,1,1) tensor) to avoid out-of-bounds indexing.
        traj_token = torch.tensor([[self.traj_start_token_id]], device=device)
        traj_cache_position = torch.tensor([cur_seq_len], device=device, dtype=torch.long)
        self._dflash_traj_forward(
            traj_token, self._cached_position_ids, traj_cache_position,
        )

        # Reconstruct output_ids for traj_start_pos finding
        # We need input_ids (streaming portion) + generated tokens + <traj_future_start>
        generated_tokens = dflash_output_ids[:, num_input_tokens:]
        output_ids = torch.cat([input_ids, generated_tokens, traj_token], dim=-1)

        output_ids = replace_padding_after_eos(
            token_ids=output_ids,
            eos_token_id=self.traj_start_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Find <traj_future_start> position
        traj_start_pos = self._find_traj_start_positions(output_ids)

        # ===== Action (Diffusion) =====
        num_samples = num_traj_samples * num_traj_sets

        # Replicate batch-0 KV to all sample slots before multi-sample action
        if num_samples > 1:
            self._past_key_values.expand_batch()

        streaming_input_len = seq_len

        # Note: we must add back the truncated input length in the streaming step.
        action_start_pos = self.prefill_seq_length + (traj_start_pos - streaming_input_len) + 1

        # Compute cur_pos for action tokens (position in the StaticCache)
        # dflash_end_pos is in dflash_output_ids coordinates (relative to num_input_tokens=prefill_seq_length)
        # We need the actual cache position
        cur_pos = action_start_pos[0].item()

        # Build position_ids for action tokens (batch = num_samples when multi-sample)
        position_ids = torch.arange(self.num_action_tokens, device=device)
        position_ids = einops.repeat(position_ids, "t -> 3 b t", b=batch_size * num_samples).clone()
        position_ids += (self._cached_rope_deltas + action_start_pos[None, :, None]).to(device)

        # Build attention mask (expand indices for num_samples)
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
                # DFlash decodes text once; broadcast across traj samples
                arr = np.array(extra[key])
                extra[key] = np.broadcast_to(
                    arr.reshape(batch_size, 1, 1),
                    [batch_size, num_traj_sets, num_traj_samples],
                ).copy()
            extra["dflash_stats"] = {
                "total_tokens": dflash_stats.total_tokens,
                "total_iterations": dflash_stats.total_iterations,
                "acceptance_rate": dflash_stats.acceptance_rate,
                "mean_acceptance_length": dflash_stats.mean_acceptance_length,
                "match_rate": dflash_stats.match_rate,
                "acceptance_lengths": list(dflash_stats.acceptance_lengths),
            }
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot

    def reset_streaming_state(self):
        """Reset streaming state to allow fresh prefill."""
        self.is_first_prefill = True
        self._cached_position_ids = None
        self._cached_attention_mask = None
        self._cached_streaming_attention_mask = None
        self._cached_rope_deltas = None
        self.vision_start_end_ids_ranges = None
        self.traj_and_text_ids_range = None
        if self._past_key_values is not None:
            self._past_key_values.reset()

        # Reset VLM internal caches (from dev/zekail)
        for block in self.vlm.model.visual.blocks:
            if hasattr(block.attn, "_num_chunks"):
                delattr(block.attn, "_num_chunks")
        for attr in ("_cached_pos_embeds", "_cached_position_embeddings", "_cached_cu_seqlens"):
            if hasattr(self.vlm.model.visual, attr):
                delattr(self.vlm.model.visual, attr)
        if hasattr(self.vlm.model.language_model, "_cached_deepstack_indices"):
            delattr(self.vlm.model.language_model, "_cached_deepstack_indices")


AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
AutoModel.register(AlpamayoR1Config, AlpamayoR1)
