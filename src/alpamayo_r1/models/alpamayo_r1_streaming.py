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
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from alpamayo_r1.action_space import ActionSpace
from alpamayo_r1.models.streaming_model import ReasoningVLA
from alpamayo_r1.config import AlpamayoR1Config
from alpamayo_r1.diffusion.base import BaseDiffusion
from alpamayo_r1.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)
from transformers import StaticCache
from alpamayo_r1.models.streaming_masking_utils import (
    create_streaming_attention_mask_sdpa_optimized,
)

# Check Flex Attention availability
FLEX_ATTENTION_AVAILABLE = False
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    from alpamayo_r1.models.streaming_masking_utils import (
        create_streaming_attention_mask_flex,
    )
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    pass

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
    """Expert model for reasoning VLA."""

    config_class: type[AlpamayoR1Config] = AlpamayoR1Config
    base_model_prefix = "vlm"

    def __init__(
        self,
        config: AlpamayoR1Config,
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

        self.max_cache_len = 3006 + 256
        self.past_key_values = StaticCache(
            config=self.vlm.config,
            max_cache_len=self.max_cache_len,
            offloading=False
        )

        self.vision_start_end_ids_ranges = None
        self.image_token_ids_ranges = None
        self.traj_and_text_ids_range = None
        self.num_image_tokens_per_frame = 180
        self.num_image_tokens_per_view = 180 * 4
        self.num_views = 4
        self.num_frames_per_view = 4
        self.prefill_seq_length = 3006
        self.is_first_prefill = True

        # Streaming attention mask cache
        self._streaming_mask_cache = {
            "sdpa": None,
            "flex": None,
            "config": None,  # (kv_length, vision_ranges_str, traj_range)
        }
        self._use_flex_attention = False  # Set to True to use Flex Attention
    
    def _set_processor(self, processor):
        self.processor = processor
    
    def embed_tokens(self, 
        input_ids: torch.LongTensor, 
        pixel_values: torch.Tensor, 
        image_grid_thw: torch.LongTensor,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed the input IDs, pixel values, and image grid THW into a tensor of shape (B, T, D).

        Args:
            input_ids: The input IDs.
            pixel_values: The pixel values.
            image_grid_thw: The image grid THW.
            pixel_values_videos: The pixel values of the videos.
            video_grid_thw: The video grid THW.

        Returns:
            inputs_embeds: The inputs embeds.
            deepstack_visual_embeds: The deepstack visual embeds.
            visual_pos_masks: The visual pos masks.
        """
        if not self.is_first_prefill:
            # exclude the system prompt tokens
            first_vision_start_token_id = self.vision_start_end_ids_ranges[0][0]  # this is safe to call because vision_start_end_ids_ranges will be generated after the first prefill
            input_ids = input_ids[first_vision_start_token_id:]

        inputs_embeds, deepstack_visual_embeds, visual_pos_masks = self.vlm.embed_tokens(
            input_ids=input_ids, 
            pixel_values=pixel_values, 
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

        return inputs_embeds, deepstack_visual_embeds, visual_pos_masks
    
    def _retrieve_streaming_related_inputs(self, input_ids):
        """
            This function returns:
            - vision_start_end_ids_ranges: The vision start and end token ids ranges. It's a list of tuples, each tuple contains the start and end index of the vision start and end token ids for a frame. The length of the list is the number of views.
            - image_tokens_ids_ranges: The image token ids ranges. It's a list of tuples, each tuple contains the start and end index of the image token ids for a frame. The length of the list is the number of views.
            - traj_and_text_ids_range: The traj and text token ids range.
            For all ranges, we follow the python style and store a (closed, open) range
        """
        vision_start_end_ids_ranges = []
        image_token_ids_ranges = []
        traj_and_text_ids_range = []
        
        # For subsequent streaming generation steps, the new kv cache (one frame per view) will be inserted in every view range. The trajectory history and text will also be recomputed.
        vision_start_token = "<|vision_start|>" if not hasattr(self.processor.tokenizer, "vision_start_token") else self.processor.tokenizer.vision_start_token
        vision_end_token = "<|vision_end|>" if not hasattr(self.processor.tokenizer, "vision_end_token") else self.processor.tokenizer.vision_end_token

        vision_start_token_id = self.processor.tokenizer.encode(vision_start_token)[0]
        vision_end_token_id = self.processor.tokenizer.encode(vision_end_token)[0]

        vision_start_token_mask = (input_ids == vision_start_token_id)
        vision_end_token_mask = (input_ids == vision_end_token_id)
            
        # Get all positions where image tokens occur
        all_vision_start_token_ids = torch.where(vision_start_token_mask)[1]
        all_vision_end_token_ids = torch.where(vision_end_token_mask)[1]

        # It's better to use <|vision_start|> and <|vision_end|> to locate image tokens since we don't need to consider the vision start and end tokens in the middle
        for vision_start, vision_end in zip(all_vision_start_token_ids, all_vision_end_token_ids):
            vision_start_end_ids_ranges.append((vision_start, vision_end + 1))
            image_tokens_start = vision_start + 1
            image_tokens_end = vision_end
            image_token_ids_ranges.append((image_tokens_start, image_tokens_end))
        
        last_vision_end_id = all_vision_end_token_ids[-1]
        traj_and_text_ids_range = (last_vision_end_id + 1, self.prefill_seq_length)

        return vision_start_end_ids_ranges, image_token_ids_ranges, traj_and_text_ids_range
    
    def update_past_key_values(self, vision_start_end_ids_ranges, image_token_ids_ranges, traj_and_text_ids_range):
        """Update the past key values.
            For each view, shift the the last three frames to the left and keep one dummy frame buffer
            Discard all trajectory and text tokens
        """
        # Create mask once for all layers
        num_layers = len(self.past_key_values)

        for layer_idx in range(num_layers):
            key_cache = self.past_key_values[layer_idx].keys
            value_cache = self.past_key_values[layer_idx].values

            for i in range(self.num_views):
                new_kv_start = vision_start_end_ids_ranges[i][0][0]  # first frame vision start
                new_kv_end = vision_start_end_ids_ranges[i][-2][1]  # third frame vision end

                old_kv_start = image_token_ids_ranges[i][1][0]  # second frame image start
                old_kv_end = image_token_ids_ranges[i][-1][1]  # fourth frame image end

                key_cache[:, :, new_kv_start:new_kv_end, :] = key_cache[:, :, old_kv_start:old_kv_end, :].clone()
                value_cache[:, :, new_kv_start:new_kv_end, :] = value_cache[:, :, old_kv_start:old_kv_end, :].clone()
                
    
    def create_cache_position(self, vision_start_end_ids_ranges, traj_and_text_ids_range):
        """Create the cache position.
            The cache positions are fixed: last frame tokens for each view + trajectory and text tokens
        """
        cache_position = []
        for i in range(self.num_views):
            last_frame_vision_start, last_frame_vision_end = vision_start_end_ids_ranges[i][-1]
            cache_position.append(torch.arange(last_frame_vision_start, last_frame_vision_end))

        cache_position.append(torch.arange(traj_and_text_ids_range[0], traj_and_text_ids_range[1]))
        cache_position = torch.cat(cache_position, dim=0)
        return cache_position

    def _get_streaming_mask_config_key(
        self,
        kv_length: int,
        vision_start_end_ids_ranges: list[list[tuple[int, int]]],
        traj_and_text_ids_range: tuple[int, int],
    ) -> tuple:
        """Generate a unique key for the streaming mask configuration."""
        # Convert vision_start_end_ids_ranges to a hashable format
        vision_ranges_tuple = tuple(
            tuple(frame_range for frame_range in view_ranges)
            for view_ranges in vision_start_end_ids_ranges
        )
        return (kv_length, vision_ranges_tuple, traj_and_text_ids_range)

    def _get_or_create_streaming_mask(
        self,
        cache_position: torch.Tensor,
        vision_start_end_ids_ranges: list[list[tuple[int, int]]],
        traj_and_text_ids_range: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        use_flex: bool = False,
    ):
        """
        Get or create streaming attention mask with caching.

        The mask is cached because in streaming inference, the mask configuration
        (vision_start_end_ids_ranges, traj_and_text_ids_range) stays constant across
        streaming steps - only the KV cache values change, not the attention pattern.

        The mask has shape [1, 1, query_length, max_cache_len] where:
        - [0:prefill_seq_length] contains the streaming attention pattern
        - [prefill_seq_length:max_cache_len] is padding (-inf)

        Args:
            cache_position: The cache positions for the query tokens.
            vision_start_end_ids_ranges: List of views, each containing list of
                (start, end) tuples for each frame's vision tokens.
            traj_and_text_ids_range: (start, end) tuple for trajectory and text tokens.
            device: The device to create tensors on.
            dtype: The dtype for SDPA mask (ignored for Flex Attention).
            use_flex: If True and available, use Flex Attention BlockMask.

        Returns:
            For SDPA: 4D float mask [batch_size, 1, query_length, max_cache_len]
            For Flex: BlockMask object
        """
        config_key = self._get_streaming_mask_config_key(
            self.prefill_seq_length, vision_start_end_ids_ranges, traj_and_text_ids_range
        )

        # Check cache hit
        if self._streaming_mask_cache["config"] == config_key:
            if use_flex and FLEX_ATTENTION_AVAILABLE and self._streaming_mask_cache["flex"] is not None:
                logger.debug("Streaming mask cache hit (Flex)")
                return self._streaming_mask_cache["flex"]
            elif not use_flex and self._streaming_mask_cache["sdpa"] is not None:
                logger.debug("Streaming mask cache hit (SDPA)")
                return self._streaming_mask_cache["sdpa"]

        # Cache miss - create new mask(s)
        logger.info("Creating streaming attention mask (cache miss)")

        # Create SDPA mask for prefill_seq_length (the valid part)
        sdpa_mask = create_streaming_attention_mask_sdpa_optimized(
            batch_size=1,
            cache_position=cache_position,
            kv_length=self.prefill_seq_length,
            vision_start_end_ids_ranges=vision_start_end_ids_ranges,
            traj_and_text_ids_range=traj_and_text_ids_range,
            device=device,
            dtype=dtype,
        )

        # Extend to max_cache_len with padding (-inf for decode positions)
        padding_length = self.max_cache_len - self.prefill_seq_length
        if padding_length > 0:
            query_length = sdpa_mask.shape[2]
            padding_mask = torch.full(
                (1, 1, query_length, padding_length),
                float('-inf'),
                device=device,
                dtype=dtype,
            )
            sdpa_mask = torch.cat([sdpa_mask, padding_mask], dim=-1)

        self._streaming_mask_cache["sdpa"] = sdpa_mask

        # Optionally create Flex Attention BlockMask (not extending for now)
        if use_flex and FLEX_ATTENTION_AVAILABLE:
            # TODO: Extend Flex mask creation to handle padding
            flex_mask = create_streaming_attention_mask_flex(
                batch_size=1,
                cache_position=cache_position,
                kv_length=self.prefill_seq_length,
                vision_start_end_ids_ranges=vision_start_end_ids_ranges,
                traj_and_text_ids_range=traj_and_text_ids_range,
                device=device,
            )
            self._streaming_mask_cache["flex"] = flex_mask
        else:
            self._streaming_mask_cache["flex"] = None

        self._streaming_mask_cache["config"] = config_key

        if use_flex and FLEX_ATTENTION_AVAILABLE:
            return self._streaming_mask_cache["flex"]
        return self._streaming_mask_cache["sdpa"]

    def clear_streaming_mask_cache(self):
        """Clear the streaming attention mask cache."""
        self._streaming_mask_cache = {
            "sdpa": None,
            "flex": None,
            "config": None,
        }
        logger.info("Streaming attention mask cache cleared")

    def crop_static_cache(self, valid_length: int):
        """
        Crop the static cache by zeroing out positions beyond valid_length.

        Unlike DynamicCache.crop() which slices tensors, StaticCache has fixed-size
        tensors. This method zeros out the invalid positions so that get_seq_length()
        returns the correct value.

        Args:
            valid_length: Keep positions [0:valid_length], zero out [valid_length:].
        """
        for layer in self.past_key_values.layers:
            if layer.is_initialized:
                layer.keys[:, :, valid_length:, :] = 0
                layer.values[:, :, valid_length:, :]  = 0
    
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
        """Sample trajectories from the data with streaming VLM rollout.

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
        n_samples_total = num_traj_samples * num_traj_sets
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."
        tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")
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
        generation_config = self.vlm.generation_config
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = num_traj_samples
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = True
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        # use custom stopping criteria to stop after EOS token + one more token,
        # because the KV cache is updated after the next token is generated
        eos_token_id = self.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=self.config.traj_token_start_idx,
                    traj_vocab_size=self.config.traj_vocab_size,
                )
            ]
        )

        inputs_embeds, deepstack_visual_embeds, visual_pos_masks = self.embed_tokens(
            input_ids=input_ids,
            pixel_values=tokenized_data.get("pixel_values", None),
            image_grid_thw=tokenized_data.get("image_grid_thw", None),
            pixel_values_videos=tokenized_data.get("pixel_values_videos", None),
            video_grid_thw=tokenized_data.get("video_grid_thw", None),
        )

        if self.vision_start_end_ids_ranges is not None:
            cache_position = self.create_cache_position(
                vision_start_end_ids_ranges=self.vision_start_end_ids_ranges,
                traj_and_text_ids_range=self.traj_and_text_ids_range,
            )
        else:
            cache_position = torch.arange(0, self.prefill_seq_length)

        cache_position = cache_position.to(device)
        # Get or create streaming attention mask (cached for efficiency)
        if self.vision_start_end_ids_ranges is not None:
            streaming_attention_mask = self._get_or_create_streaming_mask(
                cache_position=cache_position,
                kv_length=self.prefill_seq_length,
                vision_start_end_ids_ranges=self.vision_start_end_ids_ranges,
                traj_and_text_ids_range=self.traj_and_text_ids_range,
                device=device,
                dtype=inputs_embeds.dtype,
                use_flex=self._use_flex_attention,
            )
        else:
            streaming_attention_mask = None

        logger.info(f"input_embeds: {inputs_embeds.shape}")
        logger.info(f"visual_pos_masks: {visual_pos_masks.shape}")
        logger.info(f"cache_position: {cache_position.shape if cache_position is not None else None}")
        logger.info(f"streaming_attention_mask: {streaming_attention_mask.shape if streaming_attention_mask is not None else None}")

        vlm_outputs = self.vlm.generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            past_key_values=self.past_key_values,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            cache_position=cache_position,
            streaming_attnention_mask=streaming_attention_mask,
            **tokenized_data,
        )
        vlm_outputs.rope_deltas = self.vlm.model.rope_deltas

        # manually replace padding after EOS token
        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        prefill_seq_len = self.past_key_values.get_seq_length()  # past_key_values: [1, 8, seq_len, 128]

        # find <traj_future_start> token position for each sequence, use last token if not found
        b_star = vlm_outputs.sequences.shape[0]
        traj_future_start_mask = vlm_outputs.sequences == eos_token_id
        # [b_star], True if sequence has <traj_future_start>
        has_traj_future_start = traj_future_start_mask.any(dim=1)
        for i in range(b_star):
            if not has_traj_future_start[i]:
                logger.warning(
                    f"No <traj_future_start> token found in the generated sequences for sequence {i}"
                )
        # [b_star], first occurrence position
        traj_future_start_positions = traj_future_start_mask.int().argmax(dim=1)
        last_token_positions = torch.full(
            (b_star,), vlm_outputs.sequences.shape[1] - 1, device=device
        )
        valid_token_pos_id = torch.where(
            has_traj_future_start, traj_future_start_positions, last_token_positions
        )
        # note that vlm_outputs.sequences already include the input_ids,
        # so no need to add the input_ids length
        offset = valid_token_pos_id + 1

        # modify the position ids to remove padding tokens
        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
        position_ids = torch.arange(n_diffusion_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
        delta = vlm_outputs.rope_deltas + offset[:, None]
        position_ids += delta.to(position_ids.device)

        # modify the attention_masks to remove padding tokens
        attention_mask = torch.zeros(
            (b_star, 1, n_diffusion_tokens, self.past_key_values.get_seq_length() + n_diffusion_tokens),
            dtype=torch.float32,
            device=device,
        )
        for i in range(b_star):
            attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = torch.finfo(
                attention_mask.dtype
            ).min

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        # 2) Define denoising step that consumes noisy action and timestep
        def step_fn(
            x: torch.Tensor,
            t: torch.Tensor,
        ) -> torch.Tensor:
            # x: (B*, *action_dim)
            # t: broadcastable to x leading dims
            b_star = x.shape[0]
            # Project noisy action to expert token embeddings for the n future tokens
            # Expect shape (b*, n_token_per_traj, hidden_size)
            future_token_embeds = self.action_in_proj(x, t)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)

            # Run expert with cached prefill, only on the future tokens
            expert_out_base = self.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=self.past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            # crop the prompt cache to remove the newly added tokens
            self.past_key_values.crop(prefill_seq_len)
            last_hidden = expert_out_base.last_hidden_state  # (b*, Tf, hidden_size)
            last_hidden = last_hidden[:, -n_diffusion_tokens:]
            pred = self.action_out_proj(last_hidden).view(
                -1, *self.action_space.get_action_space_dims()
            )  # (b*, Tf, C_action) -> noise/vector field
            return pred

        # 3) Diffusion sampling in action space with multiple samples per input
        total_batch = B * n_samples_total
        if diffusion_kwargs is None:
            diffusion_kwargs = {}

        sampled_action = self.diffusion.sample(
            batch_size=total_batch,
            step_fn=step_fn,
            device=device,
            return_all_steps=False,
            **diffusion_kwargs,
        )

        # Repeat history to align with num_traj_samples
        hist_xyz_rep = einops.repeat(
            ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )
        hist_rot_rep = einops.repeat(
            ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )

        pred_xyz, pred_rot = self.action_space.action_to_traj(
            sampled_action, hist_xyz_rep, hist_rot_rep
        )

        # 4) Reshape to (B, num_traj_samples, n_traj, ...)
        pred_xyz = einops.rearrange(
            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )
        pred_rot = einops.rearrange(
            pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )

        # update the past key values
        logger.info("before update the past key values: ", self.past_key_values[0].keys.shape)
        vision_start_end_ids_ranges, image_token_ids_ranges, traj_and_text_ids_range = self._retrieve_streaming_related_inputs(input_ids)
        self.update_past_key_values(
            vision_start_end_ids_ranges=vision_start_end_ids_ranges,
            image_token_ids_ranges=image_token_ids_ranges,
            traj_and_text_ids_range=traj_and_text_ids_range,
        )
        logger.info("after update the past key values: ", self.past_key_values[0].keys.shape)
        self.vision_start_end_ids_ranges = vision_start_end_ids_ranges
        self.image_token_ids_ranges = image_token_ids_ranges
        self.traj_and_text_ids_range = traj_and_text_ids_range

        logger.info("vision_start_end_ids_ranges: ", vision_start_end_ids_ranges)
        logger.info("image_token_ids_ranges: ", image_token_ids_ranges)
        logger.info("traj_and_text_ids_range: ", traj_and_text_ids_range)

        # return the text tokens generated by the VLM
        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)
            # rearrange text tokens to shape [B, ns, nj] to match trajectory shape
            for text_tokens in extra.keys():
                extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                    [input_ids.shape[0], num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot
        
        
AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
AutoModel.register(AlpamayoR1Config, AlpamayoR1)
