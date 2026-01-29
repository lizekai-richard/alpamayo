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

        self.past_key_values = None
    
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
        inputs_embeds, deepstack_visual_embeds, visual_pos_masks = self.vlm.embed_tokens(
            input_ids=input_ids, 
            pixel_values=pixel_values, 
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        return inputs_embeds, deepstack_visual_embeds, visual_pos_masks
    
    def _retrieve_streaming_related_inputs(self, input_ids, num_views=4, num_images_per_view=4, merge_ratio=4):
        # For the first generation step, we follow the original prefillin logic and do not update kv cache in the streaming manner.
        if self.past_key_values is None:
            return None, None, None, None, None
        
        # For subsequent streaming generation steps, the new kv cache (one frame per view) will be inserted in every view range. The trajectory history and text will also be recomputed.
        image_token = "<|image_pad|>" if not hasattr(self.processor.tokenizer, "image_token") else self.processor.tokenizer.image_token
        image_token_id = self.processor.tokenizer.encode(image_token)[0]

        mask = (input_ids == image_token_id)
        total_image_tokens = mask.sum().item()

        # hard-code number: 4 * 720 = 2880
        num_image_tokens_per_view = total_image_tokens // num_views \
            if total_image_tokens % num_views == 0 \
                else total_image_tokens // num_views + 1
            
        # hard-code number: 720
        num_image_tokens_per_frame = num_image_tokens_per_view // num_images_per_view \
            if num_image_tokens_per_view % num_images_per_view == 0 \
                else num_image_tokens_per_view // num_images_per_view + 1

        all_image_start = torch.where(mask)[1][0].item()
        all_image_end = torch.where(mask)[1][-1].item()
        # we want first frame ids range for each view
        first_frame_ids_ranges = []
        image_token_ids_per_view_ranges = []
        image_placeholder_ids_ranges = []
        for i in range(num_views):
            first_frame_start = all_image_start + i * num_image_tokens_per_view
            first_frame_end = all_image_start + i * num_image_tokens_per_view + num_image_tokens_per_frame
            first_frame_ids_ranges.append((first_frame_start, first_frame_end))

            image_token_start = first_frame_end
            image_token_end = first_frame_end + num_image_tokens_per_view
            image_token_ids_per_view_ranges.append((image_token_start, image_token_end))

            image_placeholder_end = first_frame_start + num_image_tokens_per_view
            image_placeholder_start = image_token_end - num_image_tokens_per_frame
            image_placeholder_ids_ranges.append((image_placeholder_start, image_placeholder_end))

        traj_and_text_ids_range = (all_image_end, input_ids.shape[1])
        return first_frame_ids_ranges, image_token_ids_per_view_ranges, image_placeholder_ids_ranges, traj_and_text_ids_range, num_image_tokens_per_frame
    
    def update_past_key_values(self, first_frame_ids_ranges, image_token_ids_per_view_ranges, image_placeholder_ids_ranges, traj_and_text_ids_range):
        """Update the past key values.

        Args:
            past_key_values: The past key values.
            first_frame_ids_ranges: The first frame ids ranges.
            traj_and_text_ids_range: The traj and text ids range.
        """
        # Create mask once for all layers
        batch_size, num_kv_heads, seq_len, head_dim = self.past_key_values[0][0].shape

        evicted_image_kv_len = first_frame_ids_ranges[0][1] - first_frame_ids_ranges[0][0]
        evicted_text_kv_len = traj_and_text_ids_range[1] - traj_and_text_ids_range[0]
        image_place_holder_key_cache = torch.zeros(batch_size, num_kv_heads, evicted_image_kv_len, head_dim, device=self.past_key_values[0][0].device, dtype=self.past_key_values[0][0].dtype)
        image_place_holder_value_cache = torch.zeros(batch_size, num_kv_heads, evicted_image_kv_len, head_dim, device=self.past_key_values[0][0].device, dtype=self.past_key_values[0][0].dtype)
        traj_and_text_place_holder_key_cache = torch.zeros(batch_size, num_kv_heads, evicted_text_kv_len, head_dim, device=self.past_key_values[0][0].device, dtype=self.past_key_values[0][0].dtype)
        traj_and_text_place_holder_value_cache = torch.zeros(batch_size, num_kv_heads, evicted_text_kv_len, head_dim, device=self.past_key_values[0][0].device, dtype=self.past_key_values[0][0].dtype)
        
        # Process all layers
        new_past_key_values = []
        for layer_idx in range(len(self.past_key_values)):
            key_cache = self.past_key_values[layer_idx][0]  # shape [1, 20, seq_len, 128]
            value_cache = self.past_key_values[layer_idx][1]

            new_key_cache = torch.empty_like(key_cache)
            new_value_cache = torch.empty_like(value_cache)

            new_key_cache[:, :, :first_frame_ids_ranges[0][0], :].copy_(key_cache[:, :, :first_frame_ids_ranges[0][0], :])
            new_value_cache[:, :, :first_frame_ids_ranges[0][0], :].copy_(value_cache[:, :, :first_frame_ids_ranges[0][0], :])

            for first_frame_ids_range, image_token_ids_per_view_range, image_placeholder_ids_range in zip(first_frame_ids_ranges, image_token_ids_per_view_ranges, image_placeholder_ids_ranges):
                new_kv_start = first_frame_ids_range[0]
                new_kv_end = image_token_ids_per_view_range[0] - evicted_image_kv_len
                old_kv_start = first_frame_ids_range[1]
                old_kv_end = image_token_ids_per_view_range[1]
                place_holder_start = image_placeholder_ids_range[0]
                place_holder_end = image_placeholder_ids_range[1]
                new_key_cache[:, :, new_kv_start:new_kv_end, :].copy_(key_cache[:, :, old_kv_start:old_kv_end, :])
                new_value_cache[:, :, new_kv_start:new_kv_end, :].copy_(value_cache[:, :, old_kv_start:old_kv_end, :])
                image_place_holder_key_cache[:, :, place_holder_start:place_holder_end, :].copy_(image_place_holder_key_cache)
                image_place_holder_value_cache[:, :, place_holder_start:place_holder_end, :].copy_(image_place_holder_value_cache)

            new_key_cache[:, :, traj_and_text_ids_range[0]:traj_and_text_ids_range[1]].copy_(traj_and_text_place_holder_key_cache)
            new_value_cache[:, :, traj_and_text_ids_range[0]:traj_and_text_ids_range[1]].copy_(traj_and_text_place_holder_value_cache)
            
            new_past_key_values.append((new_key_cache, new_value_cache))
        self.past_key_values = tuple(new_past_key_values)
    
    def create_cache_position(self, image_placeholder_ids_ranges, traj_and_text_ids_range):
        """Create the cache position.

        Args:
            image_placeholder_ids_ranges: The image placeholder ids ranges.
            traj_and_text_ids_range: The traj and text ids range.
        """
        if self.past_key_values is None:
            return None
        cache_position = []
        for i in range(len(image_placeholder_ids_ranges)):
            cache_position.append(torch.arange(image_placeholder_ids_ranges[i][0], image_placeholder_ids_ranges[i][1], dtype=torch.long))
        cache_position.append(torch.arange(traj_and_text_ids_range[0], traj_and_text_ids_range[1], dtype=torch.long))
        cache_position = torch.cat(cache_position, dim=0)
        return cache_position
    
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
        print(f"input_ids: {input_ids.shape}")
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

        first_frame_ids_ranges, image_token_ids_per_view_ranges, image_placeholder_ids_ranges, traj_and_text_ids_range, num_image_tokens_per_frame = self._retrieve_streaming_related_inputs(input_ids)
        if first_frame_ids_ranges is not None:
            print(f"first_frame_ids_ranges: {first_frame_ids_ranges}")
            print(f"image_token_ids_per_view_ranges: {image_token_ids_per_view_ranges}")
            print(f"image_placeholder_ids_ranges: {image_placeholder_ids_ranges}")
            print(f"traj_and_text_ids_range: {traj_and_text_ids_range}")
            print(f"num_image_tokens_per_frame: {num_image_tokens_per_frame}")

        inputs_embeds, deepstack_visual_embeds, visual_pos_masks = self.embed_tokens(
            input_ids=input_ids,
            pixel_values=tokenized_data.get("pixel_values", None),
            image_grid_thw=tokenized_data.get("image_grid_thw", None),
            pixel_values_videos=tokenized_data.get("pixel_values_videos", None),
            video_grid_thw=tokenized_data.get("video_grid_thw", None),
        )

        cache_position = self.create_cache_position(image_placeholder_ids_ranges, traj_and_text_ids_range)
        if cache_position is not None:
            cache_position = cache_position.to(device)
            print(f"cache_position: {cache_position}")

        vlm_outputs = self.vlm.generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            past_key_values=self.past_key_values,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            image_placeholder_ids_ranges=image_placeholder_ids_ranges,
            traj_and_text_ids_range=traj_and_text_ids_range,
            num_image_tokens_per_frame=num_image_tokens_per_frame,
            cache_position=cache_position,
        )
        vlm_outputs.rope_deltas = self.vlm.model.rope_deltas

        # manually replace padding after EOS token
        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        self.past_key_values = vlm_outputs.past_key_values

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

        # return the text tokens generated by the VLM
        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)
            # rearrange text tokens to shape [B, ns, nj] to match trajectory shape
            for text_tokens in extra.keys():
                extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                    [input_ids.shape[0], num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, extra
        
        # update the past key values
        print("update the past key values")
        self.update_past_key_values(first_frame_ids_ranges, image_token_ids_per_view_ranges, image_placeholder_ids_ranges, traj_and_text_ids_range)

        return pred_xyz, pred_rot


AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
AutoModel.register(AlpamayoR1Config, AlpamayoR1)
