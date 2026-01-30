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


# ==============================================================================
# Flex Attention Masks for Multi-View VLM
# ==============================================================================


def create_causal_mask_mod():
    """Create a standard causal mask_mod for first prefill."""
    def mask_mod(_b, _h, q_idx, kv_idx):
        return kv_idx <= q_idx
    return mask_mod


def create_causal_block_mask(
    q_len: int,
    device: torch.device = None,
    _compile: bool = False,
):
    """
    Create a causal block mask for first prefill.

    Args:
        q_len: Total length of query sequence
        device: Device to create the mask on
        _compile: Whether to compile the mask function

    Returns:
        BlockMask for use with flex_attention

    Example:
        # First prefill with full sequence
        mask = create_causal_block_mask(q_len=seq_len, device=device)
    """
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError:
        raise ImportError(
            "flex_attention requires PyTorch 2.5+. "
            "Please upgrade PyTorch to use flex attention."
        )

    mask_mod = create_causal_mask_mod()

    block_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=q_len,
        KV_LEN=q_len,
        device=device,
        _compile=_compile,
    )

    return block_mask


# ==============================================================================
# Streaming Attention Mask
# ==============================================================================
#
# Layout:
#   KV Cache: [sys] [v0:f1-3] [v1:f1-3] [v2:f1-3] [v3:f1-3]
#   Query:    [v0:f4] [v1:f4] [v2:f4] [v3:f4] [suffix]
#
# Attention rules:
#   - v_i (new frame) → sys + v_0..v_{i-1} (full) + v_i (causal within)
#   - suffix → sys + all images + suffix (causal within)
#
# ==============================================================================


def create_streaming_mask_mod(
    sys_len: int,
    kv_len: int,
    num_views: int = 4,
    tokens_per_frame: int = 180,
    frames_in_kv: int = 3,
):
    """
    Create a mask_mod function for streaming multi-view attention.

    Args:
        sys_len: Length of system tokens in KV cache
        kv_len: Total length of KV cache
        num_views: Number of camera views (default: 4)
        tokens_per_frame: Tokens per frame after spatial merge (default: 180)
        frames_in_kv: Frames per view in KV cache (default: 3)

    Returns:
        A mask_mod function compatible with flex_attention
    """
    # KV cache layout: [sys] [v0: frames_in_kv frames] [v1: ...] ...
    tokens_per_view_kv = frames_in_kv * tokens_per_frame

    # Query layout: [v0:new_frame] [v1:new_frame] ... [suffix]
    q_images_len = num_views * tokens_per_frame
    q_suffix_start = q_images_len

    def mask_mod(_b, _h, q_idx, kv_idx):
        # ===== Determine query token type =====
        q_is_image = q_idx < q_suffix_start
        if q_is_image:
            q_view = q_idx // tokens_per_frame
            q_local = q_idx % tokens_per_frame
        else:
            q_view = -1
            q_local = q_idx - q_suffix_start

        # ===== KV in cache =====
        if kv_idx < kv_len:
            # Sys tokens: everyone can attend
            if kv_idx < sys_len:
                return True

            # Image tokens in KV cache
            kv_offset = kv_idx - sys_len
            kv_view = kv_offset // tokens_per_view_kv

            if q_is_image:
                # Image query: can attend to views <= own view
                return kv_view <= q_view
            else:
                # Suffix query: can attend to all images
                return True

        # ===== KV in query (Q-Q attention) =====
        kv_q_idx = kv_idx - kv_len
        kv_is_image = kv_q_idx < q_suffix_start

        if q_is_image:
            # Image query attending to query tokens
            if kv_is_image:
                kv_view = kv_q_idx // tokens_per_frame
                kv_local = kv_q_idx % tokens_per_frame
                if kv_view < q_view:
                    return True  # Previous views: full attention
                elif kv_view == q_view:
                    return kv_local <= q_local  # Same view: causal
                else:
                    return False  # Future views: no attention
            else:
                return False  # Image cannot attend to suffix
        else:
            # Suffix query
            if kv_is_image:
                return True  # Suffix can see all images
            else:
                # Causal within suffix
                kv_local = kv_q_idx - q_suffix_start
                return kv_local <= q_local

    return mask_mod


def create_streaming_block_mask(
    sys_len: int,
    kv_len: int,
    q_len: int,
    num_views: int = 4,
    tokens_per_frame: int = 180,
    frames_in_kv: int = 3,
    device: torch.device = None,
    _compile: bool = False,
):
    """
    Create a block mask for streaming multi-view attention.

    Args:
        sys_len: Length of system tokens in KV cache
        kv_len: Total length of KV cache
        q_len: Total length of query (new frames + suffix)
        num_views: Number of camera views (default: 4)
        tokens_per_frame: Tokens per frame after spatial merge (default: 180)
        frames_in_kv: Frames per view in KV cache (default: 3)
        device: Device to create the mask on
        _compile: Whether to compile the mask function (for performance)

    Returns:
        BlockMask for use with flex_attention

    Example:
        # KV cache has sys (128) + 4 views × 3 frames × 180 tokens
        # Query has 4 new frames + suffix (100)
        mask = create_streaming_block_mask(
            sys_len=128,
            kv_len=128 + 4*3*180,  # 2288
            q_len=4*180 + 100,      # 820
            device=device,
        )
    """
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError:
        raise ImportError(
            "flex_attention requires PyTorch 2.5+. "
            "Please upgrade PyTorch to use streaming attention."
        )

    mask_mod = create_streaming_mask_mod(
        sys_len=sys_len,
        kv_len=kv_len,
        num_views=num_views,
        tokens_per_frame=tokens_per_frame,
        frames_in_kv=frames_in_kv,
    )

    # Total KV length = KV cache + query (for Q-Q attention)
    total_kv_len = kv_len + q_len

    block_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=q_len,
        KV_LEN=total_kv_len,
        device=device,
        _compile=_compile,
    )

    return block_mask


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
        self.first_frames_ids_ranges = None
        self.traj_and_text_ids_range = None
        self.num_views = 4
        self.num_frames_per_view = 4
        self.num_image_tokens_per_frame = 180
    
    def _set_processor(self, processor):
        """Set the processor for the model.
        """
        self.processor = processor
    
    @torch.inference_mode()
    def embed_tokens(self, 
        input_ids: torch.LongTensor, 
        pixel_values: torch.Tensor, 
        image_grid_thw: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed the input IDs, pixel values, and image grid THW into a tensor of shape (B, T, D).
        """
        
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)  # we only embed the new tokens

        if self.past_key_values is not None:
            vision_start_token_id = self.processor.tokenizer.encode('<|vision_start|>')[0]
            first_image_token_position = torch.where(input_ids == vision_start_token_id)[1][0]
            inputs_embeds = inputs_embeds[:, first_image_token_position:]  # exclude system prompt
        image_embeds, deepstack_image_embeds = self.vlm.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.vlm.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        return inputs_embeds, deepstack_image_embeds, visual_pos_masks
    
    def create_cache_position(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Create the cache position for the input IDs.
        """
        past_seen_tokens = self.past_key_values.get_seq_length() if self.past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + input_ids.shape[1], device=input_ids.device
        )
        return cache_position
    
    def retrieve_streaming_related_info(self, input_ids: torch.LongTensor) -> tuple[torch.LongTensor, torch.LongTensor]:
        """Retrieve the streaming related information for the input IDs.
        """
        past_seen_tokens = self.past_key_values.get_seq_length() if self.past_key_values is not None else 0
        vision_start_token, vision_end_token = '<|vision_start|>', '<|vision_end|>'
        vision_start_token_id, vision_end_token_id = self.processor.tokenizer.encode(vision_start_token)[0], self.processor.tokenizer.encode(vision_end_token)[0]

        start_mask = input_ids == vision_start_token_id
        end_mask = input_ids == vision_end_token_id
        vision_start_token_positions = torch.where(start_mask)[1][::self.num_frames_per_view]  # we extract the vision start token positions for the first frame of each view
        vision_end_token_positions = torch.where(end_mask)[1][::self.num_frames_per_view]  # we extract the vision end token positions for the last frame of each view
        
        first_frames_ids_ranges = [(start, end + 1) for start, end in zip(vision_start_token_positions, vision_end_token_positions)]
        
        last_image_token_position = torch.where(end_mask)[1][-1]
        traj_and_text_ids_range = (last_image_token_position + 1, past_seen_tokens)
        return first_frames_ids_ranges, traj_and_text_ids_range
    
    def update_past_key_values(self, past_key_values, first_frames_ids_ranges, traj_and_text_ids_range, device):
        """Update the past key values for the first frames and the traj and text tokens.
        """
        past_seen_tokens = past_key_values.get_seq_length()
        mask = torch.ones(past_seen_tokens, device=device, dtype=torch.bool)

        for start, end in first_frames_ids_ranges:
            mask[start:end] = False
        mask[traj_and_text_ids_range[0]:traj_and_text_ids_range[1]] = False
        
        new_past_key_values = []
        for layer in range(len(past_key_values)):
            new_keys = past_key_values[layer][0][:, :, mask, :]
            new_values = past_key_values[layer][1][:, :, mask, :]

            new_past_key_values.append((new_keys, new_values))
        
        return new_past_key_values
    
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
            pixel_values=tokenized_data["pixel_values"], 
            image_grid_thw=tokenized_data["image_grid_thw"],
        )
        cache_position = self.create_cache_position(input_ids=input_ids)

        logger.info(f"inputs_embeds: {inputs_embeds.shape}")
        logger.info(f"cache_position: {cache_position.shape}")

        if self.past_key_values is None:
            flex_attn_block_mask = create_causal_block_mask(
                q_len=inputs_embeds.shape[1],
                device=device,
            )
        else:
            first_image_token_id = self.first_frames_ids_ranges[0][0]
            kv_len = self.past_key_values[0][0].shape[2]
            flex_attn_block_mask = create_streaming_block_mask(
                sys_len=first_image_token_id,
                kv_len=kv_len,
                q_len=inputs_embeds.shape[1],
                device=device
            )
        
        vlm_outputs = self.vlm.generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            flex_attn_block_mask=flex_attn_block_mask,
            deepstack_visual_embeds=deepstack_visual_embeds,
            visual_pos_masks=visual_pos_masks,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            past_key_values=self.past_key_values,
            cache_position=cache_position,
            **tokenized_data,
        )
        vlm_outputs.rope_deltas = self.vlm.model.rope_deltas

        # manually replace padding after EOS token
        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()

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
            (b_star, 1, n_diffusion_tokens, prompt_cache.get_seq_length() + n_diffusion_tokens),
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
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            # crop the prompt cache to remove the newly added tokens
            prompt_cache.crop(prefill_seq_len)
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

        first_frames_ids_ranges, traj_and_text_ids_range = self.retrieve_streaming_related_info(input_ids=input_ids)
        self.first_frames_ids_ranges = first_frames_ids_ranges
        self.traj_and_text_ids_range = traj_and_text_ids_range
        logger.info(f"first_frames_ids_ranges: {first_frames_ids_ranges}")
        logger.info(f"traj_and_text_ids_range: {traj_and_text_ids_range}")
        self.past_key_values = self.update_past_key_values(
            past_key_values=prompt_cache, 
            first_frames_ids_ranges=first_frames_ids_ranges, 
            traj_and_text_ids_range=traj_and_text_ids_range,
            device=device,
        )
        logger.info(f"past_key_values: {self.past_key_values[0][0].shape}")

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
