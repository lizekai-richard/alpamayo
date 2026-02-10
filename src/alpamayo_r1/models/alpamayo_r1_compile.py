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

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


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

    def _encode(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if not hasattr(self, "_encode_fn"):
            # Initialize static buffers
            self._encode_pixel_values = torch.empty_like(pixel_values)
            self._encode_image_grid_thw = torch.empty_like(image_grid_thw)

            def encode_fn():
                pixels = self._encode_pixel_values.type(self.vlm.model.visual.dtype)
                return self.vlm.model.visual(
                    pixels, 
                    grid_thw=self._encode_image_grid_thw,
                    target_layers=[25, 26],
                )

            self._encode_fn = encode_fn

        # Copy inputs to static buffers
        self._encode_pixel_values.copy_(pixel_values)
        self._encode_image_grid_thw.copy_(image_grid_thw)

        # Warmup and compile on first call (if enabled)
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
    ) -> torch.Tensor:
        if not hasattr(self, "_prefill_fn"):
            # Initialize static buffers
            self._prefill_inputs_embeds = torch.empty_like(inputs_embeds)
            self._prefill_position_ids = torch.empty_like(position_ids)
            self._prefill_cache_position = torch.empty_like(cache_position)
            self._prefill_visual_pos_masks = torch.empty_like(visual_pos_masks)
            self._prefill_deepstack_embeds = [torch.empty_like(e) for e in deepstack_image_embeds]

            def prefill_fn():
                hidden = self.vlm.model.language_model(
                    inputs_embeds=self._prefill_inputs_embeds,
                    position_ids=self._prefill_position_ids,
                    past_key_values=self._past_key_values,
                    cache_position=self._prefill_cache_position,
                    visual_pos_masks=self._prefill_visual_pos_masks,
                    deepstack_visual_embeds=self._prefill_deepstack_embeds,
                    use_cache=True,
                ).last_hidden_state[:, -1]
                return self.vlm.lm_head(hidden)

            self._prefill_fn = prefill_fn

        # Copy inputs to static buffers
        self._prefill_inputs_embeds.copy_(inputs_embeds)
        self._prefill_position_ids.copy_(position_ids)
        self._prefill_cache_position.copy_(cache_position)
        self._prefill_visual_pos_masks.copy_(visual_pos_masks)
        for buf, emb in zip(self._prefill_deepstack_embeds, deepstack_image_embeds):
            buf.copy_(emb)

        # Warmup and compile on first call (if enabled)
        if not hasattr(self, "_compiled_prefill_fn"):
            if self._torch_compile:
                self._prefill_fn()  # Warmup
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
        if not hasattr(self, "_decode_fn"):
            # Initialize static buffers
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

        # Copy inputs to static buffers
        self._decode_input_ids.copy_(input_ids)
        self._decode_position_ids.copy_(position_ids)
        self._decode_cache_position.copy_(cache_position)

        # Warmup and compile on first call (if enabled)
        if not hasattr(self, "_compiled_decode_fn"):
            if self._torch_compile:
                self._decode_fn()  # Warmup
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
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        diffusion_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if not hasattr(self, "_action_fn"):
            # Initialize static buffers
            self._action_position_ids = torch.empty_like(position_ids)
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
                    position_ids=self._action_position_ids,
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
        self._action_position_ids.copy_(position_ids)
        self._action_attention_mask.copy_(attention_mask)
        self._action_cache_position.copy_(cache_position)

        # Generate noise outside compiled graph for deterministic RNG
        self._action_noise.normal_()

        # Warmup and compile on first call (if enabled)
        if not hasattr(self, "_compiled_action_fn"):
            if self._torch_compile:
                self._action_fn()  # Warmup
                self._compiled_action_fn = torch.compile(
                    self._action_fn, mode=self._torch_compile, fullgraph=True
                )
            else:
                self._compiled_action_fn = self._action_fn

        if self._torch_compile:
            torch.compiler.cudagraph_mark_step_begin()
        return self._compiled_action_fn()
    
    # ==================== Token Pruning ====================
    
    def _merge_colsum(self, colsums, image_grid_thw):
        # colsums is a list of tensors, each tensor is of shape [B, H, L].
        colsums = torch.stack(colsums, dim=0)  # [num_layers, B, H, L]
        colsums = colsums.mean(dim=(0, 2))  # [B, L]
        B, H, W = colsums.shape[0], image_grid_thw[0, 1], image_grid_thw[0, 2]

        ratio = self.vlm.config.vision_config.spatial_merge_size
        colsums = colsums.view(B, H // ratio, ratio, W // ratio, ratio)
        colsums = colsums.sum(dim=(2, 4)).reshape(B, -1)
        return colsums
    
    def _prune_tokens(self, colsums: torch.Tensor, sparsity_ratio: float) -> torch.Tensor:
        """Select top-k tokens per image based on colsum importance scores.

        Args:
            colsums: [num_images, tokens_per_image] importance scores.
            sparsity_ratio: Fraction of tokens to remove (0.5 = keep 50%).

        Returns:
            Sorted indices of kept tokens, shape [num_images, K].
        """
        num_tokens = colsums.shape[-1]
        num_keep = int(num_tokens * (1 - sparsity_ratio))
        _, indices = torch.topk(colsums, num_keep, dim=-1)
        return indices.sort(dim=-1).values

    def _get_pruned_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: torch.LongTensor,
        token_indices: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct mRoPE position_ids after token pruning.

        Mirrors Qwen3VLModel.get_rope_index but replaces the full T×H×W grid
        with only the kept tokens, remapping h and w independently to contiguous
        0..N-1 values.

        Args:
            input_ids: [B, seq_len] original (unpruned) input IDs.
            image_grid_thw: [num_images, 3] with (T, H, W) per image.
            token_indices: [num_images, K] sorted kept token indices per image.

        Returns:
            position_ids: [3, B, new_seq_len] with remapped mRoPE positions.
            rope_deltas: [B, 1] position delta for decode phase.
            keep_mask: [B, seq_len] bool mask (True = keep this position).
        """
        spatial_merge_size = self.vlm.config.vision_config.spatial_merge_size
        image_token_id = self.vlm.config.image_token_id
        vision_start_token_id = self.vlm.config.vision_start_token_id

        total_input_ids = input_ids
        batch_size, seq_len = total_input_ids.shape
        keep_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)
        K = token_indices.shape[1]

        mrope_position_deltas = []
        all_new_position_ids = []
        image_index = 0

        for i in range(batch_size):
            ids = total_input_ids[i]
            input_tokens = ids.tolist()

            # Count images
            vision_start_indices = torch.argwhere(ids == vision_start_token_id).squeeze(1)
            vision_tokens = ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum().item()

            llm_pos_ids_list: list = []
            st = 0

            for _ in range(image_nums):
                ed = input_tokens.index(image_token_id, st)

                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                llm_grid_h = h.item() // spatial_merge_size
                llm_grid_w = w.item() // spatial_merge_size
                original_num_tokens = t.item() * llm_grid_h * llm_grid_w

                # --- Text segment before this image (includes <|vision_start|>) ---
                text_len = ed - st
                if text_len > 0:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
                    )

                # --- Pruned image tokens ---
                kept = token_indices[image_index]  # [K], sorted indices within this image
                h_orig = kept // llm_grid_w
                w_orig = kept % llm_grid_w

                # Remap each dimension to contiguous 0..N-1
                h_unique = torch.unique(h_orig, sorted=True)
                w_unique = torch.unique(w_orig, sorted=True)
                h_new = torch.searchsorted(h_unique, h_orig)
                w_new = torch.searchsorted(w_unique, w_orig)
                t_new = torch.zeros_like(h_new)

                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(
                    torch.stack([t_new, h_new, w_new]) + text_len + st_idx
                )

                # --- Update keep_mask: keep first K of the image placeholders ---
                keep_mask[i, ed + K : ed + original_num_tokens] = False

                # Advance past all original image tokens in input_ids
                st = ed + original_num_tokens
                image_index += 1

            # --- Trailing text segment ---
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            all_new_position_ids.append(llm_positions)

            new_seq_len = keep_mask[i].sum().item()
            mrope_position_deltas.append(llm_positions.max() + 1 - new_seq_len)

        # Stack across batch
        position_ids = torch.stack(all_new_position_ids, dim=1).to(input_ids.device)  # [3, B, new_seq_len]
        rope_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)  # [B, 1]

        return position_ids, rope_deltas, keep_mask

    def _prune_embeddings(
        self,
        image_embeds: torch.Tensor,
        deepstack_image_embeds: list[torch.Tensor],
        image_grid_thw: torch.LongTensor,
        token_indices: torch.LongTensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Prune image embeddings to keep only selected tokens per image.

        Args:
            image_embeds: [total_img_tokens, hidden_dim] all images concatenated.
            deepstack_image_embeds: list of [total_img_tokens, hidden_dim] tensors.
            image_grid_thw: [num_images, 3] with (T, H, W) per image.
            token_indices: [num_images, K] sorted kept indices per image.

        Returns:
            Pruned image_embeds and deepstack_image_embeds with same structure.
        """
        spatial_merge_size = self.vlm.config.vision_config.spatial_merge_size
        tokens_per_image = (
            image_grid_thw[:, 0]
            * (image_grid_thw[:, 1] // spatial_merge_size)
            * (image_grid_thw[:, 2] // spatial_merge_size)
        )  # [num_images]

        # Split by image, select kept tokens, concatenate back
        per_image = image_embeds.split(tokens_per_image.tolist(), dim=0)
        pruned = torch.cat([emb[idx] for emb, idx in zip(per_image, token_indices)], dim=0)

        pruned_deepstack = []
        for ds_embeds in deepstack_image_embeds:
            per_image_ds = ds_embeds.split(tokens_per_image.tolist(), dim=0)
            pruned_deepstack.append(
                torch.cat([emb[idx] for emb, idx in zip(per_image_ds, token_indices)], dim=0)
            )

        return pruned, pruned_deepstack

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

    def sample_trajectories_from_data_with_vlm_rollout(
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
        """Sample trajectories from the data with VLM rollout.

        Args:
            data: The input data.
            top_p: The top-p value for sampling.
            top_k: The top-k value for sampling.
            temperature: The temperature for sampling.
            num_traj_samples: The number of trajectory samples.
            num_traj_sets: The number of trajectory sets.
            diffusion_kwargs: Additional kwargs for diffusion sampling.
            torch_compile: torch.compile mode (e.g. "max-autotune", "reduce-overhead").
                "max-autotune" is recommended.
            fuse_qkv: Whether to fuse the QKV linear layers.
            fuse_gate_up: Whether to fuse the gate and up linear layers.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pred_xyz: The predicted xyz.
            pred_rot: The predicted rotation.
            logprob: The log probability.
        """
        # Patch model for torch.compile on first call
        self._torch_compile = torch_compile
        if self._torch_compile and not hasattr(self, "_patched_for_compile"):
            patch_for_torch_compile(self, mode="non_streaming", fuse_qkv=fuse_qkv, fuse_gate_up=fuse_gate_up)
            self._patched_for_compile = True
            del self.expert.embed_tokens

        # Extract inputs
        tokenized = data["tokenized_data"]
        input_ids = tokenized["input_ids"]
        pixel_values = tokenized["pixel_values"]
        image_grid_thw = tokenized["image_grid_thw"]
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]

        batch_size, num_traj_groups, _, _ = ego_history_xyz.shape
        num_samples = num_traj_samples * num_traj_sets
        assert num_traj_groups == 1, "Only one trajectory group is supported for inference."
        device = input_ids.device

        # Fuse trajectory tokens into input
        input_ids = self.fuse_traj_tokens(
            input_ids, {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot}
        )

        # Setup generation
        max_new_tokens = kwargs.get("max_generation_length", self.config.tokens_per_future_traj)
        logits_processor = self._build_logits_processor(temperature, top_k, top_p)

        # ===== Encode + Prune =====
        image_embeds, deepstack_image_embeds, colsums = self._encode(pixel_values, image_grid_thw)
        colsums = self._merge_colsum(colsums, image_grid_thw)
        token_indices = self._prune_tokens(colsums, sparsity_ratio=0.5)
        logger.info(f"token_indices: {token_indices.shape}")

        # Prune image embeddings
        image_embeds, deepstack_image_embeds = self._prune_embeddings(
            image_embeds, deepstack_image_embeds, image_grid_thw, token_indices
        )

        # Reconstruct position IDs with contiguous h/w remapping
        position_ids, rope_deltas, keep_mask = self._get_pruned_rope_index(
            input_ids, image_grid_thw, token_indices
        )
        logger.info(f"position_ids: {position_ids.shape}")
        logger.info(f"rope_deltas: {rope_deltas}")

        # Prune input_ids using keep_mask and recompute embeddings
        input_ids = input_ids[keep_mask].view(batch_size, -1)
        inputs_embeds = self.vlm.model.get_input_embeddings()(input_ids)
        seq_len = input_ids.shape[1]
        logger.info(f"seq_len: {seq_len}")

        # Scatter pruned image embeds into pruned inputs_embeds
        image_mask = (
            (input_ids == self.vlm.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        logger.info(f"inputs_embeds: {inputs_embeds.shape}")

        # Initialize KV cache (includes space for action tokens)
        if not hasattr(self, "_past_key_values"):
            self._past_key_values = StaticCache(
                config=self.vlm.config,
                max_cache_len=seq_len + max_new_tokens + self.num_action_tokens,
                max_batch_size=num_samples,
            )
        self._past_key_values.reset()

        # ===== Prefill =====
        logits = self._prefill(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            cache_position=torch.arange(seq_len, device=device),
            visual_pos_masks=image_mask[..., 0],
            deepstack_image_embeds=deepstack_image_embeds,
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
        # Note: Action only attends to prompt tokens, NOT reasoning tokens (they are masked out).
        # This is by design - the expert model conditions only on the original prompt.
        action_start_pos = traj_start_pos + 1

        # Build position_ids for action tokens
        position_ids = torch.arange(self.num_action_tokens, device=device)
        position_ids = einops.repeat(position_ids, "t -> 3 b t", b=batch_size * num_samples).clone()
        position_ids += (rope_deltas + action_start_pos[None, :, None]).to(device)

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
            position_ids=position_ids,
            attention_mask=attention_mask,
            cache_position=cache_position,
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

        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, output_ids)
            for key in extra:
                extra[key] = np.array(extra[key]).reshape(
                    [batch_size, num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot

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


AutoConfig.register("alpamayo_r1", AlpamayoR1Config)
AutoModel.register(AlpamayoR1Config, AlpamayoR1)
