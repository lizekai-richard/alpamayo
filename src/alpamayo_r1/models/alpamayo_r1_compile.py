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
from transformers import AutoConfig, AutoModel, StaticCache
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
from alpamayo_r1.models.qwen3vl_patches import patch_qwen3vl_for_inference
from alpamayo_r1.models.token_utils import (
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
                return self.vlm.model.visual(pixels, grid_thw=self._encode_image_grid_thw)

            self._encode_fn = encode_fn

        # Copy inputs to static buffers
        self._encode_pixel_values.copy_(pixel_values)
        self._encode_image_grid_thw.copy_(image_grid_thw)

        # Warmup and compile on first call (if enabled)
        if not hasattr(self, "_compiled_encode_fn"):
            if self._use_compile:
                self._encode_fn()  # Warmup for compile
                self._compiled_encode_fn = torch.compile(
                    self._encode_fn, mode="max-autotune", fullgraph=True
                )
            else:
                self._compiled_encode_fn = self._encode_fn

        if self._use_compile:
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
            if self._use_compile:
                self._prefill_fn()  # Warmup for compile
                self._compiled_prefill_fn = torch.compile(
                    self._prefill_fn, mode="max-autotune", fullgraph=True
                )
            else:
                self._compiled_prefill_fn = self._prefill_fn

        if self._use_compile:
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
            if self._use_compile:
                self._decode_fn()  # Warmup for compile
                self._compiled_decode_fn = torch.compile(
                    self._decode_fn, mode="max-autotune", fullgraph=True
                )
            else:
                self._compiled_decode_fn = self._decode_fn

        if self._use_compile:
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
            if self._use_compile:
                self._action_fn()  # Warmup for compile
                self._compiled_action_fn = torch.compile(
                    self._action_fn, mode="max-autotune", fullgraph=True
                )
            else:
                self._compiled_action_fn = self._action_fn

        if self._use_compile:
            torch.compiler.cudagraph_mark_step_begin()
        return self._compiled_action_fn()

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
        use_compile: bool = True,
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
            use_compile: Whether to use torch.compile for inference optimization.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pred_xyz: The predicted xyz.
            pred_rot: The predicted rotation.
            logprob: The log probability.
        """
        # Patch model for inference on first call
        self._use_compile = use_compile
        if self._use_compile and not hasattr(self, "_inference_initialized"):
            patch_qwen3vl_for_inference(self)
            self._inference_initialized = True

        # Extract inputs
        tokenized = data["tokenized_data"]
        input_ids = tokenized["input_ids"]
        pixel_values = tokenized["pixel_values"]
        image_grid_thw = tokenized["image_grid_thw"]
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]

        batch_size, num_traj_groups, _, _ = ego_history_xyz.shape
        assert num_traj_groups == 1, "Only one trajectory group is supported for inference."
        device = input_ids.device

        # Fuse trajectory tokens into input
        input_ids = self.fuse_traj_tokens(
            input_ids, {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot}
        )

        # Setup generation
        max_new_tokens = kwargs.get("max_generation_length", self.config.tokens_per_future_traj)
        logits_processor = self._build_logits_processor(temperature, top_k, top_p)

        # Initialize positions and embeddings
        seq_len = input_ids.shape[1]
        position_ids, rope_deltas = self.vlm.model.get_rope_index(input_ids, image_grid_thw)
        inputs_embeds = self.vlm.model.get_input_embeddings()(input_ids)

        # ===== Encode =====
        image_embeds, deepstack_image_embeds = self._encode(pixel_values, image_grid_thw)

        image_mask = (
            (input_ids == self.vlm.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # Initialize KV cache (includes space for action tokens)
        if not hasattr(self, "_past_key_values"):
            self._past_key_values = StaticCache(
                config=self.vlm.config,
                max_cache_len=seq_len + max_new_tokens + self.num_action_tokens,
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
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
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
        num_samples = num_traj_samples * num_traj_sets
        action_start_pos = traj_start_pos + 1

        # Build position_ids for action tokens
        position_ids = torch.arange(self.num_action_tokens, device=device)
        position_ids = einops.repeat(position_ids, "t -> 3 b t", b=batch_size).clone()
        position_ids += (rope_deltas + action_start_pos[:, None]).to(device)

        # Build attention mask: attend to prompt only, mask out reasoning tokens
        indices = torch.arange(self._past_key_values.max_cache_len, device=device)
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
