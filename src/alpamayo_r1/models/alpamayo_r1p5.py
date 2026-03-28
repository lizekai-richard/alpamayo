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
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.generation.logits_process import LogitsProcessor

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
from alpamayo_r1.nav_utils import remove_nav_text
from alpamayo_r1.utils import patch_for_baseline

logger = logging.getLogger(__name__)


class ExpertLogitsProcessor(LogitsProcessor):
    """Masks out the logits for discrete trajectory tokens."""

    def __init__(self, traj_token_offset: int, traj_vocab_size: int):
        super().__init__()
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
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

    @staticmethod
    def _find_eos_offset(
        sequences: torch.Tensor,
        eos_token_id: int,
        device: torch.device,
        warn: bool = True,
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
        """Build position IDs and 4D attention mask for the expert denoiser."""
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
            input_mask = prefix_mask[:, None, None, :]
            attention_mask[:, :, :, : input_mask.shape[-1]] = torch.where(
                input_mask == 0,
                torch.finfo(attention_mask.dtype).min,
                attention_mask[:, :, :, : input_mask.shape[-1]],
            )

        return position_ids, attention_mask

    def sample_trajectories_from_data_with_vlm_rollout(
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
        """Sample trajectories from the data with VLM rollout."""
        patch_for_baseline(self)
        data = copy.deepcopy(data)
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

        # Timing events:
        #   _ev[0]: encode start
        #   _ev[1]: encode end / prefill start
        #   _ev[2]: prefill end / decode start
        #   _ev[3]: decode end / action start
        #   _ev[4]: action end
        _ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]

        # Register hooks to split encode / prefill / decode inside vlm.generate()
        _vlm_fwd_count = [0]

        def _visual_pre_hook(module, args):
            _ev[0].record()

        def _visual_post_hook(module, args, output):
            _ev[1].record()

        def _vlm_post_hook(module, args, output):
            _vlm_fwd_count[0] += 1
            if _vlm_fwd_count[0] == 1:
                _ev[2].record()  # first forward = prefill done

        _hooks = [
            self.vlm.model.visual.register_forward_pre_hook(_visual_pre_hook),
            self.vlm.model.visual.register_forward_hook(_visual_post_hook),
            self.vlm.register_forward_hook(_vlm_post_hook),
        ]

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
        vlm_outputs = self.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **tokenized_data,
        )
        vlm_outputs.rope_deltas = self.vlm.model.rope_deltas
        _ev[3].record()  # decode end
        # Clean up hooks
        for h in _hooks:
            h.remove()
        num_decode_tokens = vlm_outputs.sequences.shape[1] - input_ids.shape[1]

        # manually replace padding after EOS token
        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()

        b_star = vlm_outputs.sequences.shape[0]
        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
        offset = self._find_eos_offset(
            sequences=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            device=device,
        )
        prefix_mask = tokenized_data.get("attention_mask")
        if prefix_mask is not None:
            prefix_mask = torch.repeat_interleave(prefix_mask, n_samples_total, dim=0)
        position_ids, attention_mask = self._build_expert_pos_ids_and_attn_mask(
            offset=offset,
            rope_deltas=vlm_outputs.rope_deltas,
            kv_cache_seq_len=prefill_seq_len,
            n_diffusion_tokens=n_diffusion_tokens,
            b_star=b_star,
            device=device,
            prefix_mask=prefix_mask,
        )

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        # 2) Define denoising step
        def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            b_star = x.shape[0]
            future_token_embeds = self.action_in_proj(x, t)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)

            expert_out_base = self.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            prompt_cache.crop(prefill_seq_len)
            last_hidden = expert_out_base.last_hidden_state[:, -n_diffusion_tokens:]
            pred = self.action_out_proj(last_hidden).view(
                -1, *self.action_space.get_action_space_dims()
            )
            return pred

        # 3) Diffusion sampling
        total_batch = B * n_samples_total
        if diffusion_kwargs is None:
            diffusion_kwargs = {}

        action_noise = torch.randn(
            total_batch, *self.action_space.get_action_space_dims(), device=device, dtype=torch.bfloat16
        )

        sampled_action = self.diffusion.sample(
            noise=action_noise,
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

        _ev[4].record()  # action end

        # 4) Reshape to (B, num_traj_samples, n_traj, ...)
        pred_xyz = einops.rearrange(
            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )
        pred_rot = einops.rearrange(
            pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )

        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)
            for text_tokens in extra.keys():
                extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                    [input_ids.shape[0], num_traj_sets, num_traj_samples]
                )
            torch.cuda.synchronize()
            extra["timing"] = {
                "encode_time_ms": _ev[0].elapsed_time(_ev[1]),
                "prefill_time_ms": _ev[1].elapsed_time(_ev[2]),
                "decode_time_ms": _ev[2].elapsed_time(_ev[3]),
                "action_time_ms": _ev[3].elapsed_time(_ev[4]),
                "total_time_ms": _ev[0].elapsed_time(_ev[4]),
                "num_decode_tokens": num_decode_tokens,
            }
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot

    @torch.no_grad()
    def sample_trajectories_from_data_with_vlm_rollout_cfg_nav(
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
        """Sample trajectories with classifier-free guidance on navigation."""
        data = copy.deepcopy(data)
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
        vlm_outputs = self.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **tokenized_data,
        )
        del vlm_outputs.logits
        torch.cuda.empty_cache()
        vlm_outputs.rope_deltas = self.vlm.model.rope_deltas

        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_cache = vlm_outputs.past_key_values

        b_star = vlm_outputs.sequences.shape[0]
        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
        offset = self._find_eos_offset(
            sequences=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            device=device,
        )
        prefix_mask = tokenized_data.get("attention_mask")
        if prefix_mask is not None:
            prefix_mask = torch.repeat_interleave(prefix_mask, n_samples_total, dim=0)
        position_ids, attention_mask = self._build_expert_pos_ids_and_attn_mask(
            offset=offset,
            rope_deltas=vlm_outputs.rope_deltas,
            kv_cache_seq_len=prompt_cache.get_seq_length(),
            n_diffusion_tokens=n_diffusion_tokens,
            b_star=b_star,
            device=device,
            prefix_mask=prefix_mask,
        )

        # Build unguided KV cache
        unguided_input_ids = []
        for i in range(input_ids.shape[0]):
            unguided_input_ids.append(remove_nav_text(input_ids, self.tokenizer, i)[0])
        unguided_input_ids = torch.nn.utils.rnn.pad_sequence(
            unguided_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            padding_side="left",
        ).to(device)
        unguided_prefix_mask = unguided_input_ids.ne(self.tokenizer.pad_token_id).long()

        unguided_prefill_outputs = self.vlm(
            input_ids=unguided_input_ids,
            attention_mask=unguided_prefix_mask,
            image_grid_thw=tokenized_data.get("image_grid_thw"),
            pixel_values=tokenized_data.get("pixel_values"),
            use_cache=True,
            logits_to_keep=1,
        )

        unguided_prompt_cache = unguided_prefill_outputs.past_key_values
        del unguided_prefill_outputs
        torch.cuda.empty_cache()
        unguided_prompt_cache.batch_repeat_interleave(n_samples_total)

        generated_tokens = vlm_outputs.sequences[:, input_ids.shape[1] :]
        unguided_prefix_len = unguided_input_ids.shape[1]
        gen_len = generated_tokens.shape[1]

        prefix_mask_repeated = unguided_prefix_mask.repeat_interleave(n_samples_total, dim=0)
        gen_mask = generated_tokens.ne(self.tokenizer.pad_token_id).long()
        full_attention_mask = torch.cat([prefix_mask_repeated, gen_mask], dim=1)

        cache_position = torch.arange(
            unguided_prefix_len,
            unguided_prefix_len + gen_len,
            device=device,
            dtype=torch.long,
        )

        unguided_vlm_outputs = self.vlm(
            input_ids=generated_tokens,
            attention_mask=full_attention_mask,
            past_key_values=unguided_prompt_cache,
            cache_position=cache_position,
            use_cache=True,
            logits_to_keep=1,
        )
        unguided_prompt_cache = unguided_vlm_outputs.past_key_values
        del unguided_vlm_outputs.logits
        torch.cuda.empty_cache()

        full_unguided_tokens = torch.cat(
            [torch.repeat_interleave(unguided_input_ids, n_samples_total, dim=0), generated_tokens],
            dim=1,
        )
        unguided_offset = self._find_eos_offset(
            sequences=full_unguided_tokens,
            eos_token_id=eos_token_id,
            device=device,
            warn=False,
        )
        unguided_prefix_mask_repeated = torch.repeat_interleave(
            unguided_prefix_mask, n_samples_total, dim=0
        )
        unguided_position_ids, unguided_attention_mask = self._build_expert_pos_ids_and_attn_mask(
            offset=unguided_offset,
            rope_deltas=unguided_vlm_outputs.rope_deltas,
            kv_cache_seq_len=unguided_prompt_cache.get_seq_length(),
            n_diffusion_tokens=n_diffusion_tokens,
            b_star=b_star,
            device=device,
            prefix_mask=unguided_prefix_mask_repeated,
        )

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        def step_fn(
            x: torch.Tensor,
            t: torch.Tensor,
            position_ids: torch.Tensor,
            past_key_values: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            b_star = x.shape[0]
            future_token_embeds = self.action_in_proj(x, t)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)

            prefill_seq_len = past_key_values.get_seq_length()
            expert_out_base = self.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            past_key_values.crop(prefill_seq_len)
            last_hidden = expert_out_base.last_hidden_state[:, -n_diffusion_tokens:]
            pred = self.action_out_proj(last_hidden).view(
                -1, *self.action_space.get_action_space_dims()
            )
            return pred

        total_batch = B * n_samples_total
        if diffusion_kwargs is None:
            diffusion_kwargs = {}

        sampled_action = self.diffusion.sample(
            batch_size=total_batch,
            step_fn=partial(
                step_fn,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ),
            unguided_step_fn=partial(
                step_fn,
                past_key_values=unguided_prompt_cache,
                attention_mask=unguided_attention_mask,
                position_ids=unguided_position_ids,
            ),
            device=device,
            return_all_steps=False,
            **diffusion_kwargs,
        )

        hist_xyz_rep = einops.repeat(
            ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )
        hist_rot_rep = einops.repeat(
            ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )

        pred_xyz, pred_rot = self.action_space.action_to_traj(
            sampled_action, hist_xyz_rep, hist_rot_rep
        )

        pred_xyz = einops.rearrange(
            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )
        pred_rot = einops.rearrange(
            pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )

        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)
            for text_tokens in extra.keys():
                extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                    [input_ids.shape[0], num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot


AutoConfig.register("alpamayo1_5", Alpamayo1_5Config)
AutoModel.register(Alpamayo1_5Config, Alpamayo1_5)
