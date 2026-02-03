from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3vl
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision
from transformers.models.qwen3_vl.modeling_qwen3_vl import rotate_half
from transformers.integrations.flex_attention import flex_attention_forward
from transformers.integrations.sdpa_attention import sdpa_attention_forward


class Qwen3VLVisionPatchEmbed(qwen3vl.Qwen3VLVisionPatchEmbed):
    def __init__(self, config) -> None:
        nn.Module.__init__(self)
        self.in_features = (
            config.in_channels * config.temporal_patch_size * config.patch_size**2
        )
        self.proj = nn.Linear(self.in_features, config.hidden_size, bias=True)

        # Hook to convert Conv3d weights [out, in, t, h, w] -> Linear [out, in*t*h*w]
        def convert_conv3d_weights(state_dict, prefix, *args):
            weight_key = prefix + "weight"
            if weight_key in state_dict and state_dict[weight_key].ndim == 5:
                state_dict[weight_key] = state_dict[weight_key].flatten(1).contiguous()

        self.proj._register_load_state_dict_pre_hook(convert_conv3d_weights, with_module=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states.reshape(-1, self.in_features))


class Qwen3VLVisionAttention(qwen3vl.Qwen3VLVisionAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]

        if not hasattr(self, "_num_chunks"):
            self._num_chunks = cu_seqlens.numel() - 1

        qkv = self.qkv(hidden_states).reshape(seq_len, 3, self.num_heads, -1)
        query, key, value = qkv.permute(1, 0, 2, 3).unbind(0)

        query, key = apply_rotary_pos_emb_vision(query, key, *position_embeddings)

        chunk_size = seq_len // self._num_chunks
        query = query.reshape(self._num_chunks, chunk_size, self.num_heads, -1).transpose(1, 2)
        key = key.reshape(self._num_chunks, chunk_size, self.num_heads, -1).transpose(1, 2)
        value = value.reshape(self._num_chunks, chunk_size, self.num_heads, -1).transpose(1, 2)

        output = F.scaled_dot_product_attention(query, key, value, scale=self.scaling)
        return self.proj(output.transpose(1, 2).reshape(seq_len, -1))


class Qwen3VLVisionModel(qwen3vl.Qwen3VLVisionModel):
    def _init_caches(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> None:
        seq_len = hidden_states.size(0)

        self._cached_pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

        rotary_emb = self.rot_pos_emb(grid_thw).reshape(seq_len, -1)
        rotary_emb = torch.cat((rotary_emb, rotary_emb), dim=-1)
        self._cached_position_embeddings = (rotary_emb.cos(), rotary_emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        cu_seqlens = cu_seqlens.cumsum(dim=0, dtype=torch.int32)
        self._cached_cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)

        if not hasattr(self, "_cached_pos_embeds"):
            self._init_caches(hidden_states, grid_thw)

        hidden_states = (hidden_states + self._cached_pos_embeds).reshape(hidden_states.size(0), -1)

        deepstack_features = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=self._cached_cu_seqlens,
                position_embeddings=self._cached_position_embeddings,
            )
            if layer_idx in self.deepstack_visual_indexes:
                merger_idx = self.deepstack_visual_indexes.index(layer_idx)
                deepstack_features.append(self.deepstack_merger_list[merger_idx](hidden_states))

        return self.merger(hidden_states), deepstack_features


def apply_mrope_emb_single(tensor, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to a single tensor."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    tensor_embed = (tensor * cos) + (rotate_half(tensor) * sin)
    return tensor_embed


class Qwen3VLTextAttention(qwen3vl.Qwen3VLTextAttention):
    def forward(
        self,
        hidden_states=None,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        cos_q, sin_q = cos[:, cache_position, :], sin[:, cache_position, :]
            
        # store the un-roped keys and values
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {'cache_position': cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        query_states = apply_mrope_emb_single(query_states, cos_q, sin_q)
        key_states = apply_mrope_emb_single(key_states, cos, sin)

        attn_output, attn_weights = sdpa_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3VLTextModel(qwen3vl.Qwen3VLTextModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        position_ids = position_ids[0]

        if inputs_embeds.shape[1] > 1:
            attention_mask = create_causal_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

        if deepstack_visual_embeds is not None and not hasattr(self, "_cached_deepstack_indices"):
            self._cached_deepstack_indices = visual_pos_masks.flatten().nonzero(as_tuple=True)[0]

        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                flat_hidden = hidden_states.view(-1, hidden_states.shape[-1])
                flat_hidden.index_add_(
                    0, self._cached_deepstack_indices, deepstack_visual_embeds[layer_idx]
                )

        return BaseModelOutputWithPast(
            last_hidden_state=self.norm(hidden_states), past_key_values=past_key_values
        )


_PATCHED_CLASSES = {
    "Qwen3VLVisionModel": Qwen3VLVisionModel,
    "Qwen3VLVisionPatchEmbed": Qwen3VLVisionPatchEmbed,
    "Qwen3VLVisionAttention": Qwen3VLVisionAttention,
    "Qwen3VLTextAttention": Qwen3VLTextAttention,
    "Qwen3VLTextModel": Qwen3VLTextModel,
}


def _get_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    param = next(module.parameters(), None)
    if param is None:
        return torch.device("cpu"), torch.float32
    return param.device, param.dtype


def _replace_module(
    model: nn.Module, module_path: str, old_module: nn.Module, new_class: type
) -> None:
    *parent_parts, name = module_path.split(".")
    parent = model
    for part in parent_parts:
        parent = getattr(parent, part)

    config = getattr(old_module, "config", None) or getattr(parent, "config", None)
    device, dtype = _get_device_dtype(old_module)

    if new_class is Qwen3VLTextAttention:
        layer_idx = getattr(old_module, "layer_idx", None)
        new_module = new_class(config, layer_idx)
    else:
        new_module = new_class(config)
    new_module.load_state_dict(old_module.state_dict(), assign=True)
    if device.type != "meta":
        new_module = new_module.to(device=device, dtype=dtype)

    setattr(parent, name, new_module)


def patch_for_torch_compile(model: nn.Module) -> None:
    """Patch Qwen3-VL modules for torch.compile compatibility."""
    for class_name, patched_class in _PATCHED_CLASSES.items():
        for module_path, module in model.named_modules():
            if type(module).__name__ == class_name:
                _replace_module(model, module_path, module, patched_class)
