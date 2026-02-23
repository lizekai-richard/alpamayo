import logging

import torch
import torch.nn as nn
import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3vl
import transformers.cache_utils as cache_utils
from transformers.utils import is_torchdynamo_compiling
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import rotate_half
from transformers.integrations.sdpa_attention import sdpa_attention_forward

logger = logging.getLogger(__name__)


class StaticLayer(cache_utils.CacheLayerMixin):

    def __init__(self, max_cache_len, max_batch_size=1):
        super().__init__()
        self.max_cache_len = max_cache_len
        self._max_batch_size = max_batch_size

    def lazy_initialization(self, key_states, value_states):

        self.dtype, self.device = key_states.dtype, key_states.device
        # Use pre-configured max_batch_size (allows pre-allocating for multi-sample decode)
        self.max_batch_size = max(self._max_batch_size, key_states.shape[0])
        self.num_heads = key_states.shape[1]
        self.v_head_dim = value_states.shape[-1]
        self.k_head_dim = key_states.shape[-1]

        self.keys = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.k_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.v_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        # Note: `mark_static_address` is used to tag the cache as a fixed data pointer, preventing compiled graph
        # breaks when updating the cache. However, it is not supported when tracing the graph, so we skip it in this case.
        # As prefill should never be compiled, this is not an issue and it will still be run (except when users compile
        # prefill explicitly, but this should be avoided!)
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)

        self.is_initialized = True

    def expand_batch(self):
        """Copy batch 0's KV to all other batch slots.

        Call this after prefill (batch=1) and before multi-sample decode,
        so all samples start with the same prompt KV.
        """
        if self.max_batch_size > 1 and self.is_initialized:
            self.keys[1:].copy_(self.keys[:1].expand(self.max_batch_size - 1, -1, -1, -1))
            self.values[1:].copy_(self.values[:1].expand(self.max_batch_size - 1, -1, -1, -1))

    def update(
        self,
        key_states,
        value_states,
        cache_kwargs=None,
    ):
        """
        Update the key and value caches in-place.

        Handles batch size mismatch:
          - src_batch < dst_batch (prefill batch=1 into multi-batch cache): write batch 0 only
          - src_batch == dst_batch (decode with num_samples): normal update
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        cache_position = (
            cache_position if cache_position is not None else torch.arange(key_states.shape[-2], device=self.device)
        )

        # Update the cache
        src_batch_size = key_states.shape[0]
        dst_batch_size = self.keys.shape[0]

        if src_batch_size < dst_batch_size:
            # Prefill (batch=1) writing into larger cache: only update batch 0
            self.keys[:src_batch_size].index_copy_(2, cache_position, key_states)
            self.values[:src_batch_size].index_copy_(2, cache_position, value_states)
            return self.keys[:src_batch_size], self.values[:src_batch_size]
        else:
            # Normal: src_batch == dst_batch (during multi-sample decode)
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
            return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        kv_offset = 0
        kv_length = self.max_cache_len
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return (self.keys[0, 0].any(dim=-1)).sum() if self.is_initialized else 0

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.max_cache_len


class StaticCache(cache_utils.Cache):
    def __init__(
        self,
        config,
        max_cache_len,
        max_batch_size=1,
        offloading=False,
        offload_only_non_sliding=True,
        **kwargs,
    ):
        config = config.get_text_config(decoder=True)
        layers = []
        for _ in range(config.num_hidden_layers):
            layer = StaticLayer(max_cache_len=max_cache_len, max_batch_size=max_batch_size)
            layers.append(layer)

        super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)

    def expand_batch(self):
        """Copy batch 0's KV to all batch slots. Call before multi-sample decode."""
        for layer in self.layers:
            if layer.is_initialized:
                layer.expand_batch()


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

        # Store the un-roped keys and values in cache
        if past_key_values is not None:
            cache_kwargs = {'cache_position': cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

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
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        streaming_attention_mask=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        position_ids = position_ids[0]

        if inputs_embeds.shape[1] > 1:  # prefill, attention handles decode by default
            if streaming_attention_mask is not None:
                attention_mask = streaming_attention_mask
            else:
                attention_mask = create_causal_mask(
                    config=self.config,
                    input_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )

        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=past_key_values
        )

_PATCHED_CLASSES = {
    "Qwen3VLTextModel": Qwen3VLTextModel,
    "Qwen3VLTextAttention": Qwen3VLTextAttention,
}


def patch_for_training(model: nn.Module) -> None:
    """Patch Qwen3-VL modules with streaming-compatible implementations.

    Swaps __class__ on matching modules so that our custom forward methods
    (streaming RoPE, streaming attention mask, etc.) are used instead of
    the upstream transformers code.
    """
    for _path, module in model.named_modules():
        class_name = type(module).__name__
        if class_name in _PATCHED_CLASSES:
            if type(module) is not _PATCHED_CLASSES[class_name]:
                module.__class__ = _PATCHED_CLASSES[class_name]
