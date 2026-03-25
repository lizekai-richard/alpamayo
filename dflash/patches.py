from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3vl
import transformers.cache_utils as cache_utils
from transformers.cache_utils import Cache
from transformers.utils import is_torchdynamo_compiling
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, apply_rotary_pos_emb_vision
from transformers.models.qwen3_vl.modeling_qwen3_vl import rotate_half
from transformers.integrations.flex_attention import flex_attention_forward
from transformers.integrations.sdpa_attention import sdpa_attention_forward
import logging
import gc

logger = logging.getLogger(__name__)


# ==================== Custom StaticCache with batch expansion ====================

class StaticLayer(cache_utils.CacheLayerMixin):
    """Single layer of static KV cache with batch expansion support."""

    def __init__(self, max_cache_len, max_batch_size=1):
        super().__init__()
        self.max_cache_len = max_cache_len
        self._max_batch_size = max_batch_size

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        # Pre-allocate for max_batch_size (allows multi-sample decode)
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
        # Mark as static to prevent graph breaks during cache updates
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

    def update(self, key_states, value_states, cache_kwargs=None):
        """Update the key and value caches in-place.

        Handles batch size mismatch:
          - src_batch < dst_batch (prefill batch=1 into multi-batch cache): write batch 0 only
          - src_batch == dst_batch (decode with num_samples): normal update
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        cache_position = (
            cache_position if cache_position is not None else torch.arange(key_states.shape[-2], device=self.device)
        )

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
        """Return the length and offset of the cache for attention mask generation."""
        return self.max_cache_len, 0

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return (self.keys[0, 0].any(dim=-1)).sum() if self.is_initialized else 0

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape."""
        return self.max_cache_len


class StaticCache(cache_utils.Cache):
    """Static KV cache with support for batch expansion (num_traj_samples > 1).

    This cache pre-allocates memory for a fixed max_batch_size and max_cache_len.
    After prefill with batch=1, call expand_batch() to replicate the KV cache
    to all batch slots before multi-sample decoding.
    """

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

    def reset(self):
        """Reset the cache by zeroing all KV tensors."""
        for layer in self.layers:
            if layer.is_initialized:
                layer.keys.zero_()
                layer.values.zero_()


class QKVLinear(nn.Module):
    """Fused Query-Key-Value projection for multi-head attention.
    
    Instead of three separate linear layers (q_proj, k_proj, v_proj),
    this combines them into a single projection:
    
        [Q, K, V] = x @ W^T  where W = [W_q; W_k; W_v]
    """
    
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        """Initialize fused QKV projection.
        
        Args:
            hidden_size: Input embedding dimension.
            head_size: Dimension per attention head.
            total_num_heads: Number of query heads.
            total_num_kv_heads: Number of key/value heads (for GQA). 
                               Defaults to total_num_heads.
            bias: Whether to include bias terms.
        """
        super().__init__()
        total_num_kv_heads = total_num_kv_heads or total_num_heads

        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_heads = total_num_heads
        self.num_kv_heads = total_num_kv_heads

        # Output size: Q heads + K heads + V heads
        output_size = (self.num_heads + 2 * self.num_kv_heads) * self.head_size
        self.in_features = hidden_size
        self.out_features = output_size
        self.weight = nn.Parameter(torch.empty(output_size, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input to Q, K, V tensors.
        
        Args:
            x: Input tensor [B, L, hidden_size].
            
        Returns:
            q: Query tensor [B, num_heads, L, head_size].
            k: Key tensor [B, num_kv_heads, L, head_size].
            v: Value tensor [B, num_kv_heads, L, head_size].
        """
        if x.dim() != 3:
            raise ValueError(f"QKVLinear expects 3D input [B, L, D], got {x.shape}")

        bsz, seqlen, _ = x.shape
        
        # Single fused projection
        out = F.linear(x, self.weight, self.bias)

        # Reshape and split into Q, K, V
        total_heads = self.num_heads + 2 * self.num_kv_heads
        out = out.view(bsz, seqlen, total_heads, self.head_size)
        out = out.permute(0, 2, 1, 3).contiguous()  # [B, H_total, L, D]

        q = out[:, : self.num_heads]
        k = out[:, self.num_heads : self.num_heads + self.num_kv_heads]
        v = out[:, self.num_heads + self.num_kv_heads :]
        
        return q, k, v


class MergedColumnLinear(nn.Module):
    """Fused column-parallel linear layer for MLP.
    
    Combines multiple linear projections (e.g., gate and up in SwiGLU)
    into a single matrix multiplication:
    
        [gate, up] = x @ W^T  where W = [W_gate; W_up]
    
    The outputs are split along the last dimension.
    """
    
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        """Initialize merged linear layer.
        
        Args:
            input_size: Input dimension.
            output_sizes: List of output dimensions for each split.
            bias: Whether to include bias terms.
        """
        super().__init__()
        self.input_size = input_size
        self.output_sizes = list(output_sizes)

        output_size = sum(self.output_sizes)
        self.in_features = input_size
        self.out_features = output_size
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Project and split input.
        
        Args:
            x: Input tensor [..., input_size].
            
        Returns:
            Tuple of tensors, one for each output_size.
        """
        out = F.linear(x, self.weight, self.bias)
        return torch.split(out, self.output_sizes, dim=-1)
        

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

        if not hasattr(self, "_cached_pos_embeds") or self._cached_pos_embeds.shape[0] != hidden_states.shape[0]:
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
    """Patched Qwen3VL Text Attention with optional fused QKV projection."""

    def __init__(self, config, layer_idx: int, mode: str = "streaming", fuse_qkv: bool = False):
        super().__init__(config, layer_idx)

        self.mode = mode
        self.fuse_qkv = fuse_qkv
        # HF Qwen3VL doesn't set these as instance attributes
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        if fuse_qkv:
            # Replace separate q/k/v projections with fused QKVLinear
            self.qkv_proj = QKVLinear(
                hidden_size=config.hidden_size,
                head_size=self.head_dim,
                total_num_heads=config.num_attention_heads,
                total_num_kv_heads=config.num_key_value_heads,
                bias=config.attention_bias,
            )

            # Remove original separate projections
            delattr(self, "q_proj")
            delattr(self, "k_proj")
            delattr(self, "v_proj")

            # Register hook to fuse q/k/v weights when loading from state_dict
            self._register_load_state_dict_pre_hook(self._fuse_qkv_hook)

    def _fuse_qkv_hook(self, state_dict, prefix, *args, **kwargs):
        """Hook to fuse separate q/k/v weights into qkv_proj when loading state dict."""
        q_key = f"{prefix}q_proj.weight"
        k_key = f"{prefix}k_proj.weight"
        v_key = f"{prefix}v_proj.weight"

        if q_key in state_dict and k_key in state_dict and v_key in state_dict:
            q_weight = state_dict.pop(q_key).cpu()
            k_weight = state_dict.pop(k_key).cpu()
            v_weight = state_dict.pop(v_key).cpu()

            q_size, k_size = q_weight.shape[0], k_weight.shape[0]
            qkv_weight = torch.empty(q_size + k_size + v_weight.shape[0], q_weight.shape[1], dtype=q_weight.dtype, device='cpu')
            qkv_weight[:q_size].copy_(q_weight)
            qkv_weight[q_size:q_size + k_size].copy_(k_weight)
            qkv_weight[q_size + k_size:].copy_(v_weight)
            state_dict[f"{prefix}qkv_proj.weight"] = qkv_weight

            del q_weight, k_weight, v_weight

            # Handle bias if exists
            q_bias_key = f"{prefix}q_proj.bias"
            if q_bias_key in state_dict:
                q_bias = state_dict.pop(q_bias_key).cpu()
                k_bias = state_dict.pop(f"{prefix}k_proj.bias").cpu()
                v_bias = state_dict.pop(f"{prefix}v_proj.bias").cpu()
                qkv_bias = torch.empty(q_size + k_size + v_bias.shape[0], dtype=q_bias.dtype, device='cpu')
                qkv_bias[:q_size].copy_(q_bias)
                qkv_bias[q_size:q_size + k_size].copy_(k_bias)
                qkv_bias[q_size + k_size:].copy_(v_bias)
                state_dict[f"{prefix}qkv_proj.bias"] = qkv_bias

                del q_bias, k_bias, v_bias

            gc.collect()
            torch.cuda.empty_cache()

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

        if self.fuse_qkv:
            result = self.qkv_proj(hidden_states)
            if isinstance(result, tuple):
                # QKVLinear returns (q, k, v) each with shape [B, H, L, D]
                query_states, key_states, value_states = result
                query_states = self.q_norm(query_states.transpose(1, 2)).transpose(1, 2)
                key_states = self.k_norm(key_states.transpose(1, 2)).transpose(1, 2)
            else:
                # WQLinear returns single tensor [B, L, q+k+v], split manually
                q_size = self.num_heads * self.head_dim
                kv_size = self.num_kv_heads * self.head_dim
                q_out, k_out, v_out = result.split([q_size, kv_size, kv_size], dim=-1)
                query_states = self.q_norm(q_out.view(*input_shape, self.num_heads, self.head_dim)).transpose(1, 2)
                key_states = self.k_norm(k_out.view(*input_shape, self.num_kv_heads, self.head_dim)).transpose(1, 2)
                value_states = v_out.view(*input_shape, self.num_kv_heads, self.head_dim).transpose(1, 2)
        else:
            # Original separate q/k/v projections
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings

        if self.mode == "streaming":
            cos_q, sin_q = cos[:, cache_position, :], sin[:, cache_position, :]

        if self.mode == "non-streaming":
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Store the un-roped keys and values in cache
        if past_key_values is not None:
            cache_kwargs = {'cache_position': cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        if self.mode == "streaming":
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


class Qwen3VLTextMLP(qwen3vl.Qwen3VLTextMLP):
    def __init__(self, config, fuse_gate_up: bool = False):
        super().__init__(config)

        self.fuse_gate_up = fuse_gate_up

        if fuse_gate_up:
            self.gate_up_proj = MergedColumnLinear(
                input_size=self.hidden_size,
                output_sizes=[self.intermediate_size, self.intermediate_size],
                bias=False,
            )
            delattr(self, "gate_proj")
            delattr(self, "up_proj")

            # Register hook to fuse gate/up weights when loading from state_dict
            self._register_load_state_dict_pre_hook(self._fuse_gate_up_hook)

    def _fuse_gate_up_hook(self, state_dict, prefix, *args, **kwargs):
        """Hook to fuse separate gate/up weights into gate_up_proj when loading state dict."""
        gate_key = f"{prefix}gate_proj.weight"
        up_key = f"{prefix}up_proj.weight"

        if gate_key in state_dict and up_key in state_dict:
            gate_weight = state_dict.pop(gate_key).cpu()
            up_weight = state_dict.pop(up_key).cpu()

            gate_up_weight = torch.empty(gate_weight.shape[0] + up_weight.shape[0], gate_weight.shape[1], dtype=gate_weight.dtype, device='cpu')
            gate_up_weight[:gate_weight.shape[0]].copy_(gate_weight)
            gate_up_weight[gate_weight.shape[0]:].copy_(up_weight)
            state_dict[f"{prefix}gate_up_proj.weight"] = gate_up_weight

            del gate_weight, up_weight
            gc.collect()
            torch.cuda.empty_cache()

    def forward(self, x):
        if self.fuse_gate_up:
            result = self.gate_up_proj(x)
            if isinstance(result, tuple):
                gate, up = result
            else:
                # WQLinear returns single tensor, split manually
                gate, up = result.split([self.intermediate_size, self.intermediate_size], dim=-1)
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
        down_proj = self.down_proj(self.act_fn(gate) * up)
        return down_proj


class Qwen3VLTextModel(qwen3vl.Qwen3VLTextModel):
    def set_capture_layer_ids(self, layer_ids: list[int] | None):
        """Configure which layer hidden states to capture during forward.

        This is torch.compile-friendly because the set is a constant determined
        before compilation — the compiler traces ``if layer_idx in set`` as
        static control flow.

        Args:
            layer_ids: Layer indices whose hidden states should be returned in
                ``BaseModelOutputWithPast.hidden_states``.  Pass ``None`` to
                disable capturing.
        """
        if layer_ids:
            self._capture_layer_ids = layer_ids
            self._capture_layer_ids_set = set(layer_ids)  # O(1) lookup
        else:
            self._capture_layer_ids = None
            self._capture_layer_ids_set = set()

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
        streaming_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:
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

        if deepstack_visual_embeds is not None and not hasattr(self, "_cached_deepstack_indices"):
            self._cached_deepstack_indices = visual_pos_masks.flatten().nonzero(as_tuple=True)[0]

        hidden_states = inputs_embeds
        captured: list[torch.Tensor] = []
        _do_capture = hasattr(self, '_capture_layer_ids') and self._capture_layer_ids is not None
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            # Capture specific layers for DFlash draft context (compile-friendly)
            if _do_capture and layer_idx in self._capture_layer_ids_set:
                captured.append(hidden_states)
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                flat_hidden = hidden_states.view(-1, hidden_states.shape[-1])
                flat_hidden.index_add_(
                    0, self._cached_deepstack_indices, deepstack_visual_embeds[layer_idx]
                )

        # Store captured states on the model instance so they can be read
        # after a VLM-level forward (VLM wrapper doesn't propagate hidden_states).
        if captured:
            self._last_captured_hidden_states = tuple(captured)

        return BaseModelOutputWithPast(
            last_hidden_state=self.norm(hidden_states),
            past_key_values=past_key_values,
            hidden_states=tuple(captured) if captured else None,
        )


_PATCHED_CLASSES_STREAMING = {
    "Qwen3VLVisionModel": Qwen3VLVisionModel,
    "Qwen3VLTextModel": Qwen3VLTextModel,
    "Qwen3VLVisionPatchEmbed": Qwen3VLVisionPatchEmbed,
    "Qwen3VLVisionAttention": Qwen3VLVisionAttention,
    "Qwen3VLTextAttention": Qwen3VLTextAttention,  # Order matters. Replace the parent(Qwen3VLTextModel) first
}

_PATCHED_CLASSES_NON_STREAMING = {
    "Qwen3VLVisionModel": Qwen3VLVisionModel,
    "Qwen3VLVisionPatchEmbed": Qwen3VLVisionPatchEmbed,
    "Qwen3VLVisionAttention": Qwen3VLVisionAttention,
    "Qwen3VLTextModel": Qwen3VLTextModel,
}


def _get_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    param = next(module.parameters(), None)
    if param is None:
        return torch.device("cpu"), torch.float32
    return param.device, param.dtype


def _fuse_wqlinears(*linears: nn.Module) -> nn.Module:
    """Fuse multiple WQLinear modules into one by concatenating along the output dimension.

    Column-parallel fusion: quantization groups run along the input (reduction)
    dimension, so concatenating along output preserves quantization correctness.
    Uses ``type(linears[0])`` to construct the fused module, avoiding a hard
    import of the AWQ WQLinear class.
    """
    first = linears[0]
    cls = type(first)
    total_out = sum(l.out_features for l in linears)

    fused = cls(
        w_bit=first.w_bit,
        group_size=first.group_size,
        in_features=first.in_features,
        out_features=total_out,
        bias=first.bias is not None,
        dev=first.qweight.device,
        dtype=first.scales.dtype,
    )

    fused.qweight = torch.cat([l.qweight for l in linears], dim=0)
    fused.scales = torch.cat([l.scales for l in linears], dim=1).contiguous()
    fused.scaled_zeros = torch.cat([l.scaled_zeros for l in linears], dim=1).contiguous()

    if first.bias is not None:
        fused.bias = torch.cat([l.bias for l in linears], dim=0)

    fused.split_k_iters = first.split_k_iters
    return fused


def _copy_children(dst: nn.Module, src: nn.Module) -> None:
    """Copy all sub-modules, parameters, and buffers from *src* to *dst*."""
    for child_name, child in src.named_children():
        setattr(dst, child_name, child)
    for pname, param in src.named_parameters(recurse=False):
        setattr(dst, pname, param)
    for bname, buf in src.named_buffers(recurse=False):
        setattr(dst, bname, buf)


def _replace_module(
    model: nn.Module,
    module_path: str,
    old_module: nn.Module,
    new_class: type,
    mode: str = "streaming",
    fuse_qkv: bool = False,
    fuse_gate_up: bool = False,
) -> None:
    *parent_parts, name = module_path.split(".")
    parent = model
    for part in parent_parts:
        parent = getattr(parent, part)

    config = getattr(old_module, "config", None) or getattr(parent, "config", None)
    device, dtype = _get_device_dtype(old_module)

    # Check if old module contains quantized (non-nn.Linear) weight layers
    _has_quantized = any(
        hasattr(m, "qweight") for m in old_module.modules()
    )

    if new_class is Qwen3VLTextAttention:
        layer_idx = getattr(old_module, "layer_idx", None)

        if _has_quantized:
            # Create without fusion, copy quantized children, then fuse WQLinears
            new_module = new_class(config, layer_idx, mode=mode, fuse_qkv=False)
            _copy_children(new_module, old_module)

            if fuse_qkv:
                new_module.qkv_proj = _fuse_wqlinears(
                    new_module.q_proj, new_module.k_proj, new_module.v_proj,
                )
                delattr(new_module, "q_proj")
                delattr(new_module, "k_proj")
                delattr(new_module, "v_proj")
                new_module.fuse_qkv = True
        else:
            new_module = new_class(config, layer_idx, mode=mode, fuse_qkv=fuse_qkv)
            old_module = old_module.to("cpu")
            new_module.load_state_dict(old_module.state_dict(), assign=True)
    elif new_class is Qwen3VLTextMLP:
        if _has_quantized:
            new_module = new_class(config, fuse_gate_up=False)
            _copy_children(new_module, old_module)

            if fuse_gate_up:
                new_module.gate_up_proj = _fuse_wqlinears(
                    new_module.gate_proj, new_module.up_proj,
                )
                delattr(new_module, "gate_proj")
                delattr(new_module, "up_proj")
                new_module.fuse_gate_up = True
        else:
            new_module = new_class(config, fuse_gate_up=fuse_gate_up)
            old_module = old_module.to("cpu")
            new_module.load_state_dict(old_module.state_dict(), assign=True)
    else:
        new_module = new_class(config)
        new_module.load_state_dict(old_module.state_dict(), assign=True)

    if device.type != "meta":
        new_module = new_module.to(device=device, dtype=dtype)

    setattr(parent, name, new_module)


def patch_for_torch_compile(
    model: nn.Module,
    mode: str = "streaming",
    fuse_qkv: bool = False,
    fuse_gate_up: bool = False,
) -> None:
    """Patch Qwen3-VL modules for torch.compile compatibility.

    Args:
        model: The model to patch.
        mode: "streaming" or "non-streaming".
        fuse_qkv: If True, fuse q/k/v projections into a single QKVLinear.
        fuse_gate_up: If True, fuse gate/up projections into a single MergedColumnLinear.
    """
    assert mode in ["streaming", "non-streaming"], "Invalid mode"
    
    if mode == "streaming":
        patched_classes = _PATCHED_CLASSES_STREAMING.copy()  # Copy to avoid modifying global
        if fuse_gate_up:
            patched_classes["Qwen3VLTextMLP"] = Qwen3VLTextMLP
    else:
        patched_classes = _PATCHED_CLASSES_NON_STREAMING.copy()
        if fuse_qkv:
            patched_classes["Qwen3VLTextAttention"] = Qwen3VLTextAttention
        if fuse_gate_up:
            patched_classes["Qwen3VLTextMLP"] = Qwen3VLTextMLP

    # Collect modules to replace first (avoid modifying during iteration)
    modules_to_replace = []
    for module_path, module in model.named_modules():
        class_name = type(module).__name__
        if class_name in patched_classes:
            # Skip if already patched
            if type(module) is patched_classes[class_name]:
                continue
            modules_to_replace.append((module_path, module, patched_classes[class_name]))

    for module_path, module, patched_class in modules_to_replace:
        _replace_module(model, module_path, module, patched_class, mode=mode, fuse_qkv=fuse_qkv, fuse_gate_up=fuse_gate_up)
