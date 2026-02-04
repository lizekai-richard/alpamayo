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
import logging

logger = logging.getLogger(__name__)


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
    """Patched Qwen3VL Text Attention with fused QKV projection."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        # Replace separate q/k/v projections with fused QKVLinear
        self.qkv_proj = QKVLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            bias=config.attention_bias,
        )

        # Remove original separate projections
        del self.q_proj
        del self.k_proj
        del self.v_proj

        # Register hook to fuse q/k/v weights when loading from checkpoint
        self._register_load_state_dict_pre_hook(self._fuse_qkv_hook)

    def _fuse_qkv_hook(self, state_dict, prefix, *args, **kwargs):
        """Hook to fuse separate q/k/v weights into qkv_proj when loading state dict."""
        q_weight_key = f"{prefix}q_proj.weight"
        k_weight_key = f"{prefix}k_proj.weight"
        v_weight_key = f"{prefix}v_proj.weight"
        qkv_weight_key = f"{prefix}qkv_proj.weight"

        # Check if we need to fuse (original separate weights exist)
        if q_weight_key in state_dict and k_weight_key in state_dict and v_weight_key in state_dict:
            q_weight = state_dict.pop(q_weight_key)
            k_weight = state_dict.pop(k_weight_key)
            v_weight = state_dict.pop(v_weight_key)

            # Fuse weights: [q_dim, hidden] + [kv_dim, hidden] + [kv_dim, hidden]
            state_dict[qkv_weight_key] = torch.cat([q_weight, k_weight, v_weight], dim=0)

            # Handle bias if exists
            q_bias_key = f"{prefix}q_proj.bias"
            if q_bias_key in state_dict:
                state_dict[f"{prefix}qkv_proj.bias"] = torch.cat([
                    state_dict.pop(q_bias_key),
                    state_dict.pop(f"{prefix}k_proj.bias"),
                    state_dict.pop(f"{prefix}v_proj.bias"),
                ], dim=0)

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

        # QKVLinear returns (q, k, v) each with shape [B, H, L, D]
        query_states, key_states, value_states = self.qkv_proj(hidden_states)

        # Apply norms: transpose to [B, L, H, D] for norm, then back to [B, H, L, D]
        query_states = self.q_norm(query_states.transpose(1, 2)).transpose(1, 2)
        key_states = self.k_norm(key_states.transpose(1, 2)).transpose(1, 2)

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


def patch_for_torch_compile(model: nn.Module, mode: ["streaming", "non-streaming"] = "streaming") -> None:
    """Patch Qwen3-VL modules for torch.compile compatibility."""
    if mode == "streaming":
        _PATCHED_CLASSES = _PATCHED_CLASSES_STREAMING
    else:
        _PATCHED_CLASSES = _PATCHED_CLASSES_NON_STREAMING
    for class_name, patched_class in _PATCHED_CLASSES.items():
        for module_path, module in model.named_modules():
            if type(module).__name__ == class_name:
                _replace_module(model, module_path, module, patched_class)
