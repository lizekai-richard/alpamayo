#!/usr/bin/env python3
"""DFlash integration for accelerating Alpamayo Chain-of-Causation reasoning generation.

This module provides utilities to use DFlash block diffusion speculative decoding
to accelerate the autoregressive text generation in Alpamayo's VLM backbone.

Usage:
    from alpamayo_r1.dflash_integration import DFlashAlpamayoAccelerator, load_dflash_draft_model

    # Load models
    alpamayo = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", ...)
    draft_model = load_dflash_draft_model("z-lab/Qwen3-8B-DFlash-b16")

    # Create accelerator
    accelerator = DFlashAlpamayoAccelerator(draft_model, alpamayo.vlm, alpamayo.tokenizer)
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import DynamicCache, StaticCache
from transformers.generation.logits_process import LogitsProcessor

logger = logging.getLogger(__name__)


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """Build target layer IDs for context feature extraction.

    Samples layer indices evenly distributed across the target model's layers.
    This allows the draft model to condition on multi-scale representations.

    Args:
        num_target_layers: Number of layers in the target model.
        num_draft_layers: Number of layers in the draft model.

    Returns:
        List of layer indices to extract features from.
    """
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids


def extract_context_feature(
    hidden_states: tuple[torch.Tensor, ...],
    layer_ids: list[int],
) -> torch.Tensor:
    """Extract and concatenate hidden states from specified layers.

    Args:
        hidden_states: Tuple of hidden states from all layers (including embeddings).
        layer_ids: List of layer indices to extract from.

    Returns:
        Concatenated hidden states tensor of shape (batch, seq_len, hidden_size * num_layers).
    """
    offset = 1  # Account for embedding layer in hidden_states
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    logits_processor: LogitsProcessor | None = None,
) -> torch.Tensor:
    """Sample tokens from logits with optional temperature and logits processing.

    Args:
        logits: Logits tensor of shape (batch, seq_len, vocab_size).
        temperature: Sampling temperature. 0 means greedy decoding.
        logits_processor: Optional processor to apply before sampling.

    Returns:
        Sampled token IDs of shape (batch, seq_len).
    """
    bsz, seq_len, vocab_size = logits.shape

    # Apply logits processor if provided (e.g., to mask trajectory tokens)
    # Vectorized: apply mask directly to 3D tensor instead of looping per position
    if logits_processor is not None:
        if isinstance(logits_processor, _TrajectoryTokenMask):
            # Optimized path: apply mask to all positions at once
            offset = logits_processor.traj_token_offset
            size = logits_processor.traj_vocab_size
            logits[:, :, offset : offset + size] = float("-inf")
        else:
            # Fallback: loop for custom processors (preserves compatibility)
            processed_logits = []
            for pos in range(seq_len):
                pos_logits = logits_processor(None, logits[:, pos, :])
                processed_logits.append(pos_logits)
            logits = torch.stack(processed_logits, dim=1)

    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


class _TrajectoryTokenMask(LogitsProcessor):
    """Masks out logits for discrete trajectory tokens during CoC generation.

    This prevents the model from generating action/trajectory tokens prematurely
    during Chain-of-Causation text generation.
    """

    def __init__(self, traj_token_offset: int, traj_vocab_size: int):
        super().__init__()
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size

    def __call__(
        self, input_ids: torch.LongTensor | None, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        scores[:, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size] = float(
            "-inf"
        )
        return scores


@dataclass
class DFlashConfig:
    """Configuration for DFlash speculative decoding."""

    block_size: int = 8
    mask_token: str = "<|MASK|>"
    temperature: float = 0.0
    # Layer IDs to extract from target model (auto-computed if None)
    target_layer_ids: list[int] | None = None
    # Trajectory token masking (required for Alpamayo CoC generation)
    traj_token_offset: int | None = None
    traj_vocab_size: int | None = None
    # If set, use this existing token ID as the mask token instead of adding a new one.
    # This is critical for pretrained DFlash checkpoints that were trained with a specific
    # mask token (e.g., <|extra_0|> in Qwen vocabulary). Using a mismatched mask token
    # will cause severe degradation in draft quality.
    mask_token_id_override: int | None = None
    # If True and a new mask token must be added, initialize its embedding to the mean
    # of all existing embeddings rather than random. This provides more stable behavior
    # when the exact mask token from DFlash training is unknown.
    init_mask_to_mean: bool = True
    # torch.compile settings — set to None to disable compilation (fallback to
    # DynamicCache + hooks path)
    torch_compile: str | None = None
    max_cache_len: int = 3120  # StaticCache size for compiled verify
    # Use StaticCache even without torch.compile (for isolating cache effects)
    use_static_cache: bool = False
    # Use set_capture_layer_ids instead of hooks (requires patched model, no compile)
    use_capture_layer_ids: bool = False


@dataclass
class GenerationStats:
    """Statistics from speculative generation."""

    total_tokens: int = 0
    total_iterations: int = 0
    acceptance_lengths: list[int] = field(default_factory=list)
    draft_matches: list[int] = field(default_factory=list)  # actual draft tokens accepted (for accurate rate)
    tokens_verified: list[int] = field(default_factory=list)  # tokens verified before stop (for match rate)
    hit_stop: list[bool] = field(default_factory=list)  # whether each step hit stop token
    drafting_iterations: int = 0  # iterations where draft model was actually used
    prefill_time_ms: float = 0.0
    vit_time_ms: float = 0.0        # Visual encoder time within prefill
    llm_prefill_time_ms: float = 0.0  # LLM-only prefill time (prefill - vit)
    decode_time_ms: float = 0.0
    block_size: int = 8  # Must match accelerator's block_size
    # Detailed timing breakdown
    draft_time_ms: float = 0.0      # Total time in draft model
    verify_time_ms: float = 0.0     # Total time in target model verification
    cache_time_ms: float = 0.0      # Total time in KV cache operations
    sample_time_ms: float = 0.0     # Total time in token sampling
    lm_head_time_ms: float = 0.0    # Total time in lm_head projection

    @property
    def mean_acceptance_length(self) -> float:
        """Mean number of tokens accepted per iteration (0-8, includes posterior)."""
        if not self.acceptance_lengths:
            return 0.0
        return sum(self.acceptance_lengths) / len(self.acceptance_lengths)

    @property
    def mean_match_length(self) -> float:
        """Mean number of draft tokens matched per iteration (0-7, draft only)."""
        if not self.draft_matches:
            return 0.0
        return sum(self.draft_matches) / len(self.draft_matches)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of possible tokens that were accepted (measures efficiency).

        This metric penalizes early termination (e.g., hitting <|cot_end|> mid-block)
        because it counts all possible tokens (8 per iteration) in the denominator.
        Numerator includes draft matches + posterior (always accepted from target).
        """
        if self.drafting_iterations == 0:
            return 0.0
        # Total possible tokens = 8 per iteration (7 draft + 1 posterior)
        total_possible = self.drafting_iterations * self.block_size
        total_accepted = sum(self.acceptance_lengths)
        return total_accepted / total_possible if total_possible > 0 else 0.0

    @property
    def match_rate(self) -> float:
        """Fraction of verified tokens that matched (measures draft quality).

        Includes all steps, but for the final step that hits <|cot_end|>, only counts
        tokens BEFORE the stop token. This measures draft model prediction accuracy
        on actual content tokens (excluding <|cot_end|>).
        """
        total_verified = sum(self.tokens_verified)
        total_matched = sum(self.draft_matches)
        return total_matched / total_verified if total_verified > 0 else 0.0


class DFlashAlpamayoAccelerator:
    """Accelerator for Alpamayo CoC generation using DFlash speculative decoding.

    This class wraps a DFlash draft model and provides methods to accelerate
    the Chain-of-Causation text generation in Alpamayo's inference pipeline.

    The accelerator performs:
    1. Multimodal prefill with visual inputs (handled by target VLM)
    2. Speculative decoding loop for text generation (DFlash acceleration)

    Example:
        accelerator = DFlashAlpamayoAccelerator(draft_model, vlm, tokenizer)
        output_ids, stats, kv_cache, rope_deltas = accelerator.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=256,
        )
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_vlm: nn.Module,
        tokenizer: Any,
        config: DFlashConfig | None = None,
    ):
        """Initialize the accelerator.

        Args:
            draft_model: The DFlash draft model (DFlashDraftModel).
            target_vlm: The target VLM (Qwen3VLForConditionalGeneration).
            tokenizer: The tokenizer with mask token support.
            config: Configuration for speculative decoding.
        """
        self.draft_model = draft_model
        self.target_vlm = target_vlm
        self.tokenizer = tokenizer
        self.config = config or DFlashConfig()

        # Get block size from draft model or config
        self.block_size = getattr(draft_model, 'block_size', self.config.block_size)

        # Get or compute target layer IDs
        if self.config.target_layer_ids is not None:
            self.target_layer_ids = self.config.target_layer_ids
        elif hasattr(draft_model, 'target_layer_ids'):
            self.target_layer_ids = draft_model.target_layer_ids
        else:
            # Compute based on model configs
            num_target_layers = target_vlm.config.text_config.num_hidden_layers
            num_draft_layers = draft_model.config.num_hidden_layers
            self.target_layer_ids = build_target_layer_ids(num_target_layers, num_draft_layers)

        # Setup mask token
        self.mask_token_id = self._setup_mask_token()

        # Cache for embed_tokens and lm_head references
        # Qwen3VL structure: vlm.model.language_model.embed_tokens
        self._embed_tokens = target_vlm.model.language_model.embed_tokens
        self._lm_head = target_vlm.lm_head

        # Setup logits processor for trajectory token masking (prevents premature action tokens)
        self._logits_processor = None
        if self.config.traj_token_offset is not None and self.config.traj_vocab_size is not None:
            self._logits_processor = _TrajectoryTokenMask(
                traj_token_offset=self.config.traj_token_offset,
                traj_vocab_size=self.config.traj_vocab_size,
            )
            logger.info(
                f"[DFlash] Trajectory token masking enabled: "
                f"offset={self.config.traj_token_offset}, vocab_size={self.config.traj_vocab_size}"
            )

        # torch.compile settings
        self._torch_compile = self.config.torch_compile
        self._language_model = target_vlm.model.language_model

        self._use_capture_layer_ids = self.config.use_capture_layer_ids

        if self._torch_compile is not None:
            # Compiled path: use model-level hidden state capture (hooks break fullgraph)
            self._language_model.set_capture_layer_ids(self.target_layer_ids)
            self._captured_hidden_states: dict[int, torch.Tensor] = {}
            self._hooks: list = []
            self._static_cache: StaticCache | None = None  # created lazily in generate()
            logger.info(
                f"[DFlash] torch.compile enabled (mode={self._torch_compile}): "
                "using model-level hidden state capture instead of hooks"
            )
        elif self._use_capture_layer_ids:
            # Non-compiled but using model-level capture (for isolating capture method)
            self._language_model.set_capture_layer_ids(self.target_layer_ids)
            self._captured_hidden_states: dict[int, torch.Tensor] = {}
            self._hooks: list = []
            if self.config.use_static_cache:
                self._static_cache: StaticCache | None = None
            logger.info(
                "[DFlash] Using model-level set_capture_layer_ids (no compile, no hooks)"
            )
        else:
            # Non-compiled path: use hooks for efficient hidden state capture
            self._captured_hidden_states: dict[int, torch.Tensor] = {}
            self._hooks: list = []
            self._setup_hidden_state_hooks()
            if self.config.use_static_cache:
                self._static_cache: StaticCache | None = None

        # Detect model inference dtype from embedding weights
        self._inference_dtype = target_vlm.get_input_embeddings().weight.dtype
        self._inference_device = next(target_vlm.parameters()).device
        logger.info(f"[DFlash] Detected model inference dtype: {self._inference_dtype}")

        logger.info(
            f"DFlash accelerator initialized: block_size={self.block_size}, "
            f"target_layers={self.target_layer_ids}, mask_token_id={self.mask_token_id}, "
            f"torch_compile={self._torch_compile}"
        )

    def _setup_mask_token(self) -> int:
        """Setup the tokenizer with mask token for DFlash.

        For pretrained DFlash checkpoints, it's critical to use the same mask token
        that was used during training. Using a mismatched mask token will cause
        the draft model to receive out-of-distribution inputs, severely degrading quality.

        Returns:
            The mask token ID.
        """
        # Option 1: Use override if specified (preferred for pretrained DFlash)
        if self.config.mask_token_id_override is not None:
            mask_id = self.config.mask_token_id_override
            vocab_size = self.target_vlm.get_input_embeddings().weight.shape[0]
            # If mask_id is at vocab boundary, resize embeddings (training added this token)
            if mask_id == vocab_size:
                logger.info(f"[DFlash] Resizing embeddings from {vocab_size} to {vocab_size + 1} for mask token")
                self.target_vlm.resize_token_embeddings(vocab_size + 1)
            elif mask_id > vocab_size:
                raise ValueError(
                    f"mask_token_id_override={mask_id} is out of vocabulary range ({vocab_size}). "
                    "Ensure the ID matches the token used during DFlash training."
                )
            # CRITICAL: Always reset mask embedding to mean to match training distribution
            if self.config.init_mask_to_mean:
                logger.info(f"[DFlash] Resetting mask token {mask_id} embedding to vocab mean (matching training)")
                self._init_mask_embedding_to_mean(mask_id)
            logger.info(f"[DFlash] Using override mask_token_id={mask_id}")
            return mask_id

        # Option 2: Check if tokenizer already has a mask token
        if self.tokenizer.mask_token_id is not None:
            logger.info(
                f"[DFlash] Using existing mask token: {self.tokenizer.mask_token} "
                f"(ID: {self.tokenizer.mask_token_id})"
            )
            # CRITICAL: Reset existing mask embedding to mean to match training distribution
            if self.config.init_mask_to_mean:
                logger.info(f"[DFlash] Resetting mask token {self.tokenizer.mask_token_id} embedding to vocab mean (matching training)")
                self._init_mask_embedding_to_mean(self.tokenizer.mask_token_id)
            return self.tokenizer.mask_token_id

        # Option 3: Try to find a reserved token that DFlash might have used
        # Many DFlash implementations use <|extra_0|> or similar reserved tokens
        for reserved_token in ["<|extra_0|>", "<|placeholder|>", "[MASK]"]:
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(reserved_token)
                if token_id != self.tokenizer.unk_token_id:
                    logger.info(
                        f"[DFlash] Found potential mask token '{reserved_token}' (ID: {token_id}). "
                        "If draft quality is poor, verify this matches the DFlash training config."
                    )
                    self.tokenizer.mask_token = reserved_token
                    # CRITICAL: Reset reserved token embedding to mean to match training distribution
                    if self.config.init_mask_to_mean:
                        logger.info(f"[DFlash] Resetting reserved token {token_id} embedding to vocab mean (matching training)")
                        self._init_mask_embedding_to_mean(token_id)
                    return token_id
            except Exception:
                continue

        # Option 4: Add new token as last resort
        logger.warning(
            "[DFlash] No existing mask token found. Adding new token. "
            "WARNING: If using a pretrained DFlash checkpoint, this may cause poor draft quality. "
            "Set mask_token_id_override in DFlashConfig to the correct token ID."
        )
        num_added = self.tokenizer.add_special_tokens({"mask_token": self.config.mask_token})
        if num_added > 0:
            # Resize embeddings
            self.target_vlm.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"[DFlash] Added mask token '{self.config.mask_token}' to tokenizer")

            # Initialize new embedding to mean of existing embeddings for stability
            if self.config.init_mask_to_mean:
                self._init_mask_embedding_to_mean()

        return self.tokenizer.mask_token_id

    def _init_mask_embedding_to_mean(self, mask_id: int | None = None) -> None:
        """Initialize the mask token embedding to the mean of all existing embeddings.

        This provides more stable behavior than random initialization when the exact
        mask token from DFlash training is unknown.
        """
        input_embeddings = self.target_vlm.get_input_embeddings()
        if mask_id is None:
            mask_id = self.tokenizer.mask_token_id

        with torch.no_grad():
            # Compute mean of all embeddings (excluding the new mask token)
            all_embeddings = input_embeddings.weight[:-1]  # Exclude last (new) token
            mean_embedding = all_embeddings.mean(dim=0)
            input_embeddings.weight[mask_id] = mean_embedding

        logger.info(
            f"[DFlash] Initialized mask token embedding (ID: {mask_id}) to mean of vocabulary. "
            "For best results, use the exact mask token from DFlash training."
        )

    def _setup_hidden_state_hooks(self) -> None:
        """Register forward hooks to capture hidden states from specific layers only.

        This avoids storing all layer hidden states (output_hidden_states=True),
        which causes significant memory allocation overhead.
        """
        def make_hook(layer_id: int):
            def hook_fn(module, input, output):
                # output is typically (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    self._captured_hidden_states[layer_id] = output[0]
                else:
                    self._captured_hidden_states[layer_id] = output
            return hook_fn

        # Register hooks on the specific layers we need
        # Qwen3VL structure: target_vlm.model.language_model.layers[i]
        language_model = self.target_vlm.model.language_model
        for layer_id in self.target_layer_ids:
            layer = language_model.layers[layer_id]
            hook = layer.register_forward_hook(make_hook(layer_id))
            self._hooks.append(hook)

        logger.info(f"[DFlash] Registered hooks on {len(self.target_layer_ids)} layers for efficient hidden state capture")

    def _extract_hooked_hidden_states(self) -> torch.Tensor:
        """Extract and concatenate hidden states captured by hooks.

        Returns:
            Concatenated hidden states tensor of shape (batch, seq_len, hidden_size * num_layers).
        """
        selected_states = [self._captured_hidden_states[layer_id] for layer_id in self.target_layer_ids]
        return torch.cat(selected_states, dim=-1)

    def _clear_captured_states(self) -> None:
        """Clear captured hidden states before next forward pass."""
        self._captured_hidden_states.clear()

    @staticmethod
    def _set_fp8_enabled(enabled: bool) -> None:
        """Toggle FP8 dispatch globally across all WQLinear layers.

        Used to enable FP8 only during initial prefill (for speed),
        then disable for decode/verify (for consistent speculative decoding).
        """
        try:
            import sys
            awq_root = str(Path(__file__).parent.parent / "awq")
            if awq_root not in sys.path:
                sys.path.insert(0, awq_root)
            from awq.quantize.qmodule import WQLinear
            WQLinear._fp8_globally_enabled = enabled
        except ImportError:
            pass

    # ==================== Compiled Functions (torch.compile path) ====================

    def _draft(
        self,
        block_output_ids: torch.Tensor,   # (1, block_size)
        target_context: torch.Tensor,     # (1, 1, context_dim)
        position_ids: torch.Tensor,       # (1, 1+block_size)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compiled draft step: embed -> draft_model -> lm_head.

        Returns:
            (draft_hidden, draft_logits) where draft_hidden is full hidden
            output and draft_logits is lm_head applied to positions 1: onward.
        """
        if not hasattr(self, "_draft_fn"):
            # 1. Static buffers
            self._draft_block_ids = torch.empty_like(block_output_ids)
            self._draft_context = torch.empty_like(target_context)
            self._draft_pos_ids = torch.empty_like(position_ids)

            # 2. Closure
            def draft_fn():
                noise = self._embed_tokens(self._draft_block_ids)
                hidden = self.draft_model(
                    target_hidden=self._draft_context,
                    noise_embedding=noise,
                    position_ids=self._draft_pos_ids,
                    past_key_values=None,
                    use_cache=False,
                    is_causal=False,
                )
                logits = self._lm_head(hidden[:, 1:, :])
                return hidden, logits

            self._draft_fn = draft_fn

        # 3. Copy to static buffers
        self._draft_block_ids.copy_(block_output_ids)
        self._draft_context.copy_(target_context)
        self._draft_pos_ids.copy_(position_ids)

        # 4. Warmup + compile (first call only)
        if not hasattr(self, "_compiled_draft_fn"):
            self._draft_fn()  # Warmup
            self._compiled_draft_fn = torch.compile(
                self._draft_fn, mode=self._torch_compile, fullgraph=True
            )

        # 5. Execute
        torch.compiler.cudagraph_mark_step_begin()
        return self._compiled_draft_fn()

    def _verify(
        self,
        input_ids: torch.Tensor,       # (1, block_size)
        position_ids: torch.Tensor,    # (3, 1, block_size) or (1, block_size)
        cache_position: torch.Tensor,  # (block_size,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compiled verify step: language_model -> lm_head + hidden state extraction.

        Returns:
            (logits, context) where context is the concatenated captured hidden
            states from the target layers.
        """
        if not hasattr(self, "_verify_fn"):
            self._verify_input_ids = torch.empty_like(input_ids)
            self._verify_position_ids = torch.empty_like(position_ids)
            self._verify_cache_position = torch.empty_like(cache_position)

            def verify_fn():
                output = self._language_model(
                    input_ids=self._verify_input_ids,
                    position_ids=self._verify_position_ids,
                    past_key_values=self._static_cache,
                    cache_position=self._verify_cache_position,
                    use_cache=True,
                )
                logits = self._lm_head(output.last_hidden_state)
                # hidden_states is tuple of captured layers (set via set_capture_layer_ids)
                context = torch.cat(output.hidden_states, dim=-1)
                return logits, context

            self._verify_fn = verify_fn

        self._verify_input_ids.copy_(input_ids)
        self._verify_position_ids.copy_(position_ids)
        self._verify_cache_position.copy_(cache_position)

        if not hasattr(self, "_compiled_verify_fn"):
            self._verify_fn()  # Warmup
            self._compiled_verify_fn = torch.compile(
                self._verify_fn, mode=self._torch_compile, fullgraph=True
            )

        torch.compiler.cudagraph_mark_step_begin()
        return self._compiled_verify_fn()

    # ==================== Static Cache Helpers ====================

    @staticmethod
    def _patch_static_cache_dtype(cache: StaticCache) -> None:
        """Patch StaticLayer.update to cast key/value states to match cache dtype.

        HF StaticLayer.update uses index_copy_ which requires matching dtypes.
        When the model forward produces float32 KV but the cache is bfloat16
        (or vice versa), this causes a dtype mismatch error.
        """
        from transformers.cache_utils import StaticLayer

        _orig_update = StaticLayer.update

        def _update_with_cast(self, key_states, value_states, cache_kwargs=None):
            if self.is_initialized and self.keys is not None:
                key_states = key_states.to(self.keys.dtype)
                value_states = value_states.to(self.values.dtype)
            return _orig_update(self, key_states, value_states, cache_kwargs)

        StaticLayer.update = _update_with_cast

    def _transfer_dynamic_to_static_cache(
        self,
        dynamic_cache: DynamicCache,
        seq_len: int,
    ) -> StaticCache:
        """Transfer DynamicCache contents to a StaticCache after prefill.

        The StaticCache is created once (lazily) and reused across generate()
        calls.  Only the first ``seq_len`` positions are copied.
        """
        if self._static_cache is None:
            text_config = self.target_vlm.config.text_config
            self._static_cache = StaticCache(
                config=text_config,
                max_cache_len=self.config.max_cache_len,
                dtype=self._inference_dtype,
            )
            self._patch_static_cache_dtype(self._static_cache)

        self._static_cache.reset()

        for layer_idx in range(len(dynamic_cache)):
            k, v = dynamic_cache[layer_idx]
            k = k[:, :, :seq_len, :].to(self._inference_dtype)
            v = v[:, :, :seq_len, :].to(self._inference_dtype)
            cache_position = torch.arange(seq_len, device=k.device)
            self._static_cache.update(k, v, layer_idx, {"cache_position": cache_position})

        return self._static_cache

    def _crop_static_cache(self, valid_length: int) -> None:
        """Zero out cache positions beyond valid_length (compile-friendly crop)."""
        for layer in self._static_cache.layers:
            if layer.is_initialized:
                layer.keys[:, :, valid_length:, :].zero_()
                layer.values[:, :, valid_length:, :].zero_()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        max_new_tokens: int = 256,
        stop_token_ids: list[int] | None = None,
        temperature: float | None = None,
        position_ids: torch.Tensor | None = None,
        rope_deltas: torch.Tensor | None = None,
        enable_detailed_timing: bool = False,
        log_file: str | None = None,
        **prefill_kwargs,
    ) -> tuple[torch.Tensor, GenerationStats]:
        """Generate text using speculative decoding with DFlash.

        This method performs:
        1. Multimodal prefill with visual inputs
        2. Speculative decoding loop for text generation

        Args:
            input_ids: Input token IDs including prompt, shape (1, seq_len).
            pixel_values: Visual inputs (images), optional.
            image_grid_thw: Image grid dimensions, optional.
            max_new_tokens: Maximum new tokens to generate.
            stop_token_ids: Token IDs that stop generation.
            temperature: Sampling temperature (overrides config).
            position_ids: Optional 3D position IDs for MROPE (shape: 3, batch, seq_len).
                If not provided, will be computed during prefill.
            rope_deltas: Optional rope deltas from previous processing.
            enable_detailed_timing: If True, measure detailed timing breakdown using
                CUDA events. When False (default), skip timing for maximum speed.
            log_file: Optional path to write JSON lines log of generation progress.
                Each line contains timestamp and step info for replay/streaming display.
            **prefill_kwargs: Additional kwargs for prefill.

        Returns:
            Tuple of (output_ids, generation_stats, past_key_values, rope_deltas).
            - output_ids: Generated token IDs
            - generation_stats: GenerationStats with timing info
            - past_key_values: KV cache from target VLM (for diffusion)
            - rope_deltas: Position encoding offsets (for diffusion)
        """
        self.draft_model.eval()
        self.target_vlm.eval()

        device = input_ids.device
        bsz = input_ids.shape[0]
        temperature = temperature if temperature is not None else self.config.temperature
        block_size = self.block_size
        stats = GenerationStats(block_size=block_size)

        # Speculative decoding with token-level acceptance requires batch_size=1
        # (different sequences would have different acceptance lengths)
        if bsz != 1:
            raise ValueError(
                f"DFlash speculative decoding requires batch_size=1, got {bsz}. "
                "For batched inference, process sequences sequentially or use standard generation."
            )

        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        # Initialize output buffer with mask tokens
        output_ids = torch.full(
            (bsz, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        # Initialize KV caches
        # Use DynamicCache for prefill (variable-length input, causal mask
        # requires matching sizes).
        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        # ====== PREFILL STAGE ======
        import time
        import json

        # Initialize log file if requested
        log_handle = None
        generation_start_time = time.perf_counter()
        if log_file is not None:
            log_handle = open(log_file, 'w')
            log_handle.write(json.dumps({
                "timestamp_ms": 0.0,
                "phase": "prefill_start",
                "num_input_tokens": num_input_tokens,
                "max_new_tokens": max_new_tokens,
                "mask_token_id": self.mask_token_id,
            }) + '\n')
            log_handle.flush()

        prefill_start = time.perf_counter()

        # Enable FP8 for prefill (large M), will be disabled after for decode consistency
        self._set_fp8_enabled(True)

        # Process multimodal inputs (images + text prompt)
        # NOTE: For Qwen2-VL/Qwen3-VL with MROPE, we let the model compute position_ids internally
        # during prefill when pixel_values are provided. The model's internal logic handles
        # the 3D positional encoding for image tokens correctly.
        # Using hooks to capture only needed layers (avoids storing all layers)
        prefill_kwargs_internal = dict(prefill_kwargs)
        self._clear_captured_states()

        # Instrument visual encoder to measure ViT time separately
        vit_start_event = torch.cuda.Event(enable_timing=True)
        vit_end_event = torch.cuda.Event(enable_timing=True)
        vit_hook = None
        visual = getattr(self.target_vlm.model, 'visual', None)
        # Only use hooks for VLM-based prefill (hooks path). For the integrated-style
        # prefill (capture-ids), we record events directly around the vision call.
        if visual is not None and pixel_values is not None and not self._use_capture_layer_ids:
            def _vit_pre_hook(module, args, kwargs):
                vit_start_event.record()
                return None
            def _vit_post_hook(module, args, kwargs, output):
                vit_end_event.record()
                return None
            vit_hook = visual.register_forward_pre_hook(_vit_pre_hook, with_kwargs=True)
            vit_post_hook = visual.register_forward_hook(_vit_post_hook, with_kwargs=True)

        # Patched TextModel needs cache_position for create_causal_mask during prefill
        use_compiled = self._torch_compile is not None
        use_static = use_compiled or self.config.use_static_cache
        prefill_cache_position = torch.arange(num_input_tokens, device=device) if use_static else None

        if self._use_capture_layer_ids:
            # ---- Integrated-style prefill (matches alpamayo_r1.py) ----
            # Decompose VLM call: encode vision separately, then call language_model
            # directly with StaticCache. This is identical to _dflash_prefill in
            # alpamayo_r1.py and ensures the prefill path can be integrated losslessly.

            # 1. Create StaticCache up front (like integrated path).
            # Size dynamically based on input length + generation budget.
            required_cache_len = num_input_tokens + max_new_tokens + block_size
            if self._static_cache is None or self._static_cache.max_cache_len < required_cache_len:
                text_config = self.target_vlm.config.text_config
                self._static_cache = StaticCache(
                    config=text_config,
                    max_cache_len=required_cache_len,
                    dtype=self._inference_dtype,
                )
                self._patch_static_cache_dtype(self._static_cache)
            self._static_cache.reset()

            # 2. Compute position_ids and rope_deltas
            if pixel_values is not None:
                position_ids, rope_deltas = self.target_vlm.model.get_rope_index(
                    input_ids, image_grid_thw,
                )
            else:
                position_ids = torch.arange(
                    num_input_tokens, device=device,
                ).unsqueeze(0)
                rope_deltas = None

            # 3. Get input embeddings
            inputs_embeds = self.target_vlm.model.get_input_embeddings()(input_ids)

            # 4. Encode vision and scatter into embeddings
            if pixel_values is not None:
                visual_model = self.target_vlm.model.visual
                vit_start_event.record()
                pixel_values_typed = pixel_values.type(visual_model.dtype)
                image_embeds, deepstack_image_embeds = visual_model(
                    pixel_values_typed, grid_thw=image_grid_thw,
                )
                vit_end_event.record()
                image_mask = (
                    input_ids == self.target_vlm.config.image_token_id
                ).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                visual_pos_masks = image_mask[..., 0]
            else:
                deepstack_image_embeds = None
                visual_pos_masks = None

            # 5. Language model prefill with StaticCache (captures hidden states)
            lm_output = self._language_model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                past_key_values=self._static_cache,
                cache_position=prefill_cache_position,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_image_embeds,
                use_cache=True,
            )
            prefill_logits = self._lm_head(lm_output.last_hidden_state[:, -1:, :])

            # 6. Extract hidden states captured during prefill
            captured = lm_output.hidden_states
            full_hidden = torch.cat(captured, dim=-1)
            target_hidden = full_hidden[:, -1:, :]  # (B, 1, context_dim)
            current_seq_len = num_input_tokens
        else:
            # ---- VLM-based prefill (hooks path) ----
            if pixel_values is not None:
                prefill_output = self.target_vlm(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    past_key_values=past_key_values_target,
                    cache_position=prefill_cache_position,
                    use_cache=True,
                    output_hidden_states=False,
                    **prefill_kwargs_internal,
                )
                rope_deltas = getattr(self.target_vlm.model, "rope_deltas", None)
            else:
                simple_position_ids = torch.arange(num_input_tokens, device=device).unsqueeze(0)
                prefill_output = self.target_vlm(
                    input_ids=input_ids,
                    position_ids=simple_position_ids,
                    past_key_values=past_key_values_target,
                    cache_position=prefill_cache_position,
                    use_cache=True,
                    output_hidden_states=False,
                    **prefill_kwargs_internal,
                )
                rope_deltas = None
            prefill_logits = prefill_output.logits[:, -1:, :]

            # Extract hidden states from hooks
            full_hidden = self._extract_hooked_hidden_states()
            target_hidden = full_hidden[:, -1:, :]  # (B, 1, hidden_dim)
            current_seq_len = num_input_tokens

        # Remove ViT timing hooks
        if vit_hook is not None:
            vit_hook.remove()
            vit_post_hook.remove()

        # Copy input tokens to output
        output_ids[:, :num_input_tokens] = input_ids

        # Sample first token from prefill logits
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample_tokens(
            prefill_logits, temperature, self._logits_processor
        )

        # Synchronize to capture full prefill GPU time and prevent bleeding into decode
        torch.cuda.synchronize()
        stats.prefill_time_ms = (time.perf_counter() - prefill_start) * 1000

        # Disable FP8 for decode/verify — draft model was trained against AWQ outputs,
        # so decode must use AWQ for consistent speculative decoding acceptance.
        self._set_fp8_enabled(False)

        # Transfer DynamicCache → StaticCache for hooks path with static cache.
        # Skip for capture-ids path (already using StaticCache from prefill).
        if use_static and not self._use_capture_layer_ids:
            if use_compiled:
                from alpamayo_r1.models.patches import patch_for_torch_compile
                if not hasattr(self, "_patched_for_compile"):
                    patch_for_torch_compile(self.target_vlm, mode="non-streaming")
                    self._patched_for_compile = True
            self._transfer_dynamic_to_static_cache(past_key_values_target, current_seq_len)

        # Compute ViT vs LLM breakdown
        if visual is not None and pixel_values is not None:
            stats.vit_time_ms = vit_start_event.elapsed_time(vit_end_event)
            stats.llm_prefill_time_ms = stats.prefill_time_ms - stats.vit_time_ms
        else:
            stats.vit_time_ms = 0.0
            stats.llm_prefill_time_ms = stats.prefill_time_ms

        # Log prefill completion
        if log_handle is not None:
            log_handle.write(json.dumps({
                "timestamp_ms": (time.perf_counter() - generation_start_time) * 1000,
                "phase": "prefill_end",
                "prefill_ms": stats.prefill_time_ms,
            }) + '\n')
            log_handle.flush()

        # ====== DECODE STAGE ======
        decode_start = time.perf_counter()
        start = num_input_tokens

        # Precompute tensors that are constant across iterations (avoid per-iteration allocations)
        # Draft model always uses relative positions 0..block_size
        reset_position_ids = torch.arange(
            0, 1 + block_size, device=device
        ).unsqueeze(0).expand(bsz, -1)
        # Base positions for target model (will add `start` offset each iteration)
        base_block_positions = torch.arange(0, block_size, device=device, dtype=torch.long)
        # Precompute rope_deltas tensor if needed
        if rope_deltas is not None:
            rope_deltas_long = rope_deltas.to(dtype=torch.long, device=device)

        # CUDA events for detailed timing (only created if needed)
        # Events allow measuring GPU time without blocking CPU-GPU overlap
        if enable_detailed_timing:
            draft_events = []  # List of (start_event, end_event) tuples
            verify_events = []
            sample_events = []
            cache_events = []
            lm_head_events = []

        while start < max_length:
            # 1. Prepare the block (first token is known, rest are masks)
            block_output_ids = output_ids[:, start : start + block_size].clone()

            # Compute position IDs for TARGET MODEL verification (uses absolute positions)
            # Use precomputed base + offset to avoid per-iteration tensor allocation
            block_positions = base_block_positions + start
            if rope_deltas is not None:
                # MROPE: Create 3D position IDs, applying rope_deltas offset
                block_position_ids = block_positions.view(1, 1, -1).expand(3, bsz, -1) + rope_deltas_long.unsqueeze(-1)
            else:
                # Standard 1D position IDs
                block_position_ids = block_positions.unsqueeze(0).expand(bsz, -1)

            # Note: reset_position_ids for draft model is precomputed outside loop (constant 0..block_size)

            # 3-5. Draft: embed + forward + lm_head (stateless mode, no KV cache)
            current_context = target_hidden[:, -1:, :]  # (B, 1, hidden_dim)

            if enable_detailed_timing:
                draft_start_event = torch.cuda.Event(enable_timing=True)
                draft_end_event = torch.cuda.Event(enable_timing=True)
                draft_start_event.record()

            if use_compiled:
                _draft_hidden, draft_logits = self._draft(
                    block_output_ids, current_context, reset_position_ids,
                )
            else:
                noise_embedding = self._embed_tokens(block_output_ids)
                draft_hidden = self.draft_model(
                    target_hidden=current_context,
                    noise_embedding=noise_embedding,
                    position_ids=reset_position_ids,
                    past_key_values=None,
                    use_cache=False,
                    is_causal=False,
                )
                draft_logits = self._lm_head(draft_hidden[:, 1:, :])

            if enable_detailed_timing:
                draft_end_event.record()
                draft_events.append((draft_start_event, draft_end_event))
                # LM head is now inside draft graph — record zero-duration placeholder
                lm_head_start_event = torch.cuda.Event(enable_timing=True)
                lm_head_end_event = torch.cuda.Event(enable_timing=True)
                lm_head_start_event.record()
                lm_head_end_event.record()
                lm_head_events.append((lm_head_start_event, lm_head_end_event))

            # Sample draft tokens (apply logits processor to prevent trajectory tokens)
            if enable_detailed_timing:
                sample_start_event = torch.cuda.Event(enable_timing=True)
                sample_end_event = torch.cuda.Event(enable_timing=True)
                sample_start_event.record()

            block_output_ids[:, 1:] = sample_tokens(
                draft_logits, temperature, self._logits_processor
            )

            if enable_detailed_timing:
                sample_end_event.record()
                sample_events.append((sample_start_event, sample_end_event))

            # Verify: Run target model on the drafted block
            if enable_detailed_timing:
                verify_start_event = torch.cuda.Event(enable_timing=True)
                verify_end_event = torch.cuda.Event(enable_timing=True)
                verify_start_event.record()

            if use_compiled:
                verify_cache_position = torch.arange(
                    current_seq_len, current_seq_len + block_size, device=device
                )
                verify_logits, verify_context = self._verify(
                    block_output_ids, block_position_ids, verify_cache_position,
                )
            elif use_static:
                verify_cache_position = torch.arange(
                    current_seq_len, current_seq_len + block_size, device=device
                )
                self._clear_captured_states()
                # Call language model directly (VLM wrapper doesn't forward
                # cache_position properly, causing shape mismatches with StaticCache)
                inputs_embeds = self._embed_tokens(block_output_ids)
                lm_output = self._language_model(
                    inputs_embeds=inputs_embeds,
                    position_ids=block_position_ids,
                    past_key_values=self._static_cache,
                    cache_position=verify_cache_position,
                    use_cache=True,
                )
                verify_logits = self._lm_head(lm_output[0])
                if self._use_capture_layer_ids:
                    verify_context = torch.cat(lm_output.hidden_states, dim=-1)
            else:
                self._clear_captured_states()
                verify_output = self.target_vlm(
                    input_ids=block_output_ids,
                    position_ids=block_position_ids,
                    past_key_values=past_key_values_target,
                    use_cache=True,
                    output_hidden_states=False,  # Using hooks instead
                )
                verify_logits = verify_output.logits

            if enable_detailed_timing:
                verify_end_event.record()
                verify_events.append((verify_start_event, verify_end_event))

            # Sample from target model's logits (apply logits processor)
            if enable_detailed_timing:
                sample2_start_event = torch.cuda.Event(enable_timing=True)
                sample2_end_event = torch.cuda.Event(enable_timing=True)
                sample2_start_event.record()

            posterior = sample_tokens(
                verify_logits, temperature, self._logits_processor
            )

            if enable_detailed_timing:
                sample2_end_event.record()
                sample_events.append((sample2_start_event, sample2_end_event))

            # Compute acceptance: count consecutive matching tokens
            matches = block_output_ids[:, 1:] == posterior[:, :-1]
            acceptance_length = matches.cumprod(dim=1).sum(dim=1)[0].item()

            # Accept matched tokens + target's next token
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

            # Check for stop tokens in the ENTIRE accepted sequence, not just the last token
            # This handles cases where <|cot_end|> appears mid-block
            # NOTE: Using tensor ops to minimize CPU-GPU sync points
            hit_stop = False
            stop_position = None
            tokens_to_advance = acceptance_length + 1  # default: acceptance_length drafts + 1 posterior
            period_token_id = 13  # "." token triggers stop heuristic
            if stop_token_ids is not None:
                accepted_tokens = output_ids[0, start : start + acceptance_length + 2]
                stop_token = stop_token_ids[0]

                # Find stop token using tensor ops (single sync at .any())
                is_stop = (accepted_tokens == stop_token)
                if is_stop.any():
                    # Get first stop position
                    stop_position = is_stop.nonzero(as_tuple=True)[0][0].item()
                    hit_stop = True
                    tokens_to_advance = stop_position
                    # Clear tokens after stop token
                    if stop_position < acceptance_length + 1:
                        output_ids[:, start + stop_position + 1:] = self.mask_token_id
                # Also check if posterior is "." - period heuristic will stop next iteration
                elif posterior[0, acceptance_length].item() == period_token_id:
                    # Treat as stop at acceptance_length + 1 (the "." position)
                    hit_stop = True
                    stop_position = acceptance_length + 1

            # Update position and stats
            # tokens_to_advance already includes the +1 for posterior token
            start += tokens_to_advance
            # acceptance_lengths tracks total tokens accepted (draft + posterior, max 8)
            if hit_stop and stop_position is not None:
                # Stop step: tokens before stop position
                stats.acceptance_lengths.append(stop_position)
            else:
                # Non-stop step: draft matched + 1 posterior
                stats.acceptance_lengths.append(acceptance_length + 1)
            # draft_matches: if stop token found, use stop_position (capped at block_size-1)
            # otherwise use acceptance_length
            # tokens_verified: how many draft tokens were verified before stop/end
            #   - If stop token hit at position P: min(P, block_size-1) tokens verified
            #   - If no stop: all block_size-1 draft tokens were verified
            if hit_stop and stop_position is not None:
                # For stop step: only count draft tokens before <|cot_end|>
                # stop_position is index in block, position 0 is first known token
                # Draft tokens are positions 1 to stop_position-1
                draft_matched = min(stop_position - 1, block_size - 1)
                draft_verified = draft_matched  # same tokens
                stats.draft_matches.append(draft_matched)
                stats.tokens_verified.append(draft_verified)
                stats.hit_stop.append(True)
            else:
                # For non-stop step: count draft tokens only (positions 1-7)
                stats.draft_matches.append(acceptance_length)  # pure draft matches
                stats.tokens_verified.append(block_size - 1)  # 7 draft tokens
                stats.hit_stop.append(False)
            stats.drafting_iterations += 1
            stats.total_iterations += 1

            # Log decode step (write output token IDs for streaming replay)
            if log_handle is not None:
                total_accepted = sum(stats.acceptance_lengths)
                total_possible = stats.total_iterations * block_size
                # Get valid output tokens (non-mask) up to current position
                current_end = min(start, output_ids.shape[1])
                valid_ids = output_ids[0, :current_end].tolist()
                log_handle.write(json.dumps({
                    "timestamp_ms": (time.perf_counter() - generation_start_time) * 1000,
                    "phase": "decode_step",
                    "step": stats.total_iterations,
                    "tokens_generated": start - num_input_tokens,
                    "acceptance_rate": total_accepted / total_possible if total_possible > 0 else 0,
                    "acceptance_length": stats.acceptance_lengths[-1],
                    "output_ids": valid_ids,
                }) + '\n')
                log_handle.flush()

            # Stop immediately if we hit stop token
            if hit_stop:
                break

            # Update KV cache and extract hidden states
            if use_static:
                # StaticCache: zero out stale entries beyond the valid length
                current_seq_len += acceptance_length + 1
                if acceptance_length < block_size - 1:
                    if enable_detailed_timing:
                        cache_start_event = torch.cuda.Event(enable_timing=True)
                        cache_end_event = torch.cuda.Event(enable_timing=True)
                        cache_start_event.record()

                    self._crop_static_cache(current_seq_len)

                    if enable_detailed_timing:
                        cache_end_event.record()
                        cache_events.append((cache_start_event, cache_end_event))

                if use_compiled or self._use_capture_layer_ids:
                    # Extract hidden state from verify context (returned by _verify or capture_layer_ids)
                    target_hidden = verify_context[:, acceptance_length : acceptance_length + 1, :]
                else:
                    # Extract hidden state from hooks
                    new_hidden = self._extract_hooked_hidden_states()
                    target_hidden = new_hidden[:, acceptance_length : acceptance_length + 1, :]
            else:
                # DynamicCache: crop to keep only valid positions
                if acceptance_length < block_size - 1:
                    if enable_detailed_timing:
                        cache_start_event = torch.cuda.Event(enable_timing=True)
                        cache_end_event = torch.cuda.Event(enable_timing=True)
                        cache_start_event.record()

                    past_key_values_target.crop(start)

                    if enable_detailed_timing:
                        cache_end_event.record()
                        cache_events.append((cache_start_event, cache_end_event))

                # Extract hidden state from hooks
                # Key insight: hidden at index `acceptance_length` was computed using
                # ONLY accepted tokens (causal attention). It's correct to use directly.
                new_hidden = self._extract_hooked_hidden_states()
                target_hidden = new_hidden[:, acceptance_length : acceptance_length + 1, :]

        # Compute detailed timing from CUDA events (single sync point at end)
        if enable_detailed_timing:
            torch.cuda.synchronize()  # Wait for all GPU ops to complete
            for start_evt, end_evt in draft_events:
                stats.draft_time_ms += start_evt.elapsed_time(end_evt)
            for start_evt, end_evt in verify_events:
                stats.verify_time_ms += start_evt.elapsed_time(end_evt)
            for start_evt, end_evt in sample_events:
                stats.sample_time_ms += start_evt.elapsed_time(end_evt)
            for start_evt, end_evt in cache_events:
                stats.cache_time_ms += start_evt.elapsed_time(end_evt)
            for start_evt, end_evt in lm_head_events:
                stats.lm_head_time_ms += start_evt.elapsed_time(end_evt)

        stats.decode_time_ms = (time.perf_counter() - decode_start) * 1000

        # Cleanup output
        output_ids = output_ids[:, :max_length]
        # Remove mask tokens
        mask = output_ids[0] != self.mask_token_id
        output_ids = output_ids[:, mask]

        # Truncate at stop token if found
        if stop_token_ids is not None:
            stop_token_ids_tensor = torch.tensor(stop_token_ids, device=device)
            generated_tokens = output_ids[0, num_input_tokens:]
            stop_indices = torch.isin(generated_tokens, stop_token_ids_tensor).nonzero(as_tuple=True)[0]
            if stop_indices.numel() > 0:
                output_ids = output_ids[:, :num_input_tokens + stop_indices[0] + 1]

        stats.total_tokens = output_ids.shape[1] - num_input_tokens

        # Log final stats and close log file
        if log_handle is not None:
            log_handle.write(json.dumps({
                "timestamp_ms": (time.perf_counter() - generation_start_time) * 1000,
                "phase": "done",
                "total_tokens": stats.total_tokens,
                "prefill_ms": stats.prefill_time_ms,
                "decode_ms": stats.decode_time_ms,
                "total_iterations": stats.total_iterations,
                "acceptance_rate": stats.acceptance_rate,
                "final_output_ids": output_ids[0].tolist(),
            }) + '\n')
            log_handle.close()

        if use_static:
            # Convert StaticCache → DynamicCache for downstream compatibility.
            # The DFlash decode loop uses StaticCache throughout; this conversion
            # is only for sample_trajectories_with_cache which expects DynamicCache.
            kv_cache = DynamicCache()
            valid_len = current_seq_len
            for layer_idx, layer in enumerate(self._static_cache.layers):
                if layer.is_initialized:
                    k = layer.keys[:, :, :valid_len, :].clone()
                    v = layer.values[:, :, :valid_len, :].clone()
                    kv_cache.update(k, v, layer_idx)
        else:
            kv_cache = past_key_values_target
        return output_ids, stats, kv_cache, rope_deltas


def load_dflash_draft_model(
    draft_model_name_or_path: str = "~/exp/dflash",
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    dflash_path: str | Path | None = None,
    local_files_only: bool = True,
) -> nn.Module:
    """Load a DFlash draft model.

    Args:
        draft_model_name_or_path: HuggingFace model ID or local path.
        device: Device to load the model on.
        dtype: Data type for the model.
        dflash_path: Path to DFlash source code (for importing).
            Defaults to project_root/dflash.
        local_files_only: Only use local files (no HuggingFace download).

    Returns:
        The loaded DFlash draft model. The model may have a `mask_token_id` attribute
        if it was saved with the training config; use this value for DFlashConfig.mask_token_id_override.
    """
    # Add DFlash to path if needed
    if dflash_path is None:
        # Default: dflash folder is at project root (../../dflash from this file)
        dflash_path = Path(__file__).parent.parent.parent / "dflash"
    dflash_path = Path(dflash_path)

    if str(dflash_path) not in sys.path:
        sys.path.insert(0, str(dflash_path))

    from alpamayo_r1.dflash_model import DFlashDraftModel

    logger.info(f"Loading DFlash draft model from {draft_model_name_or_path}")

    draft_model = DFlashDraftModel.from_pretrained(
        draft_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    ).to(device).eval()

    # Log model info
    info_parts = [
        f"{draft_model.config.num_hidden_layers} layers",
        f"block_size={draft_model.block_size}",
    ]

    # Check for mask token ID in model config (important for correct setup)
    mask_token_id = getattr(draft_model.config, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = getattr(draft_model, "mask_token_id", None)
    if mask_token_id is not None:
        draft_model.mask_token_id = mask_token_id  # Ensure it's accessible
        info_parts.append(f"mask_token_id={mask_token_id}")
        logger.info(
            f"[DFlash] Found mask_token_id={mask_token_id} in checkpoint. "
            "Use this value for DFlashConfig.mask_token_id_override for best results."
        )

    logger.info(f"DFlash model loaded: {', '.join(info_parts)}")

    return draft_model


def create_dflash_accelerator(
    alpamayo_model: nn.Module,
    draft_model_name_or_path: str = "~/exp/dflash",
    config: DFlashConfig | None = None,
    dflash_path: str | Path | None = None,
    local_files_only: bool = True,
) -> DFlashAlpamayoAccelerator:
    """Create a DFlash accelerator for an Alpamayo model.

    Convenience function that loads the draft model and creates the accelerator.
    Automatically configures trajectory token masking from the Alpamayo model config.

    Args:
        alpamayo_model: The Alpamayo model instance.
        draft_model_name_or_path: Path to the DFlash draft model.
        config: DFlash configuration. If None, will be auto-configured with
            trajectory token masking from the Alpamayo model.
        dflash_path: Path to DFlash source code.
        local_files_only: Only use local files (no HuggingFace download).

    Returns:
        Configured DFlashAlpamayoAccelerator.
    """
    device = next(alpamayo_model.parameters()).device
    dtype = next(alpamayo_model.parameters()).dtype

    # Load draft model
    draft_model = load_dflash_draft_model(
        draft_model_name_or_path,
        device=device,
        dtype=dtype,
        dflash_path=dflash_path,
        local_files_only=local_files_only,
    )

    # Auto-configure from Alpamayo and draft model if not provided
    if config is None:
        config = DFlashConfig()

    # Extract mask token ID from draft model (critical for correct embedding lookup)
    if config.mask_token_id_override is None:
        draft_mask_id = getattr(draft_model, "mask_token_id", None)
        if draft_mask_id is None:
            draft_mask_id = getattr(draft_model.config, "mask_token_id", None)
        if draft_mask_id is not None:
            config.mask_token_id_override = draft_mask_id
            logger.info(f"[DFlash] Auto-configured mask_token_id_override={draft_mask_id} from draft model")

    # Extract trajectory token configuration from Alpamayo model
    if config.traj_token_offset is None and hasattr(alpamayo_model, "config"):
        alpamayo_cfg = alpamayo_model.config
        if hasattr(alpamayo_cfg, "traj_token_start_idx"):
            config.traj_token_offset = alpamayo_cfg.traj_token_start_idx
            logger.info(f"[DFlash] Auto-configured traj_token_offset={config.traj_token_offset}")
        if hasattr(alpamayo_cfg, "traj_vocab_size"):
            config.traj_vocab_size = alpamayo_cfg.traj_vocab_size
            logger.info(f"[DFlash] Auto-configured traj_vocab_size={config.traj_vocab_size}")

    # Create accelerator
    accelerator = DFlashAlpamayoAccelerator(
        draft_model=draft_model,
        target_vlm=alpamayo_model.vlm,
        tokenizer=alpamayo_model.tokenizer,
        config=config,
    )

    # Load exact MASK embedding from training if available
    # This eliminates train-inference mismatch from re-computing vocab mean
    draft_path = Path(draft_model_name_or_path)
    mask_emb_file = draft_path / "mask_embedding.pt"

    if mask_emb_file.exists():
        logger.info(f"[DFlash] Loading exact mask embedding from {mask_emb_file}")
        mask_emb = torch.load(mask_emb_file, map_location=device)

        # Ensure dtype compatibility
        mask_emb = mask_emb.to(dtype=alpamayo_model.vlm.get_input_embeddings().weight.dtype)

        # Overwrite the mask embedding with the exact one from training
        with torch.no_grad():
            alpamayo_model.vlm.get_input_embeddings().weight[accelerator.mask_token_id] = mask_emb

        logger.info(f"[DFlash] Loaded exact mask embedding for token ID {accelerator.mask_token_id}")
    else:
        logger.warning(
            f"[DFlash] No mask_embedding.pt found in {draft_path}. "
            "Using vocabulary mean initialization (may cause train-inference mismatch)."
        )

    return accelerator


def setup_dflash_for_model(
    model: nn.Module,
    draft_model_name_or_path: str,
    dflash_path: str | Path | None = None,
    local_files_only: bool = True,
) -> nn.Module:
    """Load a DFlash draft model and configure it on an AlpamayoR1 model.

    This is the recommended way to enable DFlash for integrated inference.
    After calling this, use ``model.sample_trajectories(data, dflash=True)``
    to run inference with speculative decoding.

    Args:
        model: The AlpamayoR1 model instance.
        draft_model_name_or_path: Path to the DFlash draft model checkpoint.
        dflash_path: Path to DFlash source code (for importing).
        local_files_only: Only use local files (no HuggingFace download).

    Returns:
        The model with DFlash configured (same instance, modified in-place).
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Load draft model
    draft_model = load_dflash_draft_model(
        draft_model_name_or_path,
        device=device,
        dtype=dtype,
        dflash_path=dflash_path,
        local_files_only=local_files_only,
    )

    # Extract mask token ID from draft model
    mask_token_id = getattr(draft_model, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = getattr(draft_model.config, "mask_token_id", None)

    # Handle mask embedding: resize if needed
    if mask_token_id is not None:
        vocab_size = model.vlm.get_input_embeddings().weight.shape[0]
        if mask_token_id == vocab_size:
            logger.info(f"[DFlash] Resizing embeddings from {vocab_size} to {vocab_size + 1} for mask token")
            model.vlm.resize_token_embeddings(vocab_size + 1)
        elif mask_token_id > vocab_size:
            raise ValueError(
                f"mask_token_id={mask_token_id} is out of vocabulary range ({vocab_size})."
            )

    # Configure DFlash on the model
    model.setup_dflash(
        draft_model=draft_model,
        mask_token_id=mask_token_id,
    )

    # Load exact mask embedding from training if available
    draft_path = Path(draft_model_name_or_path)
    mask_emb_file = draft_path / "mask_embedding.pt"

    if mask_emb_file.exists():
        logger.info(f"[DFlash] Loading exact mask embedding from {mask_emb_file}")
        mask_emb = torch.load(mask_emb_file, map_location=device)
        mask_emb = mask_emb.to(dtype=model.vlm.get_input_embeddings().weight.dtype)
        with torch.no_grad():
            model.vlm.get_input_embeddings().weight[mask_token_id] = mask_emb
        logger.info(f"[DFlash] Loaded exact mask embedding for token ID {mask_token_id}")
    elif mask_token_id is not None:
        # Initialize to vocabulary mean as fallback
        logger.info(f"[DFlash] Initializing mask token {mask_token_id} embedding to vocabulary mean")
        input_embeddings = model.vlm.get_input_embeddings()
        with torch.no_grad():
            mean_embedding = input_embeddings.weight[:-1].mean(dim=0)
            input_embeddings.weight[mask_token_id] = mean_embedding

    return model
