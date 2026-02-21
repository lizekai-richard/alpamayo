import torch
from typing import Callable

# =============================================================================
# Streaming Attention Mask Utilities
# =============================================================================
def create_streaming_attention_mask_sdpa(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    vision_start_end_ids_ranges: list[list[tuple[int, int]]],
    traj_and_text_ids_range: tuple[int, int],
    valid_length: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """

    Create 4D attention mask for streaming VLM using SDPA.

    Query structure: [V0_F3] + [V1_F3] + [V2_F3] + [V3_F3] + [Traj+Text]
    KV structure: [System] + [V0_F0123] + [V1_F0123] + [V2_F0123] + [V3_F0123] + [Traj+Text]

    Attention rules:
    1. System tokens are fully visible to all query tokens
    2. View_i's F3 can attend to View0~View_i (all frames), causal within own F3
    3. Traj+Text can attend to all previous content, causal within itself

    Visual representation (■ = attend, □ = masked, ◣ = causal):

                  KV: | Sys | V0_Frames | V1_Frames | V2_Frames | V3_Frames | Traj+Text |
    Query:           |     | F0 F1 F2 F3| F0 F1 F2 F3| F0 F1 F2 F3| F0 F1 F2 F3|           |
    -----------------|-----|-----------|-----------|-----------|-----------|-----------|
    V0_F3            |  ■  |  ■  ■  ■ ◣ |  □  □  □  □|  □  □  □  □|  □  □  □  □|     □     |
    V1_F3            |  ■  |  ■  ■  ■  ■|  ■  ■  ■ ◣ |  □  □  □  □|  □  □  □  □|     □     |
    V2_F3            |  ■  |  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■ ◣ |  □  □  □  □|     □     |
    V3_F3            |  ■  |  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■ ◣ |     □     |
    Traj+Text        |  ■  |  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■  ■|     ◣     |

    Args:
        batch_size: Batch size
        cache_position: Query positions in the full sequence [query_length]
        kv_length: Total KV cache length
        vision_start_end_ids_ranges: Ranges for each view's frames
        traj_and_text_ids_range: (start, end) range for trajectory and text tokens
        valid_length: Positions >= valid_length in KV dimension will be masked as padding.
        device: Device to create mask on
        dtype: Data type for the mask

    Returns:
        attention_mask: [batch_size, 1, query_length, kv_length] with padding masked
    """
    num_views = len(vision_start_end_ids_ranges)
    query_length = cache_position.shape[0]

    # Initialize with -inf (all masked)
    min_val = torch.finfo(dtype).min
    attention_mask = torch.full(
        (batch_size, 1, query_length, kv_length),
        min_val,
        dtype=dtype,
        device=device,
    )

    # Precompute view boundaries
    view_kv_ranges = []
    for view_idx in range(num_views):
        view_start = vision_start_end_ids_ranges[view_idx][0][0]
        view_end = vision_start_end_ids_ranges[view_idx][-1][1]
        view_kv_ranges.append((view_start, view_end))

    system_end = view_kv_ranges[0][0]

    # Create KV position tensor for vectorized comparison
    kv_positions = torch.arange(kv_length, device=device)

    # Build query-to-view mapping and compute frame lengths
    q_offset = 0
    query_view_mapping = []  # (q_start, q_end, view_idx, is_traj_text, frame_start_kv)

    for view_idx in range(num_views):
        last_frame_start, last_frame_end = vision_start_end_ids_ranges[view_idx][-1]
        frame_length = last_frame_end - last_frame_start
        query_view_mapping.append((q_offset, q_offset + frame_length, view_idx, False, last_frame_start))
        q_offset += frame_length

    # Add traj+text
    traj_start_kv, traj_end_kv = traj_and_text_ids_range
    traj_length = traj_end_kv - traj_start_kv
    query_view_mapping.append((q_offset, q_offset + traj_length, num_views, True, traj_start_kv))

    # Process each query region
    for q_start, q_end, view_idx, is_traj_text, frame_start_kv in query_view_mapping:
        region_length = q_end - q_start

        # System tokens - always visible
        attention_mask[:, :, q_start:q_end, :system_end] = 0

        if not is_traj_text:
            # Image query: can see View_0 to View_{view_idx-1} fully
            for prev_idx in range(view_idx):
                prev_start, prev_end = view_kv_ranges[prev_idx]
                attention_mask[:, :, q_start:q_end, prev_start:prev_end] = 0

            # Can see own view's earlier frames (F0, F1, F2)
            for frame_idx in range(len(vision_start_end_ids_ranges[view_idx]) - 1):
                f_start, f_end = vision_start_end_ids_ranges[view_idx][frame_idx]
                attention_mask[:, :, q_start:q_end, f_start:f_end] = 0

            # Causal within F3: vectorized
            q_indices = torch.arange(region_length, device=device).unsqueeze(1)  # [region_length, 1]
            kv_indices = torch.arange(region_length, device=device).unsqueeze(0)  # [1, region_length]
            causal_mask = kv_indices <= q_indices  # [region_length, region_length]

            # Apply to the F3 region
            attention_mask[:, :, q_start:q_end, frame_start_kv:frame_start_kv + region_length] = torch.where(
                causal_mask.unsqueeze(0).unsqueeze(0),
                torch.tensor(0.0, dtype=dtype, device=device),
                torch.tensor(min_val, dtype=dtype, device=device),
            )
        else:
            # Traj+Text query: can see all views
            for v_idx in range(num_views):
                v_start, v_end = view_kv_ranges[v_idx]
                attention_mask[:, :, q_start:q_end, v_start:v_end] = 0

            # Causal within traj+text: vectorized
            q_indices = torch.arange(region_length, device=device).unsqueeze(1)
            kv_indices = torch.arange(region_length, device=device).unsqueeze(0)
            causal_mask = kv_indices <= q_indices

            attention_mask[:, :, q_start:q_end, frame_start_kv:frame_start_kv + region_length] = torch.where(
                causal_mask.unsqueeze(0).unsqueeze(0),
                torch.tensor(0.0, dtype=dtype, device=device),
                torch.tensor(min_val, dtype=dtype, device=device),
            )

    # Mask padding positions: all KV positions >= valid_length should be masked
    if valid_length < kv_length:
        attention_mask[:, :, :, valid_length:] = min_val

    return attention_mask


def create_streaming_attention_mask_sdpa_training(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    vision_start_end_ids_ranges: list[list[tuple[int, int]]],
    traj_and_text_ids_range: tuple[int, int],
    output_ids_range: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create streaming attention mask for training with output tokens appended to input.

    Compared to `create_streaming_attention_mask_sdpa`, this function additionally
    handles output tokens in query by using `cache_position` directly:
    - query positions inside `output_ids_range` can attend all non-output KV tokens
    - output-token attention is causal in absolute KV positions

    Visual representation (■ = attend, □ = masked, ◣ = causal):

                  KV: | Sys | V0_Frames | V1_Frames | V2_Frames | V3_Frames | Traj+Text | Output |
    Query:           |     | F0 F1 F2 F3| F0 F1 F2 F3| F0 F1 F2 F3| F0 F1 F2 F3|           |        |
    -----------------|-----|-----------|-----------|-----------|-----------|-----------|--------|
    V0_F3            |  ■  |  ■  ■  ■ ◣ |  □  □  □  □|  □  □  □  □|  □  □  □  □|     □     |   □    |
    V1_F3            |  ■  |  ■  ■  ■  ■|  ■  ■  ■ ◣ |  □  □  □  □|  □  □  □  □|     □     |   □    |
    V2_F3            |  ■  |  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■ ◣ |  □  □  □  □|     □     |   □    |
    V3_F3            |  ■  |  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■ ◣ |     □     |   □    |
    Traj+Text        |  ■  |  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■  ■|     ◣     |   □    |
    Output           |  ■  |  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■  ■|     ■     |   ◣    |

    The Output row can attend to ALL non-output KV tokens (Sys, Views, Traj+Text)
    fully, and is causal within its own Output segment.
    """
    output_start_kv, output_end_kv = output_ids_range

    attention_mask = create_streaming_attention_mask_sdpa(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        vision_start_end_ids_ranges=vision_start_end_ids_ranges,
        traj_and_text_ids_range=traj_and_text_ids_range,
        valid_length=output_start_kv,
        device=device,
        dtype=dtype,
    )
    if output_start_kv < 0 or output_end_kv > kv_length or output_start_kv > output_end_kv:
        raise ValueError(
            f"Invalid output_ids_range={output_ids_range} for kv_length={kv_length}."
        )

    output_length = output_end_kv - output_start_kv
    if output_length == 0:
        return attention_mask

    min_val = torch.finfo(dtype).min
    zero = torch.tensor(0.0, dtype=dtype, device=device)
    neg_inf = torch.tensor(min_val, dtype=dtype, device=device)

    # Locate output queries by absolute positions from cache_position.
    output_query_mask = (cache_position >= output_start_kv) & (cache_position < output_end_kv)
    if not output_query_mask.any():
        # No output query token in this call; base streaming mask is sufficient.
        return attention_mask

    output_query_indices = torch.where(output_query_mask)[0]
    output_query_positions = cache_position[output_query_indices]

    # Reset output query rows first.
    attention_mask[:, :, output_query_indices, :] = min_val

    kv_positions = torch.arange(kv_length, device=device)
    # All non-output KV tokens are visible.
    non_output_kv_indices = torch.where(
        (kv_positions < output_start_kv) | (kv_positions >= output_end_kv)
    )[0]
    attention_mask[
        :, :, output_query_indices.unsqueeze(-1), non_output_kv_indices.unsqueeze(0)
    ] = 0

    # Causal within output segment in absolute KV positions.
    output_kv_positions = torch.arange(output_start_kv, output_end_kv, device=device)
    causal_mask = output_kv_positions.unsqueeze(0) <= output_query_positions.unsqueeze(1)
    attention_mask[:, :, output_query_indices, output_start_kv:output_end_kv] = torch.where(
        causal_mask.unsqueeze(0).unsqueeze(0),
        zero,
        neg_inf,
    )

    # Keep padding masked.
    if output_end_kv < kv_length:
        attention_mask[:, :, :, output_end_kv:] = min_val

    return attention_mask


# =============================================================================
# Flex Attention Mask Utilities (requires PyTorch >= 2.5, preferably >= 2.6)
# =============================================================================

def _build_streaming_position_mappings(
    kv_length: int,
    query_length: int,
    vision_start_end_ids_ranges: list[list[tuple[int, int]]],
    traj_and_text_ids_range: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build position mapping tensors for Flex Attention mask_mod function.

    Returns:
        kv_region: [kv_length] - region index for each KV position
                   -1 = system, 0-3 = view index, num_views = traj_text
        kv_frame: [kv_length] - frame index within view for each KV position
                  -1 for non-image positions
        kv_local_pos: [kv_length] - local position within frame/region
        query_region: [query_length] - region index for each query position
        query_frame: [query_length] - frame index for each query position
        query_local_pos: [query_length] - local position within frame/region
    """
    num_views = len(vision_start_end_ids_ranges)

    # Initialize KV mappings
    kv_region = torch.full((kv_length,), -1, dtype=torch.long, device=device)  # -1 = system
    kv_frame = torch.full((kv_length,), -1, dtype=torch.long, device=device)
    kv_local_pos = torch.zeros(kv_length, dtype=torch.long, device=device)

    # Fill in view regions
    for view_idx in range(num_views):
        for frame_idx, (start, end) in enumerate(vision_start_end_ids_ranges[view_idx]):
            kv_region[start:end] = view_idx
            kv_frame[start:end] = frame_idx
            kv_local_pos[start:end] = torch.arange(end - start, device=device)

    # Fill in traj+text region
    traj_start, traj_end = traj_and_text_ids_range
    kv_region[traj_start:traj_end] = num_views  # traj_text region
    kv_frame[traj_start:traj_end] = -1  # no frame concept
    kv_local_pos[traj_start:traj_end] = torch.arange(traj_end - traj_start, device=device)

    # Build query mappings
    query_region = torch.zeros(query_length, dtype=torch.long, device=device)
    query_frame = torch.zeros(query_length, dtype=torch.long, device=device)
    query_local_pos = torch.zeros(query_length, dtype=torch.long, device=device)

    q_offset = 0
    for view_idx in range(num_views):
        last_frame_start, last_frame_end = vision_start_end_ids_ranges[view_idx][-1]
        frame_length = last_frame_end - last_frame_start
        last_frame_idx = len(vision_start_end_ids_ranges[view_idx]) - 1

        query_region[q_offset:q_offset + frame_length] = view_idx
        query_frame[q_offset:q_offset + frame_length] = last_frame_idx
        query_local_pos[q_offset:q_offset + frame_length] = torch.arange(frame_length, device=device)
        q_offset += frame_length

    # Traj+text in query
    traj_length = traj_end - traj_start
    query_region[q_offset:q_offset + traj_length] = num_views
    query_frame[q_offset:q_offset + traj_length] = -1
    query_local_pos[q_offset:q_offset + traj_length] = torch.arange(traj_length, device=device)

    return kv_region, kv_frame, kv_local_pos, query_region, query_frame, query_local_pos


def create_streaming_attention_mask_flex(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    vision_start_end_ids_ranges: list[list[tuple[int, int]]],
    traj_and_text_ids_range: tuple[int, int],
    device: torch.device,
):
    """
    Create BlockMask for streaming VLM using Flex Attention.

    This function creates a compressed block mask that can efficiently skip
    masked regions during attention computation.

    Attention pattern (same as SDPA version):
    - View_i's F3 can attend to System + View0~View_i (causal within own F3)
    - Traj+Text can attend to all previous, causal within itself

    Args:
        batch_size: Batch size
        cache_position: Query positions in the full sequence [query_length]
        kv_length: Total KV cache length
        vision_start_end_ids_ranges: Ranges for each view's frames
        traj_and_text_ids_range: (start, end) range for trajectory and text tokens
        device: Device to create mask on

    Returns:
        BlockMask: Compressed block mask for flex_attention

    Requires:
        PyTorch >= 2.5 (preferably >= 2.6 for better tensor indexing support)
    """
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError:
        raise ImportError(
            "Flex Attention requires PyTorch >= 2.5. "
            "Please upgrade PyTorch or use create_streaming_attention_mask_sdpa instead."
        )

    num_views = len(vision_start_end_ids_ranges)
    query_length = cache_position.shape[0]

    # Build position mappings
    (
        kv_region,
        kv_frame,
        kv_local_pos,
        query_region,
        query_frame,
        query_local_pos,
    ) = _build_streaming_position_mappings(
        kv_length=kv_length,
        query_length=query_length,
        vision_start_end_ids_ranges=vision_start_end_ids_ranges,
        traj_and_text_ids_range=traj_and_text_ids_range,
        device=device,
    )

    # Define the mask_mod function
    # This function will be compiled by Flex Attention
    def streaming_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        """
        Mask modification function for streaming attention.

        Returns True if the query can attend to the key, False otherwise.
        """
        # Get region/frame info for query and kv
        q_view = query_region[q_idx]
        q_frame = query_frame[q_idx]
        q_local = query_local_pos[q_idx]

        kv_view = kv_region[kv_idx]
        kv_frame_idx = kv_frame[kv_idx]
        kv_local = kv_local_pos[kv_idx]

        # Rule 1: System tokens (kv_view == -1) are always visible
        is_system = kv_view == -1

        # Rule 2: Check if query is an image token (view 0-3) or traj_text (view == num_views)
        is_image_query = q_view < num_views
        is_traj_query = q_view == num_views

        # For image queries:
        # - Can see earlier views (kv_view < q_view)
        # - Can see same view's earlier frames (kv_view == q_view and kv_frame < q_frame)
        # - Can see same frame with causal (kv_view == q_view and kv_frame == q_frame and kv_local <= q_local)
        earlier_view = is_image_query & (kv_view >= 0) & (kv_view < q_view)
        same_view_earlier_frame = is_image_query & (kv_view == q_view) & (kv_frame_idx < q_frame)
        same_frame_causal = is_image_query & (kv_view == q_view) & (kv_frame_idx == q_frame) & (kv_local <= q_local)

        image_can_attend = earlier_view | same_view_earlier_frame | same_frame_causal

        # For traj_text queries:
        # - Can see all image views (kv_view >= 0 and kv_view < num_views)
        # - Can see traj_text with causal (kv_view == num_views and kv_local <= q_local)
        can_see_all_images = is_traj_query & (kv_view >= 0) & (kv_view < num_views)
        traj_causal = is_traj_query & (kv_view == num_views) & (kv_local <= q_local)

        traj_can_attend = can_see_all_images | traj_causal

        # Combine all rules
        return is_system | image_can_attend | traj_can_attend

    # Create the block mask
    # Note: create_block_mask will compile the mask_mod function
    block_mask = create_block_mask(
        mask_mod=streaming_mask_mod,
        B=batch_size,
        H=None,  # None means the mask is the same for all heads
        Q_LEN=query_length,
        KV_LEN=kv_length,
        device=device,
    )

    return block_mask


def create_streaming_mask_mod_factory(
    kv_length: int,
    query_length: int,
    vision_start_end_ids_ranges: list[list[tuple[int, int]]],
    traj_and_text_ids_range: tuple[int, int],
    device: torch.device,
) -> Callable:
    """
    Factory function that creates a mask_mod function for Flex Attention.

    This is useful when you need the mask_mod function separately,
    e.g., for custom score_mod combinations.

    Returns:
        mask_mod: A callable (batch_idx, head_idx, q_idx, kv_idx) -> bool
    """
    num_views = len(vision_start_end_ids_ranges)

    # Build position mappings
    (
        kv_region,
        kv_frame,
        kv_local_pos,
        query_region,
        query_frame,
        query_local_pos,
    ) = _build_streaming_position_mappings(
        kv_length=kv_length,
        query_length=query_length,
        vision_start_end_ids_ranges=vision_start_end_ids_ranges,
        traj_and_text_ids_range=traj_and_text_ids_range,
        device=device,
    )

    def streaming_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        q_view = query_region[q_idx]
        q_frame = query_frame[q_idx]
        q_local = query_local_pos[q_idx]

        kv_view = kv_region[kv_idx]
        kv_frame_idx = kv_frame[kv_idx]
        kv_local = kv_local_pos[kv_idx]

        is_system = kv_view == -1

        is_image_query = q_view < num_views
        is_traj_query = q_view == num_views

        earlier_view = is_image_query & (kv_view >= 0) & (kv_view < q_view)
        same_view_earlier_frame = is_image_query & (kv_view == q_view) & (kv_frame_idx < q_frame)
        same_frame_causal = is_image_query & (kv_view == q_view) & (kv_frame_idx == q_frame) & (kv_local <= q_local)

        image_can_attend = earlier_view | same_view_earlier_frame | same_frame_causal

        can_see_all_images = is_traj_query & (kv_view >= 0) & (kv_view < num_views)
        traj_causal = is_traj_query & (kv_view == num_views) & (kv_local <= q_local)

        traj_can_attend = can_see_all_images | traj_causal

        return is_system | image_can_attend | traj_can_attend

    return streaming_mask_mod