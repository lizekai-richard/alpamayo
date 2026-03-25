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


def create_streaming_attention_mask_sdpa_v1p5(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    vision_start_end_ids_ranges: list[list[tuple[int, int]]],
    traj_and_text_ids_range: tuple[int, int],
    valid_length: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create 4D attention mask for streaming v1.5 (Setting B: vision_only + no_labels).

    Compared to the original ``create_streaming_attention_mask_sdpa``, this version
    uses *extended* view ranges that include camera name tokens and frame label
    tokens that sit in the gaps between views / between frames.

    Query structure: [V0_F3] + [V1_F3] + [V2_F3] + [V3_F3] + [Traj+Text]
    KV structure:
        [Sys] [cam0 lbl][V0_F0123 + frame_lbls] [cam1 lbl][V1_F0123 + frame_lbls] ... [Traj+Text]

    Attention rules for View_i's F3 query:
        1. System prompt (before any camera content)
        2. Camera labels + frame labels + vision frames of View_0 .. View_{i-1} (fully)
        3. Own view's camera label + frame labels + frames 0-2 + frame 3 label (fully)
        4. Causal within own F3 vision block [VS_last .. VE_last+1)

    Traj+Text sees all views (extended) + causal within itself.

    Visual representation (■ = attend, □ = masked, ◣ = causal):
    "Ext" = camera label + frame labels + vision frames for the entire view.

                  KV: | Sys |  V0_Ext   |  V1_Ext   |  V2_Ext   |  V3_Ext   | Traj+Text |
    Query:           |     | cam+F0..F3| cam+F0..F3| cam+F0..F3| cam+F0..F3|           |
    -----------------|-----|-----------|-----------|-----------|-----------|-----------|
    V0_F3            |  ■  |  ■  ■  ■ ◣ |  □  □  □  □|  □  □  □  □|  □  □  □  □|     □     |
    V1_F3            |  ■  |  ■  ■  ■  ■|  ■  ■  ■ ◣ |  □  □  □  □|  □  □  □  □|     □     |
    V2_F3            |  ■  |  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■ ◣ |  □  □  □  □|     □     |
    V3_F3            |  ■  |  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■ ◣ |     □     |
    Traj+Text        |  ■  |  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■  ■|  ■  ■  ■  ■|     ◣     |
    """
    num_views = len(vision_start_end_ids_ranges)
    query_length = cache_position.shape[0]

    min_val = torch.finfo(dtype).min
    attention_mask = torch.full(
        (batch_size, 1, query_length, kv_length),
        min_val,
        dtype=dtype,
        device=device,
    )

    # Extended view ranges: include camera name tokens in the gap before each view.
    #   view 0: starts at VS[0] (camera name is in the system region before it)
    #   view i>0: starts at prev view's VE[last]+1 (= camera name start)
    # All views end at their own VE[last]+1.
    view_ext_ranges = []
    for view_idx in range(num_views):
        if view_idx == 0:
            ext_start = vision_start_end_ids_ranges[0][0][0]
        else:
            ext_start = vision_start_end_ids_ranges[view_idx - 1][-1][1]
        ext_end = vision_start_end_ids_ranges[view_idx][-1][1]
        view_ext_ranges.append((ext_start, ext_end))

    # System = everything before view 0's extended start (includes view 0's camera name)
    system_end = view_ext_ranges[0][0]

    # Build query-to-view mapping
    q_offset = 0
    query_view_mapping = []

    for view_idx in range(num_views):
        last_frame_start, last_frame_end = vision_start_end_ids_ranges[view_idx][-1]
        frame_length = last_frame_end - last_frame_start
        query_view_mapping.append((q_offset, q_offset + frame_length, view_idx, False, last_frame_start))
        q_offset += frame_length

    traj_start_kv, traj_end_kv = traj_and_text_ids_range
    traj_length = traj_end_kv - traj_start_kv
    query_view_mapping.append((q_offset, q_offset + traj_length, num_views, True, traj_start_kv))

    for q_start, q_end, view_idx, is_traj_text, frame_start_kv in query_view_mapping:
        region_length = q_end - q_start

        # 1. System prompt — always visible
        attention_mask[:, :, q_start:q_end, :system_end] = 0

        if not is_traj_text:
            # 2. Previous views — full extended range (camera + all frames + labels)
            for prev_idx in range(view_idx):
                prev_start, prev_end = view_ext_ranges[prev_idx]
                attention_mask[:, :, q_start:q_end, prev_start:prev_end] = 0

            # 3. Own view — from ext_start to just before F3's [VS]
            #    This includes: camera name (views 1+), all frame labels,
            #    frames 0-2 vision, AND "frame 3 " label.
            own_ext_start = view_ext_ranges[view_idx][0]
            attention_mask[:, :, q_start:q_end, own_ext_start:frame_start_kv] = 0

            # 4. Causal within F3 vision block
            q_indices = torch.arange(region_length, device=device).unsqueeze(1)
            kv_indices = torch.arange(region_length, device=device).unsqueeze(0)
            causal_mask = kv_indices <= q_indices

            attention_mask[:, :, q_start:q_end, frame_start_kv:frame_start_kv + region_length] = torch.where(
                causal_mask.unsqueeze(0).unsqueeze(0),
                torch.tensor(0.0, dtype=dtype, device=device),
                torch.tensor(min_val, dtype=dtype, device=device),
            )
        else:
            # Traj+Text — see all views (extended ranges)
            for v_idx in range(num_views):
                v_start, v_end = view_ext_ranges[v_idx]
                attention_mask[:, :, q_start:q_end, v_start:v_end] = 0

            # Causal within traj+text
            q_indices = torch.arange(region_length, device=device).unsqueeze(1)
            kv_indices = torch.arange(region_length, device=device).unsqueeze(0)
            causal_mask = kv_indices <= q_indices

            attention_mask[:, :, q_start:q_end, frame_start_kv:frame_start_kv + region_length] = torch.where(
                causal_mask.unsqueeze(0).unsqueeze(0),
                torch.tensor(0.0, dtype=dtype, device=device),
                torch.tensor(min_val, dtype=dtype, device=device),
            )

    # Mask padding positions
    if valid_length < kv_length:
        attention_mask[:, :, :, valid_length:] = min_val

    return attention_mask