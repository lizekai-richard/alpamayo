"""
Verify that create_streaming_attention_mask_sdpa_training produces the
expected attention pattern for SFT training on a streaming window.

Layout (small example):
  4 views × 4 frames, 3 tokens per frame, 2 system tokens, 4 traj+text, 5 output

  KV positions:
    [Sys:0-1] [V0_F0:2-4 V0_F1:5-7 V0_F2:8-10 V0_F3:11-13]
              [V1_F0:14-16 V1_F1:17-19 V1_F2:20-22 V1_F3:23-25]
              [V2_F0:26-28 V2_F1:29-31 V2_F2:32-34 V2_F3:35-37]
              [V3_F0:38-40 V3_F1:41-43 V3_F2:44-46 V3_F3:47-49]
              [Traj+Text:50-53] [Output:54-58]

  Query = [V0_F3:3tok] [V1_F3:3tok] [V2_F3:3tok] [V3_F3:3tok]
          [Traj+Text:4tok] [Output:5tok]  = 21 tokens total

Expected rules:
  R1. System KV (0-1) visible to ALL query tokens.
  R2. V_i F3 query sees Sys + View 0..i (all frames), causal within own F3.
       V_i F3 query does NOT see views > i, Traj+Text, or Output.
  R3. Traj+Text query sees Sys + all views, causal within Traj+Text.
       Traj+Text does NOT see Output.
  R4. Output query sees ALL non-output KV, causal within Output.
"""

import torch
from streaming_masking_utils import (
    create_streaming_attention_mask_sdpa_training,
)

# ── dimensions ──
NUM_VIEWS = 4
NUM_FRAMES = 4
TOKENS_PER_FRAME = 3
SYS_TOKENS = 2
TRAJ_TEXT_TOKENS = 4
OUTPUT_TOKENS = 5

# ── build vision_start_end_ids_ranges ──
pos = SYS_TOKENS
vision_ranges: list[list[tuple[int, int]]] = []
for v in range(NUM_VIEWS):
    view_frames = []
    for f in range(NUM_FRAMES):
        view_frames.append((pos, pos + TOKENS_PER_FRAME))
        pos += TOKENS_PER_FRAME
    vision_ranges.append(view_frames)

traj_start = pos
traj_end = traj_start + TRAJ_TEXT_TOKENS
traj_and_text_ids_range = (traj_start, traj_end)

output_start = traj_end
output_end = output_start + OUTPUT_TOKENS
output_ids_range = (output_start, output_end)

kv_length = output_end

# ── build cache_position (query positions in the full sequence) ──
# Query = [V0_F3] [V1_F3] [V2_F3] [V3_F3] [Traj+Text] [Output]
query_positions = []
for v in range(NUM_VIEWS):
    f3_start, f3_end = vision_ranges[v][-1]  # last frame
    query_positions.extend(range(f3_start, f3_end))
query_positions.extend(range(traj_start, traj_end))
query_positions.extend(range(output_start, output_end))
cache_position = torch.tensor(query_positions, device="cpu")
query_length = len(query_positions)

# ── create mask ──
mask = create_streaming_attention_mask_sdpa_training(
    batch_size=1,
    cache_position=cache_position,
    kv_length=kv_length,
    vision_start_end_ids_ranges=vision_ranges,
    traj_and_text_ids_range=traj_and_text_ids_range,
    output_ids_range=output_ids_range,
    device=torch.device("cpu"),
    dtype=torch.float32,
)

# Convert to boolean: True = can attend
can_attend = (mask[0, 0] == 0.0)  # [query_length, kv_length]

# ── helper: query index ranges ──
q_off = 0
q_view_ranges = []  # (q_start, q_end) for each view's F3
for v in range(NUM_VIEWS):
    q_view_ranges.append((q_off, q_off + TOKENS_PER_FRAME))
    q_off += TOKENS_PER_FRAME
q_traj_range = (q_off, q_off + TRAJ_TEXT_TOKENS)
q_off += TRAJ_TEXT_TOKENS
q_output_range = (q_off, q_off + OUTPUT_TOKENS)

# ── helper: KV region classification ──
def kv_region(k):
    """Return (region_type, view_idx, frame_idx) for a KV position."""
    if k < SYS_TOKENS:
        return ("sys", -1, -1)
    for v in range(NUM_VIEWS):
        for f in range(NUM_FRAMES):
            s, e = vision_ranges[v][f]
            if s <= k < e:
                return ("vision", v, f)
    if traj_start <= k < traj_end:
        return ("traj_text", -1, -1)
    if output_start <= k < output_end:
        return ("output", -1, -1)
    return ("unknown", -1, -1)


# ===========================================================================
# Rule checks
# ===========================================================================
errors = []


def check(condition, msg):
    if not condition:
        errors.append(msg)


# ── R1: System KV visible to ALL query tokens ──
for q in range(query_length):
    for k in range(SYS_TOKENS):
        check(can_attend[q, k].item(), f"R1 fail: q={q} cannot see sys kv={k}")

# ── R2: V_i F3 query attention rules ──
for v in range(NUM_VIEWS):
    qs, qe = q_view_ranges[v]
    for q in range(qs, qe):
        q_local = q - qs  # position within the F3 block
        for k in range(kv_length):
            region, kv_v, kv_f = kv_region(k)
            attends = can_attend[q, k].item()

            if region == "sys":
                check(attends, f"R2 fail: V{v}_F3 q={q} cannot see sys kv={k}")

            elif region == "vision":
                if kv_v < v:
                    # Earlier view: fully visible
                    check(attends, f"R2 fail: V{v}_F3 q={q} cannot see V{kv_v}_F{kv_f} kv={k}")
                elif kv_v == v:
                    if kv_f < NUM_FRAMES - 1:
                        # Same view, earlier frame: fully visible
                        check(attends, f"R2 fail: V{v}_F3 q={q} cannot see V{kv_v}_F{kv_f} kv={k}")
                    else:
                        # Same view, same frame (F3): causal
                        k_local = k - vision_ranges[v][-1][0]
                        if k_local <= q_local:
                            check(attends, f"R2 causal fail: V{v}_F3 q={q} cannot see kv={k} (k_local={k_local} <= q_local={q_local})")
                        else:
                            check(not attends, f"R2 causal fail: V{v}_F3 q={q} CAN see future kv={k} (k_local={k_local} > q_local={q_local})")
                else:
                    # Later view: must NOT see
                    check(not attends, f"R2 fail: V{v}_F3 q={q} CAN see later V{kv_v}_F{kv_f} kv={k}")

            elif region == "traj_text":
                check(not attends, f"R2 fail: V{v}_F3 q={q} CAN see traj+text kv={k}")

            elif region == "output":
                check(not attends, f"R2 fail: V{v}_F3 q={q} CAN see output kv={k}")

# ── R3: Traj+Text query attention rules ──
qs, qe = q_traj_range
for q in range(qs, qe):
    q_local = q - qs
    for k in range(kv_length):
        region, kv_v, kv_f = kv_region(k)
        attends = can_attend[q, k].item()

        if region == "sys":
            check(attends, f"R3 fail: traj q={q} cannot see sys kv={k}")

        elif region == "vision":
            check(attends, f"R3 fail: traj q={q} cannot see V{kv_v}_F{kv_f} kv={k}")

        elif region == "traj_text":
            k_local = k - traj_start
            if k_local <= q_local:
                check(attends, f"R3 causal fail: traj q={q} cannot see kv={k}")
            else:
                check(not attends, f"R3 causal fail: traj q={q} CAN see future kv={k}")

        elif region == "output":
            check(not attends, f"R3 fail: traj q={q} CAN see output kv={k}")

# ── R4: Output query attention rules ──
qs, qe = q_output_range
for q in range(qs, qe):
    q_abs = output_start + (q - qs)  # absolute position in KV
    for k in range(kv_length):
        region, kv_v, kv_f = kv_region(k)
        attends = can_attend[q, k].item()

        if region == "output":
            # Causal within output by absolute KV position
            if k <= q_abs:
                check(attends, f"R4 causal fail: output q={q} (abs={q_abs}) cannot see output kv={k}")
            else:
                check(not attends, f"R4 causal fail: output q={q} (abs={q_abs}) CAN see future output kv={k}")
        else:
            # All non-output KV fully visible
            check(attends, f"R4 fail: output q={q} cannot see non-output {region} kv={k}")


# ===========================================================================
# Results
# ===========================================================================
if errors:
    print(f"FAILED: {len(errors)} errors")
    for e in errors[:20]:
        print(f"  {e}")
    if len(errors) > 20:
        print(f"  ... and {len(errors) - 20} more")
else:
    print("ALL CHECKS PASSED")


# ===========================================================================
# ASCII visualization
# ===========================================================================
print("\n" + "=" * 80)
print("ATTENTION MASK VISUALIZATION  (■ = attend, □ = masked)")
print("=" * 80)

# Column labels
kv_labels = []
for k in range(kv_length):
    region, v, f = kv_region(k)
    if region == "sys":
        kv_labels.append("S")
    elif region == "vision":
        kv_labels.append(f"{v}{f}")
    elif region == "traj_text":
        kv_labels.append("T")
    elif region == "output":
        kv_labels.append("O")
    else:
        kv_labels.append("?")

# Row labels
row_labels = []
for v in range(NUM_VIEWS):
    for t in range(TOKENS_PER_FRAME):
        row_labels.append(f"V{v}F3.{t}")
for t in range(TRAJ_TEXT_TOKENS):
    row_labels.append(f"Trj.{t}")
for t in range(OUTPUT_TOKENS):
    row_labels.append(f"Out.{t}")

# Header
header_line = "            KV: "
for lab in kv_labels:
    header_line += lab.center(3)
print(header_line)
print("         " + "-" * (len(header_line) - 9))

# Rows
for q in range(query_length):
    row = f"  {row_labels[q]:>8s} | "
    for k in range(kv_length):
        row += (" ■ " if can_attend[q, k].item() else " □ ")
    print(row)
