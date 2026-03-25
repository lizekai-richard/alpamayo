"""
Verify create_streaming_attention_mask_sdpa_v1p5 for Setting A (block + keep_labels).

Same v1.5 layout as Setting B test, but query includes "frame 3 " label tokens
(expanded F3 range). The function should work identically since frame_start_kv
adapts via vision_start_end_ids_ranges[-1][0].

v1.5 Layout:
  [Sys:0-1]
  [cam0:2-3][fl0:4][V0_F0:5-7][fl1:8][V0_F1:9-11][fl2:12][V0_F2:13-15][fl3:16][V0_F3:17-19]
  [cam1:20-21][fl0:22][V1_F0:23-25]...
  ...
  [Traj+Text:74-77]

Setting A: vision_ranges[-1] is EXPANDED to (VE[2]+1, VE[3]+1) = includes fl3 + F3.
  Query includes fl3 tokens.

Expected rules for V_i F3 query (fl3 + vision as a unit):
  R1. System tokens visible to ALL.
  R2. Camera labels of View_0..View_i visible.
  R3. Previous views fully. Own view: camera + F0-F2 + their labels visible.
      (frame 3 label is INSIDE the causal block, not separately visible.)
  R4. [fl3 + F3 vision] is causal as a unit.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "alpamayo_r1", "utils", "streaming"))

import torch
from streaming_masking_utils import create_streaming_attention_mask_sdpa_v1p5

# ── dimensions ──
NUM_VIEWS = 4
NUM_FRAMES = 4
VISION_TOKENS = 3
SYS_TOKENS = 2
CAM_LABEL_TOKENS = 2
FRAME_LABEL_TOKENS = 1
TRAJ_TEXT_TOKENS = 4

# ── build layout (same as Setting B test) ──
pos = SYS_TOKENS
cam_label_positions = {}
frame_label_positions = {}
vision_ranges_raw: list[list[tuple[int, int]]] = []  # non-expanded

for v in range(NUM_VIEWS):
    cam_start = pos
    pos += CAM_LABEL_TOKENS
    cam_label_positions[v] = (cam_start, pos)

    view_frames = []
    for f in range(NUM_FRAMES):
        fl_start = pos
        pos += FRAME_LABEL_TOKENS
        frame_label_positions[(v, f)] = (fl_start, pos)

        vs_start = pos
        pos += VISION_TOKENS
        view_frames.append((vs_start, pos))

    vision_ranges_raw.append(view_frames)

traj_start = pos
traj_end = traj_start + TRAJ_TEXT_TOKENS
traj_and_text_ids_range = (traj_start, traj_end)
kv_length = traj_end

# ── EXPAND last frame range for Setting A ──
# Expand [-1] to (sec_last_end, last_end) = (VE[2]+1, VE[3]+1)
# This includes the frame 3 label tokens.
vision_ranges: list[list[tuple[int, int]]] = []
for v in range(NUM_VIEWS):
    frames = list(vision_ranges_raw[v])
    sec_last_end = frames[-2][1]  # VE[2]+1
    last_end = frames[-1][1]       # VE[3]+1
    frames[-1] = (sec_last_end, last_end)  # expanded: includes fl3 + F3
    vision_ranges.append(frames)

print(f"KV length: {kv_length}")
print(f"Vision ranges (expanded): {vision_ranges}")

# ── build cache_position (Setting A: includes frame label in query) ──
query_positions = []
for v in range(NUM_VIEWS):
    f3_start, f3_end = vision_ranges[v][-1]  # expanded range
    query_positions.extend(range(f3_start, f3_end))
query_positions.extend(range(traj_start, traj_end))
cache_position = torch.tensor(query_positions, device="cpu")
query_length = len(query_positions)

# F3 query token count per view = FRAME_LABEL_TOKENS + VISION_TOKENS
F3_QUERY_TOKENS = FRAME_LABEL_TOKENS + VISION_TOKENS

print(f"Query positions: {query_positions}")
print(f"Query length: {query_length} (F3 block = {F3_QUERY_TOKENS} tokens/view)")

# ── create mask ──
mask = create_streaming_attention_mask_sdpa_v1p5(
    batch_size=1,
    cache_position=cache_position,
    kv_length=kv_length,
    vision_start_end_ids_ranges=vision_ranges,
    traj_and_text_ids_range=traj_and_text_ids_range,
    valid_length=kv_length,
    device=torch.device("cpu"),
    dtype=torch.float32,
)

can_attend = (mask[0, 0] == 0.0)

# ── query index ranges ──
q_off = 0
q_view_ranges = []
for v in range(NUM_VIEWS):
    q_view_ranges.append((q_off, q_off + F3_QUERY_TOKENS))
    q_off += F3_QUERY_TOKENS
q_traj_range = (q_off, q_off + TRAJ_TEXT_TOKENS)

# ── KV region classification ──
def kv_region(k):
    if k < SYS_TOKENS:
        return ("sys", -1, -1)
    for v in range(NUM_VIEWS):
        cs, ce = cam_label_positions[v]
        if cs <= k < ce:
            return ("cam_label", v, -1)
        for f in range(NUM_FRAMES):
            fls, fle = frame_label_positions[(v, f)]
            if fls <= k < fle:
                return ("frame_label", v, f)
            vs, ve = vision_ranges_raw[v][f]
            if vs <= k < ve:
                return ("vision", v, f)
    if traj_start <= k < traj_end:
        return ("traj_text", -1, -1)
    return ("unknown", -1, -1)


# ===========================================================================
# Rule checks
# ===========================================================================
errors = []

def check(condition, msg):
    if not condition:
        errors.append(msg)

# R1: System visible to all
for q in range(query_length):
    for k in range(SYS_TOKENS):
        check(can_attend[q, k].item(), f"R1 fail: q={q} cannot see sys kv={k}")

# R2-R4: V_i F3 query (fl3 + vision as causal unit)
for v in range(NUM_VIEWS):
    qs, qe = q_view_ranges[v]
    expanded_start = vision_ranges[v][-1][0]  # = VE[2]+1 = start of fl3

    for q in range(qs, qe):
        q_local = q - qs  # position within the [fl3 + F3] block
        for k in range(kv_length):
            region, kv_v, kv_f = kv_region(k)
            attends = can_attend[q, k].item()

            if region == "sys":
                check(attends, f"V{v}F3 q={q}: cannot see sys kv={k}")

            elif region == "cam_label":
                if kv_v <= v:
                    check(attends, f"V{v}F3 q={q}: cannot see cam{kv_v} label kv={k}")
                else:
                    check(not attends, f"V{v}F3 q={q}: CAN see later cam{kv_v} label kv={k}")

            elif region == "frame_label":
                if kv_v < v:
                    # Previous view: all visible
                    check(attends, f"V{v}F3 q={q}: cannot see V{kv_v} fl{kv_f} kv={k}")
                elif kv_v == v:
                    if kv_f < NUM_FRAMES - 1:
                        # Own view, earlier frame label: visible
                        check(attends, f"V{v}F3 q={q}: cannot see own fl{kv_f} kv={k}")
                    else:
                        # frame 3 label: INSIDE the causal block
                        k_local = k - expanded_start
                        if k_local <= q_local:
                            check(attends, f"V{v}F3 causal fail: q={q} cannot see fl3 kv={k}")
                        else:
                            check(not attends, f"V{v}F3 causal fail: q={q} CAN see future fl3 kv={k}")
                else:
                    check(not attends, f"V{v}F3 q={q}: CAN see later V{kv_v} fl{kv_f} kv={k}")

            elif region == "vision":
                if kv_v < v:
                    check(attends, f"V{v}F3 q={q}: cannot see V{kv_v}_F{kv_f} kv={k}")
                elif kv_v == v:
                    if kv_f < NUM_FRAMES - 1:
                        check(attends, f"V{v}F3 q={q}: cannot see own F{kv_f} kv={k}")
                    else:
                        # F3 vision: inside the causal block
                        k_local = k - expanded_start
                        if k_local <= q_local:
                            check(attends, f"V{v}F3 causal fail: q={q} cannot see F3 kv={k}")
                        else:
                            check(not attends, f"V{v}F3 causal fail: q={q} CAN see future F3 kv={k}")
                else:
                    check(not attends, f"V{v}F3 q={q}: CAN see later V{kv_v}_F{kv_f} kv={k}")

            elif region == "traj_text":
                check(not attends, f"V{v}F3 q={q}: CAN see traj+text kv={k}")

# R6: Traj+Text
qs, qe = q_traj_range
for q in range(qs, qe):
    q_local = q - qs
    for k in range(kv_length):
        region, kv_v, kv_f = kv_region(k)
        attends = can_attend[q, k].item()

        if region == "sys":
            check(attends, f"Traj q={q}: cannot see sys kv={k}")
        elif region in ("cam_label", "frame_label", "vision"):
            check(attends, f"Traj q={q}: cannot see {region} V{kv_v} kv={k}")
        elif region == "traj_text":
            k_local = k - traj_start
            if k_local <= q_local:
                check(attends, f"Traj causal fail: q={q} cannot see kv={k}")
            else:
                check(not attends, f"Traj causal fail: q={q} CAN see future kv={k}")


# ===========================================================================
# Results
# ===========================================================================
if errors:
    print(f"\nFAILED: {len(errors)} errors")
    for e in errors[:30]:
        print(f"  {e}")
    if len(errors) > 30:
        print(f"  ... and {len(errors) - 30} more")
else:
    print("\nALL CHECKS PASSED")


# ===========================================================================
# ASCII visualization
# ===========================================================================
print("\n" + "=" * 80)
print("ATTENTION MASK VISUALIZATION  (■ = attend, □ = masked)")
print("=" * 80)

kv_labels = []
for k in range(kv_length):
    region, v, f = kv_region(k)
    if region == "sys":       kv_labels.append("S")
    elif region == "cam_label":  kv_labels.append(f"c{v}")
    elif region == "frame_label": kv_labels.append(f"l{f}")
    elif region == "vision":  kv_labels.append(f"{v}{f}")
    elif region == "traj_text": kv_labels.append("T")
    else: kv_labels.append("?")

row_labels = []
for v in range(NUM_VIEWS):
    for t in range(FRAME_LABEL_TOKENS):
        row_labels.append(f"V{v}fl3.{t}")
    for t in range(VISION_TOKENS):
        row_labels.append(f"V{v}F3.{t}")
for t in range(TRAJ_TEXT_TOKENS):
    row_labels.append(f"Trj.{t}")

header_line = "            KV: "
for lab in kv_labels:
    header_line += lab.center(3)
print(header_line)
print("         " + "-" * (len(header_line) - 9))

for q in range(query_length):
    row = f"  {row_labels[q]:>8s} | "
    for k in range(kv_length):
        row += (" ■ " if can_attend[q, k].item() else " □ ")
    print(row)
