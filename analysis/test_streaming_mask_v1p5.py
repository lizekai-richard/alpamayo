"""
Verify create_streaming_attention_mask_sdpa_v1p5 for Setting B (vision_only + no_labels).

v1.5 Layout (small example):
  4 views × 4 frames, 3 vision tokens per frame, 2 system tokens,
  2 camera label tokens per view, 1 frame label token per frame, 4 traj+text tokens.

  KV positions:
    [Sys:0-1]
    [cam0:2-3][fl0:4][V0_F0:5-7][fl1:8][V0_F1:9-11][fl2:12][V0_F2:13-15][fl3:16][V0_F3:17-19]
    [cam1:20-21][fl0:22][V1_F0:23-25][fl1:26][V1_F1:27-29][fl2:30][V1_F2:31-33][fl3:34][V1_F3:35-37]
    [cam2:38-39][fl0:40][V2_F0:41-43][fl1:44][V2_F1:45-47][fl2:48][V2_F2:49-51][fl3:52][V2_F3:53-55]
    [cam3:56-57][fl0:58][V3_F0:59-61][fl1:62][V3_F1:63-65][fl2:66][V3_F2:67-69][fl3:70][V3_F3:71-73]
    [Traj+Text:74-77]

  Query (Setting B: no frame labels in query, only vision tokens):
    [V0_F3:17-19] [V1_F3:35-37] [V2_F3:53-55] [V3_F3:71-73] [Traj+Text:74-77]
    = 4*3 + 4 = 16 query tokens

Expected rules for V_i F3 query:
  R1. System tokens (0-1) visible to ALL.
  R2. Camera labels + frame labels + vision of View_0..View_{i-1} fully visible.
      (For view 0 this includes cam0/fl0 which are in the system region [0, VS[0]).)
  R3. Own view's camera label + all frame labels (fl0..fl3) + frames F0-F2 visible.
  R4. Own F3 vision: causal.
  R5. Later views, Traj+Text: masked.

For Traj+Text query:
  R6. System + all views (extended) visible. Causal within Traj+Text.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "alpamayo_r1", "utils", "streaming"))

import torch
from streaming_masking_utils import create_streaming_attention_mask_sdpa_v1p5

# ── dimensions ──
NUM_VIEWS = 4
NUM_FRAMES = 4
VISION_TOKENS = 3      # [VS ... VE] per frame
SYS_TOKENS = 2
CAM_LABEL_TOKENS = 2   # "Front X camera: " per view
FRAME_LABEL_TOKENS = 1 # "frame N " per frame
TRAJ_TEXT_TOKENS = 4

# ── build layout ──
pos = SYS_TOKENS

# Track all positions for classification
cam_label_positions = {}    # view_idx -> (start, end)
frame_label_positions = {}  # (view_idx, frame_idx) -> (start, end)
vision_ranges: list[list[tuple[int, int]]] = []

for v in range(NUM_VIEWS):
    # Camera label
    cam_start = pos
    pos += CAM_LABEL_TOKENS
    cam_label_positions[v] = (cam_start, pos)

    view_frames = []
    for f in range(NUM_FRAMES):
        # Frame label
        fl_start = pos
        pos += FRAME_LABEL_TOKENS
        frame_label_positions[(v, f)] = (fl_start, pos)

        # Vision tokens
        vs_start = pos
        pos += VISION_TOKENS
        view_frames.append((vs_start, pos))

    vision_ranges.append(view_frames)

traj_start = pos
traj_end = traj_start + TRAJ_TEXT_TOKENS
traj_and_text_ids_range = (traj_start, traj_end)

kv_length = traj_end  # no output tokens in this test
valid_length = kv_length

print(f"KV length: {kv_length}")
print(f"Vision ranges: {vision_ranges}")
print(f"Cam label positions: {cam_label_positions}")
print(f"Frame label positions: {frame_label_positions}")
print(f"Traj+Text: {traj_and_text_ids_range}")

# ── build cache_position (Setting B: only vision tokens in query) ──
query_positions = []
for v in range(NUM_VIEWS):
    f3_start, f3_end = vision_ranges[v][-1]
    query_positions.extend(range(f3_start, f3_end))
query_positions.extend(range(traj_start, traj_end))
cache_position = torch.tensor(query_positions, device="cpu")
query_length = len(query_positions)

print(f"Query positions: {query_positions}")
print(f"Query length: {query_length}")

# ── create mask ──
mask = create_streaming_attention_mask_sdpa_v1p5(
    batch_size=1,
    cache_position=cache_position,
    kv_length=kv_length,
    vision_start_end_ids_ranges=vision_ranges,
    traj_and_text_ids_range=traj_and_text_ids_range,
    valid_length=valid_length,
    device=torch.device("cpu"),
    dtype=torch.float32,
)

can_attend = (mask[0, 0] == 0.0)  # [query_length, kv_length]

# ── query index ranges ──
q_off = 0
q_view_ranges = []
for v in range(NUM_VIEWS):
    q_view_ranges.append((q_off, q_off + VISION_TOKENS))
    q_off += VISION_TOKENS
q_traj_range = (q_off, q_off + TRAJ_TEXT_TOKENS)

# ── KV region classification ──
def kv_region(k):
    """Return (region_type, view_idx, frame_idx) for a KV position."""
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
            vs, ve = vision_ranges[v][f]
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


# ── R1: System KV visible to ALL query tokens ──
for q in range(query_length):
    for k in range(SYS_TOKENS):
        check(can_attend[q, k].item(), f"R1 fail: q={q} cannot see sys kv={k}")

# ── R2-R5: V_i F3 query attention rules ──
for v in range(NUM_VIEWS):
    qs, qe = q_view_ranges[v]
    for q in range(qs, qe):
        q_local = q - qs
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
                    # Previous view: all frame labels visible
                    check(attends, f"V{v}F3 q={q}: cannot see V{kv_v} frame_label{kv_f} kv={k}")
                elif kv_v == v:
                    # Own view: all frame labels visible (including frame 3 label)
                    check(attends, f"V{v}F3 q={q}: cannot see own frame_label{kv_f} kv={k}")
                else:
                    check(not attends, f"V{v}F3 q={q}: CAN see later V{kv_v} frame_label{kv_f} kv={k}")

            elif region == "vision":
                if kv_v < v:
                    check(attends, f"V{v}F3 q={q}: cannot see V{kv_v}_F{kv_f} kv={k}")
                elif kv_v == v:
                    if kv_f < NUM_FRAMES - 1:
                        check(attends, f"V{v}F3 q={q}: cannot see own V{kv_v}_F{kv_f} kv={k}")
                    else:
                        # F3: causal
                        k_local = k - vision_ranges[v][-1][0]
                        if k_local <= q_local:
                            check(attends, f"V{v}F3 causal fail: q={q} cannot see kv={k}")
                        else:
                            check(not attends, f"V{v}F3 causal fail: q={q} CAN see future kv={k}")
                else:
                    check(not attends, f"V{v}F3 q={q}: CAN see later V{kv_v}_F{kv_f} kv={k}")

            elif region == "traj_text":
                check(not attends, f"V{v}F3 q={q}: CAN see traj+text kv={k}")

# ── R6: Traj+Text query ──
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

# Column labels
kv_labels = []
for k in range(kv_length):
    region, v, f = kv_region(k)
    if region == "sys":
        kv_labels.append("S")
    elif region == "cam_label":
        kv_labels.append(f"c{v}")
    elif region == "frame_label":
        kv_labels.append(f"l{f}")
    elif region == "vision":
        kv_labels.append(f"{v}{f}")
    elif region == "traj_text":
        kv_labels.append("T")
    else:
        kv_labels.append("?")

# Row labels
row_labels = []
for v in range(NUM_VIEWS):
    for t in range(VISION_TOKENS):
        row_labels.append(f"V{v}F3.{t}")
for t in range(TRAJ_TEXT_TOKENS):
    row_labels.append(f"Trj.{t}")

# Header
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
