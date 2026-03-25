#!/usr/bin/env python3
"""Check all clips: seq_len == 3086 AND merged " frame" token (4034) present."""

import os
import torch

DATA_DIR = "/mnt/moosefs-1/users/zekail/dumped_inputs_v1_5"
EXPECTED_SEQ_LEN = 3086
MERGED_FRAME_TOKEN = 4034   # " frame" (correct)
VS_TOKEN_ID = 151652         # <|vision_start|>

clips = sorted(os.listdir(DATA_DIR))
total = len(clips)
ok = 0
bad_clips = []

for i, clip_id in enumerate(clips):
    if i % 500 == 0:
        print(f"Progress: {i}/{total}...", flush=True)

    path = os.path.join(DATA_DIR, clip_id, "sliding_window_inputs.pt")
    if not os.path.isfile(path):
        bad_clips.append((clip_id, "MISSING"))
        continue
    try:
        windows = torch.load(path, map_location="cpu", weights_only=False)
        ids = windows[0]["tokenized_data"]["input_ids"][0]
        seq_len = len(ids)

        vs_pos = (ids == VS_TOKEN_ID).nonzero(as_tuple=True)[0]
        f0_label = ids[vs_pos[0]-4:vs_pos[0]].tolist()
        has_merged = MERGED_FRAME_TOKEN in f0_label

        issues = []
        if seq_len != EXPECTED_SEQ_LEN:
            issues.append(f"seq_len={seq_len}")
        if not has_merged:
            issues.append(f"no_merged_token(f0={f0_label})")

        if issues:
            msg = ", ".join(issues)
            bad_clips.append((clip_id, msg))
            print(f"  BAD [{i}/{total}] {clip_id}  {msg}", flush=True)
        else:
            ok += 1
    except Exception as e:
        msg = f"ERROR: {str(e)[:80]}"
        bad_clips.append((clip_id, msg))
        print(f"  BAD [{i}/{total}] {clip_id}  {msg}", flush=True)

print(f"\nTotal: {total}, OK: {ok}, Bad: {len(bad_clips)}")
print(f"Pass rate: {100*ok/max(total,1):.1f}%")

if bad_clips:
    from collections import Counter
    for reason, count in Counter(r for _, r in bad_clips).most_common():
        print(f"  {reason}: {count}")
    with open("/tmp/bad_v1p5_clips.txt", "w") as f:
        for cid, reason in bad_clips:
            f.write(f"{cid}\t{reason}\n")
    print(f"Saved to /tmp/bad_v1p5_clips.txt")
else:
    print("All clips correct!")
