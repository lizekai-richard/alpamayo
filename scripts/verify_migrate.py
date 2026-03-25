"""Verify that upgrade_window_to_v1p5 produces tokens identical to reference v1.5 data.

Usage:
    conda run -n flashdrive python scripts/verify_migrate.py \
        --v1-dir /mnt/moosefs-1/users/zekail/dumped_eval_data \
        --ref-dir /mnt/moosefs/users/zekail/dumped_eval_data_v1p5 \
        [--clip CLIP_ID]
"""

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import AutoTokenizer
from alpamayo_r1.helper import upgrade_window_to_v1p5


def verify_clip(clip_id, v1_dir, ref_dir, tok):
    v1_path = os.path.join(v1_dir, clip_id, "sliding_window_inputs.pt")
    ref_path = os.path.join(ref_dir, clip_id, "sliding_window_inputs.pt")

    if not os.path.isfile(v1_path):
        print(f"SKIP {clip_id}: v1 file not found")
        return None
    if not os.path.isfile(ref_path):
        print(f"SKIP {clip_id}: ref file not found")
        return None

    v1_windows = torch.load(v1_path, weights_only=False)
    ref_windows = torch.load(ref_path, weights_only=False)

    if len(v1_windows) != len(ref_windows):
        print(f"FAIL {clip_id}: window count mismatch v1={len(v1_windows)} ref={len(ref_windows)}")
        return False

    for wi in range(len(v1_windows)):
        upgraded = upgrade_window_to_v1p5(v1_windows[wi], tok)
        u = upgraded["tokenized_data"]["input_ids"][0]
        r = ref_windows[wi]["tokenized_data"]["input_ids"][0]

        if u.shape != r.shape:
            print(f"FAIL {clip_id} W{wi}: len mismatch u={u.shape[0]} r={r.shape[0]}")
            for j in range(min(len(u), len(r))):
                if u[j] != r[j]:
                    s, e = max(0, j - 3), min(min(len(u), len(r)), j + 8)
                    print(f"  1st diff@{j}: u={[tok.decode([t]) for t in u[s:e].tolist()]}")
                    print(f"  1st diff@{j}: r={[tok.decode([t]) for t in r[s:e].tolist()]}")
                    break
            return False

        mm = (u != r).nonzero(as_tuple=True)[0]
        if len(mm) > 0:
            print(f"FAIL {clip_id} W{wi}: {len(mm)} mismatches (len={len(u)})")
            for p in mm[:3]:
                p = p.item()
                print(f"  @{p}: u={tok.decode([u[p].item()])!r} r={tok.decode([r[p].item()])!r}")
            return False

    print(f"OK   {clip_id}: {len(v1_windows)} windows, len={len(u)}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-dir", default="/mnt/moosefs-1/users/zekail/dumped_eval_data")
    ap.add_argument("--ref-dir", default="/mnt/moosefs/users/zekail/dumped_eval_data_v1p5")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--clip", default=None, help="Specific clip ID (default: random)")
    ap.add_argument("--num-clips", type=int, default=1, help="Number of random clips to check")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    if args.clip:
        clip_ids = [args.clip]
    else:
        all_clips = sorted(os.listdir(args.ref_dir))
        clip_ids = random.sample(all_clips, min(args.num_clips, len(all_clips)))

    ok, fail, skip = 0, 0, 0
    for clip_id in clip_ids:
        result = verify_clip(clip_id, args.v1_dir, args.ref_dir, tok)
        if result is True:
            ok += 1
        elif result is False:
            fail += 1
        else:
            skip += 1

    print(f"\nTotal: OK={ok} FAIL={fail} SKIP={skip}")


if __name__ == "__main__":
    main()
