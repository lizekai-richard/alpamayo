"""Test _get_pruned_rope_index with real model + processor.

Loads AlpamayoR1 and uses the real processor to build realistic input_ids /
image_grid_thw (matching actual inference inputs).

Key checks for every rope_mode:
  [A] (direct only)  pruned positions == original positions filtered by keep_mask
  [B] DECODE CONSISTENCY: seq_len + rope_deltas == max_pos + 1
       (this is what drives decode step; seq_len is the pruned sequence length,
        same as `cur_pos` at the start of the decode loop in the compile model)
  [C] No position gaps between consecutive text / image segments
  [D] keep_mask token count is correct

Usage:
    python test_rope_pruning.py --model_path ./Alpamayo-R1-10B
"""

import argparse
import sys

import torch

from alpamayo_r1.models.alpamayo_r1_compile import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_inputs(processor, clip_id: str, t0_us: int = 1_700_000, num_frames: int = 4):
    """Load a real clip from the dataset and tokenize it exactly as inference does."""
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us, num_frames=num_frames)
    frames = data["image_frames"].flatten(0, 1)  # (N_cameras * num_frames, 3, H, W)
    messages = helper.create_message(frames)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    print(f"  Loaded clip={clip_id}, t0_us={t0_us}, frames shape: {frames.shape}")
    return inputs


# ── Fixed contiguous mode (local implementation) ──────────────────────────────

def _get_pruned_rope_contiguous_v2(model, input_ids, image_grid_thw, token_indices):
    """Fixed contiguous: per-row w ranks (0,1,...,count-1), no cross-row gaps."""
    spatial_merge_size = model.vlm.config.vision_config.spatial_merge_size
    image_token_id = model.vlm.config.image_token_id
    vision_start_token_id = model.vlm.config.vision_start_token_id

    batch_size = input_ids.shape[0]
    K = token_indices.shape[1]

    keep_mask = model._get_keep_mask(input_ids, image_grid_thw, token_indices)

    all_pos = []
    all_deltas = []
    image_index = 0

    for i in range(batch_size):
        ids = input_ids[i]
        input_tokens = ids.tolist()

        vision_start_idx = torch.argwhere(ids == vision_start_token_id).squeeze(1)
        image_nums = (ids[vision_start_idx + 1] == image_token_id).sum().item()

        llm_pos_ids_list: list = []
        st = 0

        for _ in range(image_nums):
            ed = input_tokens.index(image_token_id, st)
            t, h, w = image_grid_thw[image_index]
            llm_grid_w = w.item() // spatial_merge_size
            original_num_tokens = t.item() * (h.item() // spatial_merge_size) * llm_grid_w

            # Text before image
            text_len = ed - st
            if text_len > 0:
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
                )

            # Image: per-row contiguous w
            kept = token_indices[image_index]
            h_orig = kept // llm_grid_w
            _, h_inv = torch.unique(h_orig, sorted=True, return_inverse=True)
            h_new = h_inv  # 0..new_h-1
            w_new = torch.empty(K, dtype=torch.long, device=kept.device)
            ptr = 0
            for hi in range(int(h_inv.max().item()) + 1):
                count = int((h_inv == hi).sum().item())
                w_new[ptr : ptr + count] = torch.arange(count, device=kept.device)
                ptr += count
            t_new = torch.zeros(K, dtype=torch.long, device=kept.device)

            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(torch.stack([t_new, h_new, w_new]) + st_idx)

            st = ed + original_num_tokens
            image_index += 1

        # Trailing text
        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        all_pos.append(llm_positions)
        new_seq_len = keep_mask[i].sum().item()
        all_deltas.append(llm_positions.max() + 1 - new_seq_len)

    position_ids = torch.stack(all_pos, dim=1).to(input_ids.device)
    rope_deltas = torch.tensor(all_deltas, dtype=torch.long, device=input_ids.device).unsqueeze(1)
    return position_ids, rope_deltas, keep_mask


# ── Core verification ─────────────────────────────────────────────────────────

def verify_rope(model, input_ids, image_grid_thw, sparsity_ratio: float = 0.5):
    spatial_merge_size = model.vlm.config.vision_config.spatial_merge_size
    IMAGE_TOKEN_ID = model.vlm.config.image_token_id
    num_images = image_grid_thw.shape[0]
    orig_seq = input_ids.shape[1]

    tokens_per_image = (
        image_grid_thw[:, 0]
        * (image_grid_thw[:, 1] // spatial_merge_size)
        * (image_grid_thw[:, 2] // spatial_merge_size)
    ).tolist()
    N = int(tokens_per_image[0])
    if not all(int(t) == N for t in tokens_per_image):
        print(f"  WARNING: images have different token counts {[int(t) for t in tokens_per_image]}, using min")
        N = min(int(t) for t in tokens_per_image)
    K = int(N * (1 - sparsity_ratio))

    print(f"  seq_len={orig_seq}, num_images={num_images}, N={N}/image, K={K} (sparsity={sparsity_ratio})")
    print(f"  image_grid_thw[0]: {image_grid_thw[0].tolist()}")

    # Create synthetic token_indices (uniform random subsampling)
    torch.manual_seed(42)
    token_indices = torch.stack([
        torch.randperm(N)[:K].sort().values for _ in range(num_images)
    ])

    # Ground-truth positions from the actual model
    with torch.no_grad():
        orig_pos, orig_delta = model.vlm.model.get_rope_index(
            input_ids, image_grid_thw, None, None,
        )
    orig_max = orig_pos.max().item()
    orig_decode_pos = orig_seq + orig_delta[0, 0].item()
    print(f"  Ground truth: max_pos={orig_max}, rope_delta={orig_delta[0,0].item()}, "
          f"decode_pos={orig_decode_pos} {'✓' if orig_decode_pos == orig_max + 1 else '✗'}")

    all_ok = True
    for rope_mode in ["direct", "contiguous", "reshape", "contiguous_v2"]:
        with torch.no_grad():
            if rope_mode == "contiguous_v2":
                pruned_pos, pruned_delta, keep_mask = _get_pruned_rope_contiguous_v2(
                    model, input_ids, image_grid_thw, token_indices,
                )
            else:
                pruned_pos, pruned_delta, keep_mask = model._get_pruned_rope_index(
                    input_ids, image_grid_thw, token_indices, rope_mode=rope_mode,
                )

        new_seq = keep_mask[0].sum().item()
        pruned_max = pruned_pos.max().item()
        # This is exactly what the decode loop computes for the first generated token:
        decode_pos = new_seq + pruned_delta[0, 0].item()

        print(f"\n  [{rope_mode}]  new_seq={new_seq}, max_pos={pruned_max}, "
              f"rope_delta={pruned_delta[0,0].item()}, decode_pos={decode_pos}")
        errors = []

        # [A] Direct mode: positions must match original filtered by keep_mask
        if rope_mode == "direct":
            orig_filtered = orig_pos[:, 0, keep_mask[0]]
            if torch.equal(pruned_pos[:, 0, :], orig_filtered):
                print("    [A] PASS  positions == orig filtered by keep_mask")
            else:
                n_diff = (pruned_pos[:, 0, :] != orig_filtered).sum().item()
                errors.append(f"[A] FAIL  {n_diff} position mismatches")

        # [B] DECODE CONSISTENCY — the critical check
        # decode_pos = new_seq + rope_delta must equal pruned_max + 1
        expected_decode = pruned_max + 1
        if decode_pos == expected_decode:
            print(f"    [B] PASS  decode_pos = {decode_pos} == max_pos+1 = {expected_decode}")
        else:
            errors.append(
                f"[B] FAIL  decode_pos={decode_pos} != max_pos+1={expected_decode}  "
                f"(off by {decode_pos - expected_decode}, i.e. {num_images*(N-K)} pruned tokens)"
            )

        # [C] No position gaps between consecutive segments
        tokens = input_ids[0].tolist()
        kept_indices = keep_mask[0].nonzero(as_tuple=True)[0].tolist()
        segments, cur_type, cur_start = [], None, 0
        for pos_idx, orig_idx in enumerate(kept_indices):
            seg_type = "img" if tokens[orig_idx] == IMAGE_TOKEN_ID else "txt"
            if seg_type != cur_type:
                if cur_type is not None:
                    segments.append((cur_type, cur_start, pos_idx))
                cur_type, cur_start = seg_type, pos_idx
        if cur_type is not None:
            segments.append((cur_type, cur_start, len(kept_indices)))

        gap_errors = []
        for si in range(len(segments) - 1):
            seg_max = max(pruned_pos[d, 0, segments[si][1]:segments[si][2]].max().item() for d in range(3))
            next_min = min(pruned_pos[d, 0, segments[si+1][1]].item() for d in range(3))
            if next_min - seg_max != 1:
                gap_errors.append(f"seg {si}({segments[si][0]})→{si+1}({segments[si+1][0]}): gap={next_min - seg_max}")
        if gap_errors:
            errors.extend([f"[C] Gap: {e}" for e in gap_errors])
        else:
            print(f"    [C] PASS  no gaps ({len(segments)} segments)")

        # [D] keep_mask token count
        expected_kept = orig_seq - (N - K) * num_images
        if new_seq == expected_kept:
            print(f"    [D] PASS  keep_mask count = {new_seq}")
        else:
            errors.append(f"[D] FAIL  expected {expected_kept} kept tokens, got {new_seq}")

        # [E] contiguous_v2 only: per-row w is gap-free (0,1,...,count-1 + base)
        if rope_mode == "contiguous_v2":
            from collections import defaultdict
            tokens = input_ids[0].tolist()
            kept_indices = keep_mask[0].nonzero(as_tuple=True)[0].tolist()
            w_pos = pruned_pos[2, 0, :]  # w dimension
            h_pos = pruned_pos[1, 0, :]  # h dimension
            # Find pruned-seq positions of image tokens, grouped by image
            # Each image contributes exactly K image tokens (all kept)
            img_pis: list[list[int]] = [[] for _ in range(num_images)]
            img_counter = 0
            for pi, oi in enumerate(kept_indices):
                if tokens[oi] == IMAGE_TOKEN_ID:
                    img_pis[img_counter].append(pi)
                    if len(img_pis[img_counter]) == K:
                        img_counter += 1
                        if img_counter == num_images:
                            break
            gap_errors = []
            for img_i in range(min(num_images, 2)):  # check first 2 images
                h2w: dict = defaultdict(list)
                for pi in img_pis[img_i]:
                    h2w[h_pos[pi].item()].append(w_pos[pi].item())
                for h_val, ws in sorted(h2w.items()):
                    ws_sorted = sorted(ws)
                    base = ws_sorted[0]
                    expected = list(range(base, base + len(ws_sorted)))
                    if ws_sorted != expected:
                        gap_errors.append(
                            f"img{img_i} h={int(h_val)}: w={ws_sorted} (expected {expected})"
                        )
            if gap_errors:
                errors.extend([f"[E] Gap: {e}" for e in gap_errors])
            else:
                print(f"    [E] PASS  per-row w is contiguous (checked {min(num_images,2)} images)")

        if errors:
            all_ok = False
            for e in errors:
                print(f"    {e}")

    return all_ok


# ── Position-IDs dump ─────────────────────────────────────────────────────────

def _dump_position_ids(model, input_ids, image_grid_thw, sparsity_ratio, out_path):
    spatial_merge_size = model.vlm.config.vision_config.spatial_merge_size
    IMAGE_TOKEN_ID = model.vlm.config.image_token_id
    num_images = image_grid_thw.shape[0]

    tokens_per_image = (
        image_grid_thw[:, 0]
        * (image_grid_thw[:, 1] // spatial_merge_size)
        * (image_grid_thw[:, 2] // spatial_merge_size)
    ).tolist()
    N = min(int(t) for t in tokens_per_image)
    K = int(N * (1 - sparsity_ratio))

    torch.manual_seed(42)
    token_indices = torch.stack([
        torch.randperm(N)[:K].sort().values for _ in range(num_images)
    ])

    def tok_type(tid):
        return "IMG" if tid == IMAGE_TOKEN_ID else "txt"

    def write_section(f, label, pos, seq_ids):
        max_pos = pos.max().item()
        f.write(f"\n{'='*80}\n")
        f.write(f"PRUNED [{label}]  (new_seq={len(seq_ids)}, max_pos={max_pos})\n")
        f.write(f"{'='*80}\n")
        f.write(f"{'new_pos':>7}  {'orig_pos':>8}    {'tok_id':>6}   {'type':>4}  {'t':>5}  {'h':>5}  {'w':>5}\n")
        f.write(f"{'-'*55}\n")
        for new_p, (orig_p, tid) in enumerate(seq_ids):
            t_ = pos[0, 0, new_p].item()
            h_ = pos[1, 0, new_p].item()
            w_ = pos[2, 0, new_p].item()
            f.write(f"{new_p:>7}  {orig_p:>8}    {tid:>6}   {tok_type(tid):>4}  {t_:>5}  {h_:>5}  {w_:>5}\n")

    with torch.no_grad():
        orig_pos, orig_delta = model.vlm.model.get_rope_index(
            input_ids, image_grid_thw, None, None,
        )

    tokens = input_ids[0].tolist()

    with open(out_path, "w") as f:
        f.write(f"N={N} tokens/image, K={K}, num_images={num_images}, sparsity={sparsity_ratio}\n")
        f.write(f"image_grid_thw[0]: {image_grid_thw[0].tolist()}  "
                f"(H_post={image_grid_thw[0,1].item()//spatial_merge_size}, "
                f"W_post={image_grid_thw[0,2].item()//spatial_merge_size})\n")

        # Original
        seq_len = input_ids.shape[1]
        f.write(f"\n{'='*80}\n")
        f.write(f"ORIGINAL  (seq_len={seq_len}, max_pos={orig_pos.max().item()}, "
                f"rope_delta={orig_delta[0,0].item()})\n")
        f.write(f"{'='*80}\n")
        f.write(f"{'pos':>6}    {'tok_id':>6}   {'type':>4}  {'t':>5}  {'h':>5}  {'w':>5}\n")
        f.write(f"{'-'*45}\n")
        for p, tid in enumerate(tokens):
            t_ = orig_pos[0, 0, p].item()
            h_ = orig_pos[1, 0, p].item()
            w_ = orig_pos[2, 0, p].item()
            f.write(f"{p:>6}    {tid:>6}   {tok_type(tid):>4}  {t_:>5}  {h_:>5}  {w_:>5}\n")

        # Pruned modes
        modes = [
            ("direct",        lambda: model._get_pruned_rope_index(input_ids, image_grid_thw, token_indices, rope_mode="direct")),
            ("contiguous",    lambda: model._get_pruned_rope_index(input_ids, image_grid_thw, token_indices, rope_mode="contiguous")),
            ("reshape",       lambda: model._get_pruned_rope_index(input_ids, image_grid_thw, token_indices, rope_mode="reshape")),
            ("contiguous_v2", lambda: _get_pruned_rope_contiguous_v2(model, input_ids, image_grid_thw, token_indices)),
        ]
        for label, fn in modes:
            pruned_pos, pruned_delta, keep_mask = fn()
            kept_orig = keep_mask[0].nonzero(as_tuple=True)[0].tolist()
            seq_ids = [(orig_p, tokens[orig_p]) for orig_p in kept_orig]
            write_section(f, label, pruned_pos, seq_ids)

    print(f"\n  Position IDs written to {out_path}  ({sum(1 for _ in open(out_path))} lines)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./Alpamayo-R1-10B")
    parser.add_argument("--clip_id", default="b80a15fc-d540-4c8f-81d1-5db83216b2e0")
    parser.add_argument("--t0_us", type=int, default=1_700_000)
    parser.add_argument("--sparsity_ratio", type=float, default=0.5)
    parser.add_argument("--dump_file", type=str, default=None,
                        help="If set, write full position_ids for all modes to this file.")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} ...")
    model = AlpamayoR1.from_pretrained(args.model_path, dtype=torch.bfloat16)
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    print("Model loaded.\n")
    print(f"  image_token_id     = {model.vlm.config.image_token_id}")
    print(f"  vision_start_id    = {model.vlm.config.vision_start_token_id}")
    print(f"  spatial_merge_size = {model.vlm.config.vision_config.spatial_merge_size}")

    print(f"\n{'='*70}")
    print(f"Clip: {args.clip_id}  t0_us={args.t0_us}")
    print(f"{'='*70}")
    inputs = build_inputs(processor, args.clip_id, t0_us=args.t0_us)

    print(f"\n  image_grid_thw (all images):")
    for i, row in enumerate(inputs["image_grid_thw"].tolist()):
        print(f"    [{i:2d}] T={row[0]} H={row[1]} W={row[2]}  "
              f"→ {row[0] * (row[1]//2) * (row[2]//2)} post-merger tokens")

    ok = verify_rope(
        model, inputs["input_ids"], inputs["image_grid_thw"],
        sparsity_ratio=args.sparsity_ratio,
    )

    if args.dump_file:
        _dump_position_ids(
            model, inputs["input_ids"], inputs["image_grid_thw"],
            sparsity_ratio=args.sparsity_ratio,
            out_path=args.dump_file,
        )

    print(f"\n{'='*70}")
    print(f"FINAL: {'ALL PASSED' if ok else 'SOME FAILED'}")
    print(f"{'='*70}")
    sys.exit(0 if ok else 1)
