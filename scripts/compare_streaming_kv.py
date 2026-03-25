#!/usr/bin/env python3
"""Compare KV cache and decoded CoT between base (full prefill) and streaming.

Uses the same decode loop as StreamingAlpamayo1_5: temperature=0.6, top_p=0.98,
multinomial sampling, stop at <traj_future_start>.

Seeds are fixed before each decode for fair comparison.
"""

import sys
import logging

import torch

import alpamayo_r1
sys.modules["alpamayo1_5"] = alpamayo_r1

from alpamayo_r1.train.alpamayo1_5 import Alpamayo1_5
from alpamayo_r1.train.patches import patch_for_training, StaticCache
from alpamayo_r1.helper import convert_to_streaming_window_v1p5
from alpamayo_r1.models.token_utils import (
    extract_text_tokens,
    replace_padding_after_eos,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = "/mnt/moosefs/users/zekail/dumped_eval_data_v1p5"
CLIP_IDS = [
    "2ee1b1cc-4eff-44e8-bb1b-99648ff2b3c0",
    "6ee3a518-8073-4da2-beda-63e923708fa7",
    "9ba94e96-2543-4357-89fa-5344af1a183a",
    "bbbc5d5d-b15e-4601-a639-686dbf1bdfb9",
    "53baf60a-902f-446d-8e30-5eb7dbc992e7",
]
MODEL_PATH = "nvidia/Alpamayo-1.5-10B"
NUM_STEPS = 50
MAX_NEW_TOKENS = 128
COMPARE_LAYERS = [0, 15, 27]
TEMPERATURE = 0.6
TOP_P = 0.98
DECODE_SEED = 42


def cosine_sim(a, b):
    a_flat = a.reshape(-1, a.shape[-1]).float()
    b_flat = b.reshape(-1, b.shape[-1]).float()
    return torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=-1).mean().item()


def decode_cot(model, logits, input_ids, start_pos, max_tokens, seed):
    """Decode CoT using same logic as StreamingAlpamayo1_5."""
    logits_processor = model._build_logits_processor(TEMPERATURE, None, TOP_P)
    device = logits.device

    output_ids = input_ids.clone()
    unfinished = torch.ones(1, dtype=torch.bool, device=device)
    cur_pos = start_pos

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    for _ in range(max_tokens):
        logits = logits_processor(output_ids, logits)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        next_token = torch.where(unfinished, next_token, model.tokenizer.pad_token_id)
        output_ids = torch.cat([output_ids, next_token.unsqueeze(-1)], dim=-1)

        unfinished = unfinished & (next_token != model.traj_start_token_id)
        if not unfinished.any():
            break

        logits = model._decode(
            input_ids=next_token.unsqueeze(-1),
            position_ids=model._cached_position_ids,
            cache_position=torch.tensor([cur_pos], device=device),
        )
        cur_pos += 1

    output_ids = replace_padding_after_eos(
        token_ids=output_ids,
        eos_token_id=model.traj_start_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
    )

    cot_dict = extract_text_tokens(model.tokenizer, output_ids)
    return output_ids, cot_dict


def run_base(model, window, device):
    """Fresh full prefill + decode."""
    model.reset_streaming_state()
    model.keep_frame_labels = True

    td = window["tokenized_data"]
    input_ids = td["input_ids"].to(device)
    attention_mask = td["attention_mask"].to(device)
    pixel_values = td["pixel_values"].to(device)
    image_grid_thw = td["image_grid_thw"].to(device)
    ego_h_xyz = window["ego_history_xyz"].to(device)
    ego_h_rot = window["ego_history_rot"].to(device)

    input_ids = model.fuse_traj_tokens(input_ids, {
        "ego_history_xyz": ego_h_xyz, "ego_history_rot": ego_h_rot,
    })

    model.prefill_seq_length = input_ids.shape[1]
    model.max_cache_len = model.prefill_seq_length + MAX_NEW_TOKENS + model.num_action_tokens + 16

    model._past_key_values = StaticCache(
        config=model.vlm.config,
        max_cache_len=model.max_cache_len,
        max_batch_size=1,
        offloading=False,
    )

    logits = model._first_prefill(input_ids, attention_mask, pixel_values, image_grid_thw, device)

    keys = {}
    for li in COMPARE_LAYERS:
        keys[li] = model._past_key_values.layers[li].keys[:1, :, :model.prefill_seq_length, :].clone().cpu()

    output_ids, cot_dict = decode_cot(model, logits, input_ids, model.prefill_seq_length, MAX_NEW_TOKENS, DECODE_SEED)
    return keys, cot_dict, output_ids


def run_streaming_all(model, windows, keep_frame_labels, kv_shift_mode, device):
    """Run streaming for NUM_STEPS."""
    model.reset_streaming_state()
    model.keep_frame_labels = keep_frame_labels
    model.kv_shift_mode = kv_shift_mode

    tokenizer = model.tokenizer
    vs_id = tokenizer.encode("<|vision_start|>")[0]
    ve_id = tokenizer.encode("<|vision_end|>")[0]

    results = {}

    for step in range(NUM_STEPS):
        w = windows[step]

        if step == 0:
            td = w["tokenized_data"]
            input_ids = td["input_ids"].to(device)
            attention_mask = td["attention_mask"].to(device)
            pixel_values = td["pixel_values"].to(device)
            image_grid_thw = td["image_grid_thw"].to(device)
            ego_h_xyz = w["ego_history_xyz"].to(device)
            ego_h_rot = w["ego_history_rot"].to(device)

            input_ids = model.fuse_traj_tokens(input_ids, {
                "ego_history_xyz": ego_h_xyz, "ego_history_rot": ego_h_rot,
            })

            model.prefill_seq_length = input_ids.shape[1]
            model.max_cache_len = model.prefill_seq_length + MAX_NEW_TOKENS + model.num_action_tokens + 16

            model._past_key_values = StaticCache(
                config=model.vlm.config,
                max_cache_len=model.max_cache_len,
                max_batch_size=1,
                offloading=False,
            )

            logits = model._first_prefill(input_ids, attention_mask, pixel_values, image_grid_thw, device)

            keys = {}
            for li in COMPARE_LAYERS:
                keys[li] = model._past_key_values.layers[li].keys[:1, :, :model.prefill_seq_length, :].clone().cpu()

            output_ids, cot_dict = decode_cot(model, logits, input_ids, model.prefill_seq_length, MAX_NEW_TOKENS, DECODE_SEED)
            results[step] = (keys, cot_dict, output_ids)

            # Crop decode tokens from cache before shifting
            for layer in model._past_key_values.layers:
                if layer.is_initialized:
                    layer.keys[:, :, model.prefill_seq_length:, :].zero_()
                    layer.values[:, :, model.prefill_seq_length:, :].zero_()

            model._update_past_key_values()
            model.is_first_prefill = False

        else:
            sw = convert_to_streaming_window_v1p5(
                w, ve_id,
                keep_frame_labels=keep_frame_labels,
                vision_start_id=vs_id if not keep_frame_labels else None,
            )
            td = sw["tokenized_data"]
            input_ids = td["input_ids"].to(device)
            pixel_values = td["pixel_values"].to(device)
            image_grid_thw = td["image_grid_thw"].to(device)
            ego_h_xyz = sw["ego_history_xyz"].to(device)
            ego_h_rot = sw["ego_history_rot"].to(device)

            input_ids = model.fuse_traj_tokens(input_ids, {
                "ego_history_xyz": ego_h_xyz, "ego_history_rot": ego_h_rot,
            })

            image_embeds, deepstack_image_embeds = model._encode(pixel_values, image_grid_thw)
            inputs_embeds = model.vlm.model.get_input_embeddings()(input_ids)
            cache_position = model._create_cache_position().to(device)

            image_mask = (input_ids == model.vlm.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if model._cached_streaming_attention_mask is None:
                model._cached_streaming_attention_mask = model._get_streaming_attention_mask(
                    cache_position=cache_position, device=device,
                )

            logits = model._prefill(
                inputs_embeds=inputs_embeds,
                position_ids=model._cached_position_ids,
                cache_position=cache_position,
                visual_pos_masks=image_mask[..., 0],
                deepstack_image_embeds=deepstack_image_embeds,
                streaming_attention_mask=model._cached_streaming_attention_mask,
            )

            keys = {}
            for li in COMPARE_LAYERS:
                keys[li] = model._past_key_values.layers[li].keys[:1, :, :model.prefill_seq_length, :].clone().cpu()

            decode_start = cache_position[-1].item() + 1
            output_ids, cot_dict = decode_cot(model, logits, input_ids, decode_start, MAX_NEW_TOKENS, DECODE_SEED)
            results[step] = (keys, cot_dict, output_ids)

            # Crop decode tokens from cache before shifting
            for layer in model._past_key_values.layers:
                if layer.is_initialized:
                    layer.keys[:, :, model.prefill_seq_length:, :].zero_()
                    layer.values[:, :, model.prefill_seq_length:, :].zero_()

            model._update_past_key_values()

    return results


@torch.inference_mode()
def run_clip(model, clip_id, device):
    """Run comparison for one clip. Returns summary dict."""
    import os
    clip_path = os.path.join(DATA_DIR, clip_id, "sliding_window_inputs.pt")
    windows = torch.load(clip_path, map_location="cpu", weights_only=False)
    n = min(NUM_STEPS, len(windows))

    # Base
    base_results = {}
    for step in range(n):
        keys, cot_dict, oids = run_base(model, windows[step], device)
        base_results[step] = (keys, cot_dict, oids)

    # Vision ranges (from window 0)
    model.reset_streaming_state()
    model.keep_frame_labels = True
    td = windows[0]["tokenized_data"]
    input_ids = td["input_ids"].to(device)
    input_ids = model.fuse_traj_tokens(input_ids, {
        "ego_history_xyz": windows[0]["ego_history_xyz"].to(device),
        "ego_history_rot": windows[0]["ego_history_rot"].to(device),
    })
    model.prefill_seq_length = input_ids.shape[1]

    # Streaming A & B
    stream_a = run_streaming_all(model, windows[:n], True, "block", device)
    stream_b = run_streaming_all(model, windows[:n], False, "vision_only", device)

    tokenizer = model.tokenizer
    cot_start_id = tokenizer.convert_tokens_to_ids("<|cot_start|>")

    def get_raw(oids):
        ids = oids[0].tolist()
        for i in range(len(ids) - 1, -1, -1):
            if ids[i] == cot_start_id:
                return tokenizer.decode(ids[i:], skip_special_tokens=False)
        return tokenizer.decode(ids[-50:], skip_special_tokens=False)

    # Collect stats
    a_diff_count = 0
    b_diff_count = 0
    garbled = []

    for step in range(n):
        _, _, b_oids = base_results[step]
        _, _, a_oids = stream_a[step]
        _, _, bk_oids = stream_b[step]

        b_raw = get_raw(b_oids)
        a_raw = get_raw(a_oids)
        bk_raw = get_raw(bk_oids)

        if a_raw != b_raw:
            a_diff_count += 1
        if bk_raw != b_raw:
            b_diff_count += 1

        # Check for garbled: no cot_end, or contains non-ASCII, or very short
        for label, raw in [("A", a_raw), ("B", bk_raw), ("Base", b_raw)]:
            if "<|cot_end|>" not in raw:
                garbled.append((step, label, raw))
            # Check for non-printable chars (excluding special tokens)
            clean = raw
            for st in ["<|cot_start|>", "<|cot_end|>", "<|traj_future_start|>"]:
                clean = clean.replace(st, "")
            if any(ord(c) > 127 or (ord(c) < 32 and c not in '\n\r\t') for c in clean):
                garbled.append((step, label, raw))

    # KV at last step
    b_keys = base_results[n-1][0]
    a_keys = stream_a[n-1][0]
    bk_keys = stream_b[n-1][0]
    l27_a = cosine_sim(b_keys[27], a_keys[27])
    l27_b = cosine_sim(b_keys[27], bk_keys[27])

    # Show some divergent examples
    examples = []
    for step in range(n):
        b_raw = get_raw(base_results[step][2])
        a_raw = get_raw(stream_a[step][2])
        bk_raw = get_raw(stream_b[step][2])
        if a_raw != b_raw or bk_raw != b_raw:
            examples.append((step, b_raw, a_raw, bk_raw))
            if len(examples) >= 3:
                break

    return {
        "clip_id": clip_id,
        "steps": n,
        "a_diff": a_diff_count,
        "b_diff": b_diff_count,
        "garbled": garbled,
        "l27_a": l27_a,
        "l27_b": l27_b,
        "examples": examples,
    }


@torch.inference_mode()
def main():
    device = torch.device("cuda")

    log.info("Loading model...")
    model = Alpamayo1_5.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to(device)
    patch_for_training(model)
    log.info("Model loaded")

    for ci, clip_id in enumerate(CLIP_IDS):
        log.info(f"\n{'='*80}")
        log.info(f"Clip {ci+1}/{len(CLIP_IDS)}: {clip_id}")
        log.info(f"{'='*80}")

        result = run_clip(model, clip_id, device)

        print(f"\n{'='*80}")
        print(f"Clip {clip_id[:8]}... ({result['steps']} steps)")
        print(f"  CoT divergence: A={result['a_diff']}/{result['steps']}, B={result['b_diff']}/{result['steps']}")
        print(f"  L27 cos @ last step: A={result['l27_a']:.4f}, B={result['l27_b']:.4f}")

        if result["garbled"]:
            print(f"  *** GARBLED TOKENS FOUND: {len(result['garbled'])} ***")
            for step, label, raw in result["garbled"][:5]:
                print(f"    Step {step} [{label}]: {repr(raw[:200])}")
        else:
            print(f"  No garbled tokens")

        if result["examples"]:
            print(f"  Example divergences:")
            for step, b, a, bk in result["examples"]:
                b_clean = b.replace("<|cot_start|>", "").replace("<|cot_end|>", "").replace("<|traj_future_start|>", "").strip()
                a_clean = a.replace("<|cot_start|>", "").replace("<|cot_end|>", "").replace("<|traj_future_start|>", "").strip()
                bk_clean = bk.replace("<|cot_start|>", "").replace("<|cot_end|>", "").replace("<|traj_future_start|>", "").strip()
                print(f"    Step {step}:")
                print(f"      Base: {b_clean[:150]}")
                if a_clean != b_clean:
                    print(f"      A:    {a_clean[:150]}")
                if bk_clean != b_clean:
                    print(f"      B:    {bk_clean[:150]}")

    print("\nAll clips done.")


if __name__ == "__main__":
    main()
