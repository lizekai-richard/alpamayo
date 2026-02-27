"""Compare streaming vs non-streaming KV caches (lightweight, sampling-based).

Samples a few steps / layers / (step, layer) pairs to visualise differences
without loading everything into memory.

Produces:
  1. Overall summary (L1, MSE, cosine similarity, relative error).
  2. Per-layer curves   – sample a few steps, plot metric vs layer.
  3. Per-step curves    – sample a few layers, plot metric vs step.
  4. Per-token curves   – sample a few (step, layer) pairs, plot metric vs token.

Usage:
    python compare_kv_cache.py --loss_group high_loss
    python compare_kv_cache.py --loss_group low_loss --n_sample 8
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.figsize": (14, 5),
})

NUM_LAYERS = 36


# ──────────────────────────── helpers ──────────────────────────────────── #


def _find_step_dir(parent_dir: str, step: int) -> str:
    """Resolve step directory (tolerates `step0` and `step_0`)."""
    for pat in [f"step_{step}", f"step{step}"]:
        p = os.path.join(parent_dir, pat)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"Step {step} not found under {parent_dir}")


def _available_steps(parent_dir: str) -> list[int]:
    steps = set()
    for name in os.listdir(parent_dir):
        m = re.match(r"step_?(\d+)$", name)
        if m and os.path.isdir(os.path.join(parent_dir, name)):
            steps.add(int(m.group(1)))
    return sorted(steps)


def _sample_indices(total: int, n: int) -> list[int]:
    """Return *n* evenly-spaced indices from [0, total)."""
    if n >= total:
        return list(range(total))
    return np.linspace(0, total - 1, n, dtype=int).tolist()


def load_pair(
    s_dir: str, ns_dir: str, step: int, cache_type: str, layer: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one (streaming, non-streaming) cache pair as float32."""
    s_step = _find_step_dir(s_dir, step)
    ns_step = _find_step_dir(ns_dir, step)
    fname = f"{cache_type}_cache_{layer}.pt"
    s = torch.load(os.path.join(s_step, fname), map_location="cpu", weights_only=True).float()
    ns = torch.load(os.path.join(ns_step, fname), map_location="cpu", weights_only=True).float()
    return s, ns


# ──────────────────────────── metrics ──────────────────────────────────── #


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Scalar metrics between two tensors of the same shape."""
    diff = a - b
    a_flat, b_flat = a.reshape(-1), b.reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()
    return {
        "l1": diff.abs().mean().item(),
        "mse": diff.pow(2).mean().item(),
        "cos": cos,
        "rel_err": diff.norm().item() / (b.norm().item() + 1e-8),
    }


def _per_token_metrics(a: torch.Tensor, b: torch.Tensor):
    """Per-token L1 and cosine similarity.  a, b: [B, H, S, D]."""
    token_l1 = (a - b).abs().mean(dim=(0, 1, 3)).numpy()  # [S]
    # cosine along head_dim
    a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    token_cos = (a_n * b_n).sum(dim=-1).mean(dim=(0, 1)).numpy()  # [S]
    return token_l1, token_cos


# ──────────────────────────── analysis ─────────────────────────────────── #


def run_per_layer(
    s_dir, ns_dir, steps: list[int], layers: list[int], cache_types=("key", "value")
):
    """For each sampled step, compute metric at every layer (averaged over key+value).

    Returns dict  step -> {"l1": [L], "cos": [L]}  (L = len(layers)).
    """
    out = {}
    for step in steps:
        l1_arr, cos_arr = [], []
        for layer in layers:
            m_agg = {"l1": 0.0, "cos": 0.0}
            for ct in cache_types:
                s, ns = load_pair(s_dir, ns_dir, step, ct, layer)
                m = _metrics(s, ns)
                m_agg["l1"] += m["l1"]
                m_agg["cos"] += m["cos"]
            m_agg = {k: v / len(cache_types) for k, v in m_agg.items()}
            l1_arr.append(m_agg["l1"])
            cos_arr.append(m_agg["cos"])
        out[step] = {"l1": np.array(l1_arr), "cos": np.array(cos_arr)}
    return out


def run_per_step(
    s_dir, ns_dir, steps: list[int], layers: list[int], cache_types=("key", "value")
):
    """For each sampled layer, compute metric at every step (averaged over key+value).

    Returns dict  layer -> {"l1": [S], "cos": [S]}  (S = len(steps)).
    """
    out = {}
    for layer in layers:
        l1_arr, cos_arr = [], []
        for step in steps:
            m_agg = {"l1": 0.0, "cos": 0.0}
            for ct in cache_types:
                s, ns = load_pair(s_dir, ns_dir, step, ct, layer)
                m = _metrics(s, ns)
                m_agg["l1"] += m["l1"]
                m_agg["cos"] += m["cos"]
            m_agg = {k: v / len(cache_types) for k, v in m_agg.items()}
            l1_arr.append(m_agg["l1"])
            cos_arr.append(m_agg["cos"])
        out[layer] = {"l1": np.array(l1_arr), "cos": np.array(cos_arr)}
    return out


def run_per_token(
    s_dir, ns_dir, pairs: list[tuple[int, int]], cache_types=("key", "value")
):
    """For each (step, layer) pair, compute per-token metrics (averaged over key+value).

    Returns list of {"step", "layer", "token_l1": [T], "token_cos": [T]}.
    """
    out = []
    for step, layer in pairs:
        agg_l1, agg_cos = None, None
        for ct in cache_types:
            s, ns = load_pair(s_dir, ns_dir, step, ct, layer)
            tl1, tcos = _per_token_metrics(s, ns)
            if agg_l1 is None:
                agg_l1, agg_cos = tl1, tcos
            else:
                agg_l1 += tl1
                agg_cos += tcos
        agg_l1 /= len(cache_types)
        agg_cos /= len(cache_types)
        out.append({"step": step, "layer": layer, "token_l1": agg_l1, "token_cos": agg_cos})
    return out


# ──────────────────────────── printing ─────────────────────────────────── #


def print_summary(s_dir, ns_dir, steps, layers, cache_types=("key", "value")):
    """Print a compact summary over sampled steps × layers."""
    print("\n" + "=" * 70)
    print(f"{'Step':>5} | {'L1':>10} {'MSE':>10} {'Cos':>10} {'RelErr':>10}")
    print("-" * 70)
    for step in steps:
        agg = {"l1": 0, "mse": 0, "cos": 0, "rel_err": 0}
        cnt = 0
        for layer in layers:
            for ct in cache_types:
                s, ns = load_pair(s_dir, ns_dir, step, ct, layer)
                m = _metrics(s, ns)
                for k in agg:
                    agg[k] += m[k]
                cnt += 1
        agg = {k: v / cnt for k, v in agg.items()}
        print(f"{step:>5} | {agg['l1']:>10.4f} {agg['mse']:>10.4f} "
              f"{agg['cos']:>10.4f} {agg['rel_err']:>10.4f}")
    print("=" * 70)


# ──────────────────────────── plotting ─────────────────────────────────── #


def plot_per_layer(data: dict, layers: list[int], save_dir: str):
    """data: step -> {"l1": [...], "cos": [...]}"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for step, d in data.items():
        axes[0].plot(layers, d["l1"], "o-", label=f"step {step}", alpha=0.8, markersize=3)
        axes[1].plot(layers, d["cos"], "o-", label=f"step {step}", alpha=0.8, markersize=3)

    axes[0].set(title="L1 Difference per Layer", xlabel="Layer", ylabel="Mean |Δ|")
    axes[1].set(title="Cosine Similarity per Layer", xlabel="Layer", ylabel="Cosine Sim")
    for ax in axes:
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Streaming vs Non-Streaming: Per-Layer (sampled steps)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(save_dir, "per_layer.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_per_step(data: dict, steps: list[int], save_dir: str):
    """data: layer -> {"l1": [...], "cos": [...]}"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for layer, d in data.items():
        axes[0].plot(steps, d["l1"], "o-", label=f"layer {layer}", alpha=0.8, markersize=3)
        axes[1].plot(steps, d["cos"], "o-", label=f"layer {layer}", alpha=0.8, markersize=3)

    axes[0].set(title="L1 Difference per Step", xlabel="Step", ylabel="Mean |Δ|")
    axes[1].set(title="Cosine Similarity per Step", xlabel="Step", ylabel="Cosine Sim")
    for ax in axes:
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Streaming vs Non-Streaming: Per-Step (sampled layers)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(save_dir, "per_step.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_per_token(token_results: list[dict], save_dir: str):
    """Each entry: {"step", "layer", "token_l1": [T], "token_cos": [T]}."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for r in token_results:
        label = f"step={r['step']}, layer={r['layer']}"
        axes[0].plot(r["token_l1"], label=label, alpha=0.7, linewidth=0.8)
        axes[1].plot(r["token_cos"], label=label, alpha=0.7, linewidth=0.8)

    axes[0].set(title="L1 Difference per Token", xlabel="Token Position", ylabel="Mean |Δ|")
    axes[1].set(title="Cosine Similarity per Token", xlabel="Token Position", ylabel="Cosine Sim")
    for ax in axes:
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Streaming vs Non-Streaming: Per-Token (sampled (step, layer) pairs)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(save_dir, "per_token.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ────────────────────────────── main ───────────────────────────────────── #


def main():
    parser = argparse.ArgumentParser(description="Compare streaming vs non-streaming KV caches.")
    parser.add_argument("--base_dir", type=str, default="saved_kv_cache")
    parser.add_argument("--loss_group", type=str, default="high_loss", choices=["high_loss", "low_loss"])
    parser.add_argument("--clip_id", type=str, default=None, help="Auto-detected if omitted.")
    parser.add_argument("--n_sample", type=int, default=6, help="Number of steps/layers to sample.")
    parser.add_argument("--save_dir", type=str, default="kv_cache_comparison")
    args = parser.parse_args()

    streaming_root = os.path.join(args.base_dir, args.loss_group, "streaming")
    non_streaming_root = os.path.join(args.base_dir, args.loss_group, "non-streaming")

    # Auto-detect clip_id
    if args.clip_id is None:
        clips_s = {d for d in os.listdir(streaming_root) if os.path.isdir(os.path.join(streaming_root, d))}
        clips_ns = {d for d in os.listdir(non_streaming_root) if os.path.isdir(os.path.join(non_streaming_root, d))}
        common = sorted(clips_s & clips_ns)
        assert common, "No common clip directories found."
        args.clip_id = common[0]
        print(f"Auto-detected clip_id: {args.clip_id}")

    s_dir = os.path.join(streaming_root, args.clip_id)
    ns_dir = os.path.join(non_streaming_root, args.clip_id)

    all_steps = sorted(set(_available_steps(s_dir)) & set(_available_steps(ns_dir)))
    all_layers = list(range(NUM_LAYERS))
    N = args.n_sample

    sampled_step_idx = _sample_indices(len(all_steps), N)
    sampled_steps = [all_steps[i] for i in sampled_step_idx]
    sampled_layer_idx = _sample_indices(NUM_LAYERS, N)
    sampled_layers = [all_layers[i] for i in sampled_layer_idx]

    # (step, layer) pairs for per-token analysis: sample from grid
    token_pairs = [(all_steps[si], all_layers[li])
                   for si, li in zip(
                       _sample_indices(len(all_steps), N),
                       _sample_indices(NUM_LAYERS, N))]

    print(f"\n{'=' * 60}")
    print(f"  Streaming:     {s_dir}")
    print(f"  Non-streaming: {ns_dir}")
    print(f"  Total steps: {len(all_steps)},  sampled steps: {sampled_steps}")
    print(f"  Total layers: {NUM_LAYERS},  sampled layers: {sampled_layers}")
    print(f"  Per-token pairs (step, layer): {token_pairs}")
    print(f"{'=' * 60}\n")

    # ── summary table (over sampled steps × sampled layers) ──
    print("Computing summary ...")
    print_summary(s_dir, ns_dir, sampled_steps, sampled_layers)

    # ── per-layer (sampled steps, all layers) ──
    print("\nComputing per-layer metrics ...")
    pl_data = run_per_layer(s_dir, ns_dir, sampled_steps, all_layers)

    # ── per-step (all steps, sampled layers) ──
    print("Computing per-step metrics ...")
    ps_data = run_per_step(s_dir, ns_dir, all_steps, sampled_layers)

    # ── per-token ──
    print("Computing per-token metrics ...")
    pt_data = run_per_token(s_dir, ns_dir, token_pairs)

    # ── save plots ──
    save_dir = os.path.join(args.save_dir, args.loss_group, args.clip_id)
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nSaving plots to {save_dir}/")

    plot_per_layer(pl_data, all_layers, save_dir)
    plot_per_step(ps_data, all_steps, save_dir)
    plot_per_token(pt_data, save_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
