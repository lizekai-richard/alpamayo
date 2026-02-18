"""Visualize vision encoder attention maps to assess sparsity.

Usage:
    # Single file:
    #   python visualize_attention.py --attn_path attn.pt --save_dir attn_vis/

    # All clips under a directory (e.g. saved_attn_weights/<clip_id>/attn_weights.pt):
    #   python visualize_attention.py --attn_dir saved_attn_weights/ --save_dir attn_vis/

    # With explicit grid:
    #   python visualize_attention.py --attn_path attn.pt --grid_thw 1,28,28 --save_dir attn_vis/
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_attention_maps(
    attn_weights: torch.Tensor,
    image_grid_thw: torch.Tensor,
    save_dir: str = "attn_vis",
    num_example_queries: int = 5,
    spatial_merge_size: int = 2,
):
    """Visualize attention maps from the vision encoder's last layer.

    Args:
        attn_weights: [num_chunks, num_heads, chunk_size, chunk_size]
                      Post-merger resolution (already merged by _merge_attn_weights).
        image_grid_thw: [num_images, 3] with (T, H, W) per image.
                        H, W are pre-merger grid dims.
        save_dir: directory to save figures.
        num_example_queries: number of query tokens to show spatial maps for.
        spatial_merge_size: merger size (default 2). Used to convert
                           pre-merger H,W to post-merger h,w.
    """
    os.makedirs(save_dir, exist_ok=True)
    attn_weights = attn_weights.float().cpu()
    num_heads = attn_weights.shape[1]

    # Head-averaged attention
    attn_avg = attn_weights.mean(dim=1)  # [num_chunks, chunk_size, chunk_size]

    chunk_idx = 0
    for img_i in range(image_grid_thw.shape[0]):
        t, h_pre, w_pre = image_grid_thw[img_i].tolist()
        h = h_pre // spatial_merge_size
        w = w_pre // spatial_merge_size
        for t_i in range(t):
            attn = attn_avg[chunk_idx].numpy()  # [chunk_size, chunk_size]
            attn_per_head = attn_weights[chunk_idx].numpy()  # [num_heads, cs, cs]
            chunk_size = attn.shape[0]
            prefix = f"img{img_i}_t{t_i}"
            print(f"\n{'='*60}")
            print(f"Image {img_i}, Frame {t_i}  (grid {h}x{w} = {chunk_size} tokens, {num_heads} heads)")
            print(f"{'='*60}")

            # ── 1. Full attention matrix (head-averaged) ──
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(attn, cmap="viridis", aspect="equal", interpolation="nearest")
            ax.set_title(f"Attention matrix (head avg)\nImage {img_i}, Frame {t_i} — {h}x{w}={chunk_size} tokens")
            ax.set_xlabel("Key")
            ax.set_ylabel("Query")
            plt.colorbar(im, ax=ax, fraction=0.046)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}_matrix.png"), dpi=150)
            plt.close()

            # ── 2. Spatial attention maps for selected queries ──
            q_positions = np.linspace(0, chunk_size - 1, num_example_queries, dtype=int)
            fig, axes = plt.subplots(1, len(q_positions), figsize=(4 * len(q_positions), 4))
            if len(q_positions) == 1:
                axes = [axes]
            for qi, q_idx in enumerate(q_positions):
                attn_2d = attn[q_idx].reshape(h, w)
                q_h, q_w = q_idx // w, q_idx % w
                im = axes[qi].imshow(attn_2d, cmap="hot", interpolation="nearest")
                axes[qi].scatter([q_w], [q_h], c="cyan", s=120, marker="x", linewidths=2)
                axes[qi].set_title(f"Query ({q_h},{q_w})")
                plt.colorbar(im, ax=axes[qi], fraction=0.046)
            fig.suptitle(f"Spatial attention (head avg) — Image {img_i}, Frame {t_i}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}_spatial.png"), dpi=150)
            plt.close()

            # ── 3. Column sum (key importance) — directly relates to pruning ──
            col_sum = attn.sum(axis=0)  # [chunk_size] — total attention each key receives
            col_sum_2d = col_sum.reshape(h, w)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            im0 = axes[0].imshow(col_sum_2d, cmap="magma", interpolation="nearest")
            axes[0].set_title("Column sum (key importance)")
            plt.colorbar(im0, ax=axes[0], fraction=0.046)

            # Sorted column sum
            sorted_cs = np.sort(col_sum)[::-1]
            axes[1].plot(sorted_cs, color="coral", linewidth=1.5)
            axes[1].set_xlabel("Token rank")
            axes[1].set_ylabel("Column sum")
            axes[1].set_title("Column sum (sorted)")
            axes[1].axhline(y=col_sum.mean(), color="gray", linestyle="--", label=f"mean={col_sum.mean():.2f}")
            axes[1].legend()
            fig.suptitle(f"Key importance — Image {img_i}, Frame {t_i}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}_colsum.png"), dpi=150)
            plt.close()

            # ── 4. Sparsity analysis ──
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

            # 4a. Histogram of attention weights
            axes[0].hist(attn.flatten(), bins=200, log=True, color="steelblue", edgecolor="none")
            axes[0].set_xlabel("Attention weight")
            axes[0].set_ylabel("Count (log)")
            axes[0].set_title("Weight distribution")

            # 4b. Top-k coverage (per-query)
            sorted_attn = np.sort(attn, axis=1)[:, ::-1]
            cumsum = np.cumsum(sorted_attn, axis=1)
            ks = sorted(set([1, 3, 5, 10, 20, 50, chunk_size // 4, chunk_size // 2, chunk_size]))
            ks = [k for k in ks if 0 < k <= chunk_size]
            coverages = [cumsum[:, k - 1].mean() for k in ks]
            axes[1].bar(range(len(ks)), coverages, color="coral", edgecolor="none")
            axes[1].set_xticks(range(len(ks)))
            axes[1].set_xticklabels([str(k) for k in ks], fontsize=8)
            axes[1].set_xlabel("Top-k tokens")
            axes[1].set_ylabel("Mean coverage")
            axes[1].set_title("Top-k attn coverage (per-query)")
            axes[1].set_ylim(0, 1.1)
            for bi, v in enumerate(coverages):
                axes[1].text(bi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)

            # 4c. Colsum recall — keep top-k% keys by colsum, measure retained attention
            total_attn = attn.sum()
            colsum_sorted_idx = np.argsort(col_sum)[::-1]  # keys sorted by importance
            sparsities = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            keep_counts = (sparsities * chunk_size).astype(int)
            keep_counts = np.clip(keep_counts, 1, chunk_size)
            recalls = []
            for kc in keep_counts:
                kept_keys = colsum_sorted_idx[:kc]
                recall = attn[:, kept_keys].sum() / total_attn
                recalls.append(float(recall))
            axes[2].bar(range(len(sparsities)), recalls, color="teal", edgecolor="none")
            axes[2].set_xticks(range(len(sparsities)))
            axes[2].set_xticklabels([f"{s:.0%}" for s in sparsities], fontsize=8)
            axes[2].set_xlabel("Keep ratio (by colsum)")
            axes[2].set_ylabel("Attention recall")
            axes[2].set_title("Colsum pruning recall")
            axes[2].set_ylim(0, 1.1)
            for bi, v in enumerate(recalls):
                axes[2].text(bi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)

            fig.suptitle(f"Sparsity analysis — Image {img_i}, Frame {t_i}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}_sparsity.png"), dpi=150)
            plt.close()

            # ── 5. Per-head column sum comparison ──
            n_heads_show = min(num_heads, 8)
            fig, axes = plt.subplots(1, n_heads_show, figsize=(n_heads_show * 3, 3.5))
            if n_heads_show == 1:
                axes = [axes]
            for hi in range(n_heads_show):
                head_attn = attn_per_head[hi]  # [cs, cs]
                head_cs = head_attn.sum(axis=0).reshape(h, w)
                im = axes[hi].imshow(head_cs, cmap="magma", interpolation="nearest")
                axes[hi].set_title(f"Head {hi}", fontsize=9)
                axes[hi].set_xticks([])
                axes[hi].set_yticks([])
            axes[0].set_ylabel("Column sum")
            fig.suptitle(f"Per-head column sum — Image {img_i}, Frame {t_i}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}_per_head.png"), dpi=150)
            plt.close()

            # ── Print statistics ──
            print(f"  Attention weights: mean={attn.mean():.6f} std={attn.std():.6f} "
                  f"min={attn.min():.6f} max={attn.max():.4f}")
            print(f"  Sparsity (fraction below threshold):")
            uniform = 1.0 / chunk_size
            for thresh in [uniform * 0.1, uniform * 0.5, uniform, uniform * 2]:
                frac = (attn < thresh).mean()
                print(f"    < {thresh:.6f} ({thresh/uniform:.1f}x uniform): {frac:.1%}")
            print(f"  Top-k coverage (mean over queries):")
            for k, cov in zip(ks, coverages):
                if cov < 0.999:
                    print(f"    top-{k:>4d} ({k/chunk_size:>5.1%} of tokens): {cov:.2%}")

            chunk_idx += 1

    print(f"\nFigures saved to {save_dir}/")


def load_and_visualize(attn_path: str, save_dir: str, grid_thw_str: str | None = None, num_queries: int = 5):
    """Load a single .pt file and run visualization."""
    data = torch.load(attn_path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        attn_w = data["attn_weights"]
        grid_thw = data.get("image_grid_thw")
    elif isinstance(data, list):
        attn_w = data[0]
        grid_thw = None
    else:
        attn_w = data
        grid_thw = None

    if grid_thw_str:
        rows = grid_thw_str.split(";")
        grid_thw = torch.tensor([[int(x) for x in r.split(",")] for r in rows])

    if grid_thw is None:
        num_chunks, _, cs, _ = attn_w.shape
        side = int(cs ** 0.5)
        grid_thw = torch.tensor([[num_chunks, side, side]])
        print(f"Inferred grid_thw = {grid_thw.tolist()} from shape {attn_w.shape}")

    visualize_attention_maps(attn_w, grid_thw, save_dir=save_dir, num_example_queries=num_queries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize vision encoder attention maps")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--attn_path", type=str,
                       help="Path to a single saved attention weights (.pt).")
    group.add_argument("--attn_dir", type=str,
                       help="Directory containing clip subfolders, each with attn_weights_step*.pt. "
                            "E.g. saved_attn_weights/<clip_id>/attn_weights_step0.pt")
    parser.add_argument("--grid_thw", type=str, default=None,
                        help="Grid dims as 'T,H,W' (e.g. '1,28,28'). "
                             "Can also be 'T1,H1,W1;T2,H2,W2' for multiple images.")
    parser.add_argument("--save_dir", type=str, default="attn_vis")
    parser.add_argument("--num_queries", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=6,
                        help="Max number of steps to visualize per clip (evenly sampled).")
    args = parser.parse_args()

    if args.attn_path:
        load_and_visualize(args.attn_path, args.save_dir, args.grid_thw, args.num_queries)
    else:
        import glob as glob_mod
        import re

        # Scan attn_dir for clip subfolders containing attn_weights_step*.pt
        clip_dirs = sorted([
            d for d in os.listdir(args.attn_dir)
            if os.path.isdir(os.path.join(args.attn_dir, d))
        ])
        print(f"Found {len(clip_dirs)} clip folders in {args.attn_dir}")
        for clip_id in clip_dirs:
            clip_path = os.path.join(args.attn_dir, clip_id)
            step_files = sorted(
                glob_mod.glob(os.path.join(clip_path, "attn_weights_step*.pt")),
                key=lambda f: int(re.search(r"step(\d+)", f).group(1)),
            )
            if not step_files:
                continue

            # Sample a few steps from beginning, middle, and end
            n = len(step_files)
            if n <= args.max_steps:
                selected_indices = list(range(n))
            else:
                # Pick evenly spaced indices
                selected_indices = sorted(set(
                    np.linspace(0, n - 1, args.max_steps, dtype=int).tolist()
                ))

            print(f"\n{'#'*60}")
            print(f"Clip: {clip_id}  ({n} steps, visualizing {len(selected_indices)})")
            print(f"{'#'*60}")
            for idx in selected_indices:
                step_file = step_files[idx]
                step_name = os.path.splitext(os.path.basename(step_file))[0]
                step_save_dir = os.path.join(args.save_dir, clip_id, step_name)
                print(f"\n--- {step_name} ---")
                load_and_visualize(step_file, step_save_dir, args.grid_thw, args.num_queries)
