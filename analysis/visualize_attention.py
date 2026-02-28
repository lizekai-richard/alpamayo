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


def _merge_layers_only(attn_weights_list):
    """Merge attention weights across layers only (average), keeping pre-merger spatial resolution.
    
    Args:
        attn_weights_list: List of [num_chunks, num_heads, chunk_size, chunk_size] tensors
                          (one per layer), where chunk_size = h_pre * w_pre (pre-merger).
    
    Returns:
        Averaged attention: [num_chunks, num_heads, chunk_size, chunk_size]
                           (pre-merger resolution, layers averaged).
    """
    if isinstance(attn_weights_list, list):
        # Stack and average over layers
        # Each element in list is [num_chunks, num_heads, chunk_size, chunk_size] (pre-merger)
        attn_weights = torch.stack(attn_weights_list, dim=0)  # [num_layers, N, H, S, S]
        attn_weights = attn_weights.mean(dim=0)  # [N, H, S, S] - averaged over layers
        return attn_weights
    else:
        # Already a single tensor
        return attn_weights_list


def _merge_colsum(colsums_list, image_grid_thw, spatial_merge_size: int = 2, apply_spatial_merge: bool = False):
    """Merge colsums across layers and optionally apply spatial merging.
    
    Args:
        colsums_list: List of [B, H, L] tensors (one per layer), where L is pre-merger sequence length.
        image_grid_thw: [num_images, 3] with (T, H, W) per image (pre-merger).
        spatial_merge_size: Merger size (default 2).
        apply_spatial_merge: If False, only merge layers, keep pre-merger resolution.
    
    Returns:
        If apply_spatial_merge=True: [B, L_post] where L_post = L / m²
        If apply_spatial_merge=False: [B, L] (pre-merger resolution)
    """
    if isinstance(colsums_list, list):
        # Stack and average over layers and heads
        # Each element is [B, H, L] (pre-merger)
        colsums = torch.stack(colsums_list, dim=0)  # [num_layers, B, H, L]
        colsums = colsums.mean(dim=(0, 2))  # [B, L] - averaged over layers and heads
    else:
        # Already a single tensor
        colsums = colsums_list
    
    # Convert to float32 for numpy compatibility (handles BFloat16, etc.)
    colsums = colsums.float().cpu()
    
    if not apply_spatial_merge:
        return colsums
    
    # Apply spatial merging: sum every m² consecutive tokens
    m = spatial_merge_size
    m2 = m * m
    B, L = colsums.shape
    
    if L % m2 != 0:
        print(f"Warning: colsum length {L} is not divisible by {m2}, assuming already merged.")
        return colsums
    
    # Reshape and sum: [B, L] -> [B, L//m², m²] -> sum -> [B, L//m²]
    colsums = colsums.view(B, -1, m2).sum(dim=2)
    return colsums


def _merge_attn_weights(attn_weights_list, spatial_merge_size: int = 2, apply_spatial_merge: bool = True):
    """Merge pre-merger attention weights to post-merger resolution.

    PatchMerger groups every consecutive m² tokens. To match, we:
      - Key dim: sum every consecutive m² columns (total attention received).
      - Query dim: mean every consecutive m² rows (average attention given).

    Args:
        attn_weights_list: Either:
                          - List of [num_chunks, num_heads, chunk_size, chunk_size] tensors
                            (one per layer), where chunk_size = h_pre * w_pre (pre-merger).
                            This is the format saved by alpamayo_r1_compile.py (unmerged).
                          - Single [num_chunks, num_heads, chunk_size, chunk_size] tensor
                            (pre-merger format, will be merged).
        spatial_merge_size: Merger size (default 2).
        apply_spatial_merge: If False, only merge layers (average), keep pre-merger resolution.

    Returns:
        If apply_spatial_merge=True: [num_chunks, num_heads, post_size, post_size]
        If apply_spatial_merge=False: [num_chunks, num_heads, chunk_size, chunk_size]
    """
    if isinstance(attn_weights_list, list):
        # Stack and average over layers
        # Each element in list is [num_chunks, num_heads, chunk_size, chunk_size] (pre-merger)
        attn_weights = torch.stack(attn_weights_list, dim=0)  # [num_layers, N, H, S, S]
        attn_weights = attn_weights.mean(dim=0)  # [N, H, S, S] - averaged over layers
    else:
        # Already a single tensor (pre-merger format)
        attn_weights = attn_weights_list
    
    if not apply_spatial_merge:
        # Only merge layers, keep pre-merger resolution
        return attn_weights
    
    m = spatial_merge_size
    m2 = m * m
    N, H, S, _ = attn_weights.shape
    
    # Check if already merged (size should not be divisible by m² if already merged)
    # Pre-merger tensors should always have S divisible by m²
    if S % m2 != 0:
        # Likely already merged, return as-is
        print(f"Warning: chunk_size {S} is not divisible by {m2}, assuming already merged.")
        return attn_weights
    
    # Apply spatial merging: merge every m² consecutive tokens
    post = S // m2
    # Merge key dim: [N, H, S, post, m²] -> sum -> [N, H, S, post]
    merged_k = attn_weights.view(N, H, S, post, m2).sum(dim=-1)
    # Merge query dim: [N, H, post, m², post] -> mean -> [N, H, post, post]
    merged = merged_k.view(N, H, post, m2, post).mean(dim=3)
    return merged


def visualize_attention_maps(
    attn_weights: torch.Tensor | list[torch.Tensor],
    image_grid_thw: torch.Tensor,
    save_dir: str = "attn_vis",
    num_example_queries: int = 5,
    spatial_merge_size: int = 2,
    apply_spatial_merge: bool = False,
    colsums: torch.Tensor | list[torch.Tensor] | None = None,
):
    """Visualize attention maps from the vision encoder's last layer.

    Args:
        attn_weights: Either:
                      - [num_chunks, num_heads, chunk_size, chunk_size] (already processed)
                      - List of [num_chunks, num_heads, chunk_size, chunk_size] (pre-merger, one per layer)
        image_grid_thw: [num_images, 3] with (T, H, W) per image.
                        H, W are pre-merger grid dims.
        save_dir: directory to save figures.
        num_example_queries: number of query tokens to show spatial maps for.
        spatial_merge_size: merger size (default 2). Used to convert
                           pre-merger H,W to post-merger h,w (only if apply_spatial_merge=True).
        apply_spatial_merge: If False, keep pre-merger resolution (layers are still merged/averaged).
        colsums: Optional colsums tensor or list. If provided, will be used for visualization.
                Format: [B, L] or list of [B, H, L] (one per layer).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Merge attention weights if needed (list format from saved tensors - unmerged format)
    # The saved tensors are a list of [num_chunks, num_heads, chunk_size, chunk_size] per layer
    # where chunk_size is pre-merger. We merge across layers (average) but optionally skip spatial merging.
    if isinstance(attn_weights, list):
        print(f"Merging attention weights from {len(attn_weights)} layers...")
        attn_weights = _merge_attn_weights(attn_weights, spatial_merge_size, apply_spatial_merge=apply_spatial_merge)
        print(f"Merged attention weights shape: {attn_weights.shape} (spatial_merge={apply_spatial_merge})")
    
    attn_weights = attn_weights.float().cpu()
    num_heads = attn_weights.shape[1]
    
    # Merge colsums if provided
    merged_colsums = None
    if colsums is not None:
        if isinstance(colsums, list):
            print(f"Merging colsums from {len(colsums)} layers...")
            merged_colsums = _merge_colsum(colsums, image_grid_thw, spatial_merge_size, apply_spatial_merge)
            print(f"Merged colsums shape: {merged_colsums.shape} (spatial_merge={apply_spatial_merge})")
        else:
            merged_colsums = colsums.float().cpu()
        
        # Ensure colsums are in float32 for numpy conversion
        if merged_colsums is not None:
            merged_colsums = merged_colsums.float().cpu()

    # Head-averaged attention
    attn_avg = attn_weights.mean(dim=1)  # [num_chunks, chunk_size, chunk_size]

    # Precompute token offsets for each image if colsums are provided
    colsum_offsets = None
    if merged_colsums is not None:
        # Calculate cumulative token offsets per image
        colsum_offsets = [0]
        for img_i in range(image_grid_thw.shape[0]):
            t, h_pre, w_pre = image_grid_thw[img_i].tolist()
            if apply_spatial_merge:
                tokens_per_frame = (h_pre // spatial_merge_size) * (w_pre // spatial_merge_size)
            else:
                tokens_per_frame = h_pre * w_pre
            tokens_per_image = t * tokens_per_frame
            colsum_offsets.append(colsum_offsets[-1] + tokens_per_image)

    chunk_idx = 0
    for img_i in range(image_grid_thw.shape[0]):
        t, h_pre, w_pre = image_grid_thw[img_i].tolist()
        if apply_spatial_merge:
            h = h_pre // spatial_merge_size
            w = w_pre // spatial_merge_size
        else:
            # Use pre-merger dimensions
            h = h_pre
            w = w_pre
        tokens_per_frame = h * w
        
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
            # Use provided colsums if available, otherwise compute from attention
            if merged_colsums is not None and colsum_offsets is not None:
                # Extract colsum for this image/frame
                # merged_colsums is [B, L] where B=batch_size (usually 1), L=total_tokens
                # Extract the portion for this frame
                img_offset = colsum_offsets[img_i]
                frame_offset = img_offset + t_i * tokens_per_frame
                col_sum = merged_colsums[0, frame_offset:frame_offset + chunk_size].float().cpu().numpy()
                if len(col_sum) != chunk_size:
                    # Fallback: compute from attention if sizes don't match
                    print(f"Warning: colsum size {len(col_sum)} != chunk_size {chunk_size}, using attention-based colsum")
                    col_sum = attn.sum(axis=0)
            else:
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

            # ── 6. All heads: 4x4 grid of full attention matrices ──
            n_cols = 4
            n_rows = 4
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
            # Flatten axes to 1D array for easier indexing
            axes = np.array(axes).flatten()
            
            # Find global min/max for consistent colormap across all heads
            all_head_matrices = [attn_per_head[hi] for hi in range(num_heads)]
            vmin = min(arr.min() for arr in all_head_matrices)
            vmax = max(arr.max() for arr in all_head_matrices)
            
            for hi in range(num_heads):
                if hi < len(axes):
                    head_matrix = attn_per_head[hi]  # [chunk_size, chunk_size]
                    im = axes[hi].imshow(head_matrix, cmap="viridis", aspect="equal", 
                                         interpolation="nearest", vmin=vmin, vmax=vmax)
                    axes[hi].set_title(f"Head {hi}", fontsize=10)
                    axes[hi].set_xlabel("Key", fontsize=8)
                    axes[hi].set_ylabel("Query", fontsize=8)
                    axes[hi].tick_params(labelsize=6)
            
            # Hide unused subplots (if num_heads < 16)
            for hi in range(num_heads, len(axes)):
                axes[hi].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}_all_heads_matrix.png"), dpi=150, bbox_inches='tight')
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


def load_and_visualize(
    attn_path: str,
    save_dir: str,
    grid_thw_str: str | None = None,
    num_queries: int = 5,
    spatial_merge_size: int = 2,
    colsums_path: str | None = None,
):
    """Load a single .pt file and run visualization for both merged and non-merged versions.
    
    Args:
        attn_path: Path to attention weights file. Can be:
                   - List of [num_chunks, num_heads, chunk_size, chunk_size] tensors (one per layer)
                   - Single [num_chunks, num_heads, chunk_size, chunk_size] tensor (already merged)
        save_dir: Base directory to save visualizations. Will create 'non_merged' and 'merged' subfolders.
        grid_thw_str: Grid dimensions as 'T,H,W' or 'T1,H1,W1;T2,H2,W2'.
        num_queries: Number of example queries for spatial maps.
        spatial_merge_size: Spatial merge size (default 2).
        colsums_path: Optional path to colsums file (for reference, not used in visualization).
    """
    data = torch.load(attn_path, map_location="cpu", weights_only=True)
    
    # Handle different data formats
    if isinstance(data, dict):
        attn_w = data["attn_weights"]
        grid_thw = data.get("image_grid_thw")
    elif isinstance(data, list):
        # Check if it's a list of tensors (one per layer) or a list with a single tensor
        if len(data) > 0 and isinstance(data[0], torch.Tensor):
            # Check if first element is a 4D tensor (likely attention weights)
            if len(data[0].shape) == 4:
                # List of [num_chunks, num_heads, chunk_size, chunk_size] per layer (unmerged format)
                attn_w = data
                print(f"Loaded unmerged attention weights: {len(attn_w)} layers")
                print(f"  Layer 0 shape: {attn_w[0].shape}")
            else:
                # Single tensor in a list
                attn_w = data[0]
                print(f"Loaded single tensor: {attn_w.shape}")
        else:
            attn_w = data[0] if len(data) > 0 else data
        grid_thw = None
    else:
        attn_w = data
        print(f"Loaded single tensor (not a list): {attn_w.shape}")
        grid_thw = None

    # Load colsums if provided
    colsums_data = None
    if colsums_path and os.path.exists(colsums_path):
        colsums_data = torch.load(colsums_path, map_location="cpu", weights_only=True)
        if isinstance(colsums_data, list):
            print(f"Loaded colsums: {len(colsums_data)} layers, shapes: {[c.shape for c in colsums_data]}")
        else:
            print(f"Loaded colsums: shape {colsums_data.shape}")

    if grid_thw_str:
        rows = grid_thw_str.split(";")
        grid_thw = torch.tensor([[int(x) for x in r.split(",")] for r in rows])

    if grid_thw is None:
        # Infer from attention weights shape
        if isinstance(attn_w, list):
            # Use first layer to infer shape
            sample = attn_w[0]
            num_chunks, _, cs, _ = sample.shape
        else:
            num_chunks, _, cs, _ = attn_w.shape
        # Assume square grid for inference
        side = int(cs ** 0.5)
        grid_thw = torch.tensor([[num_chunks, side, side]])
        print(f"Inferred grid_thw = {grid_thw.tolist()} from attention weights shape")

    # Create both non-merged and merged versions
    # Non-merged version (layers averaged, but spatial resolution kept)
    non_merged_dir = os.path.join(save_dir, "non_merged")
    print(f"\n{'='*60}")
    print(f"Generating NON-MERGED visualizations...")
    print(f"{'='*60}")
    visualize_attention_maps(
        attn_w,
        grid_thw,
        save_dir=non_merged_dir,
        num_example_queries=num_queries,
        spatial_merge_size=spatial_merge_size,
        apply_spatial_merge=False,
        colsums=colsums_data,
    )
    
    # Merged version (layers averaged and spatially merged)
    merged_dir = os.path.join(save_dir, "merged")
    print(f"\n{'='*60}")
    print(f"Generating MERGED visualizations...")
    print(f"{'='*60}")
    visualize_attention_maps(
        attn_w,
        grid_thw,
        save_dir=merged_dir,
        num_example_queries=num_queries,
        spatial_merge_size=spatial_merge_size,
        apply_spatial_merge=True,
        colsums=colsums_data,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize vision encoder attention maps")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--attn_path", type=str,
                       help="Path to a single saved attention weights (.pt).")
    group.add_argument("--attn_dir", type=str,
                       help="Directory containing clip subfolders, each with step subfolders. "
                            "E.g. clips/<clip_id>/step0/attn_weights.pt")
    parser.add_argument("--grid_thw", type=str, default=None,
                        help="Grid dims as 'T,H,W' (e.g. '1,28,28'). "
                             "Can also be 'T1,H1,W1;T2,H2,W2' for multiple images.")
    parser.add_argument("--save_dir", type=str, default="attn_vis")
    parser.add_argument("--num_queries", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=6,
                        help="Max number of steps to visualize per clip (evenly sampled).")
    parser.add_argument("--spatial_merge_size", type=int, default=2,
                        help="Spatial merge size (default 2).")
    parser.add_argument("--colsums_path", type=str, default=None,
                        help="Optional path to colsums file (for reference).")
    args = parser.parse_args()

    if args.attn_path:
        load_and_visualize(
            args.attn_path,
            args.save_dir,
            args.grid_thw,
            args.num_queries,
            args.spatial_merge_size,
            args.colsums_path,
        )
    else:
        import glob as glob_mod
        import re

        # Scan attn_dir for clip subfolders, then step subfolders
        # Structure: clips/<clip_id>/step*/attn_weights.pt
        clip_dirs = sorted([
            d for d in os.listdir(args.attn_dir)
            if os.path.isdir(os.path.join(args.attn_dir, d))
        ])
        print(f"Found {len(clip_dirs)} clip folders in {args.attn_dir}")
        
        for clip_id in clip_dirs:
            clip_path = os.path.join(args.attn_dir, clip_id)
            
            # Find all step subdirectories
            step_dirs = sorted([
                d for d in os.listdir(clip_path)
                if os.path.isdir(os.path.join(clip_path, d)) and d.startswith("step")
            ], key=lambda x: int(re.search(r"step(\d+)", x).group(1)) if re.search(r"step(\d+)", x) else 0)
            
            if not step_dirs:
                print(f"Warning: No step directories found in {clip_path}")
                continue
            
            # Collect all step files
            step_files = []
            for step_dir in step_dirs:
                step_path = os.path.join(clip_path, step_dir)
                attn_file = os.path.join(step_path, "attn_weights.pt")
                if os.path.exists(attn_file):
                    step_files.append((step_dir, attn_file))
            
            if not step_files:
                print(f"Warning: No attn_weights.pt files found in {clip_path}")
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
                step_dir, step_file = step_files[idx]
                step_save_dir = os.path.join(args.save_dir, clip_id, step_dir)
                
                # Try to find corresponding colsums file in the same step directory
                step_path = os.path.dirname(step_file)
                colsums_file = os.path.join(step_path, "colsums.pt")
                colsums_path = colsums_file if os.path.exists(colsums_file) else None
                
                print(f"\n--- {step_dir} ---")
                load_and_visualize(
                    step_file,
                    step_save_dir,
                    args.grid_thw,
                    args.num_queries,
                    args.spatial_merge_size,
                    colsums_path,
                )
