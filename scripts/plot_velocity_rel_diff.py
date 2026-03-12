import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import glob

random.seed(42)
np.random.seed(42)

files = sorted(glob.glob("action/all_steps_vs_*.pt"))
print(f"Found {len(files)} clips")

all_rel_diffs = []  # each entry: shape [7] (7 pairs for 8 steps)

for f in files:
    data = torch.load(f, map_location='cpu')  # list of windows, each list of 8 tensors [6,64,2]
    n_windows = len(data)
    # randomly pick 20 windows (or all if < 20)
    indices = random.sample(range(n_windows), min(20, n_windows))

    for w_idx in indices:
        steps = data[w_idx]  # list of 8 tensors [6,64,2]

        # relative difference: ||v_{i+1} - v_i|| / ||v_i|| (%)
        rel_diffs = []
        for i in range(7):
            vi = steps[i].float()
            vi1 = steps[i + 1].float()
            diff_norm = torch.norm(vi1 - vi).item()
            vi_norm = torch.norm(vi).item()
            if vi_norm > 1e-8:
                rel_diffs.append(diff_norm / vi_norm * 100)
            else:
                rel_diffs.append(0.0)
        all_rel_diffs.append(rel_diffs)

all_rel_diffs = np.array(all_rel_diffs)  # [N, 7]
print(f"Total samples: {all_rel_diffs.shape[0]}")
print(f"Mean per step pair: {all_rel_diffs.mean(axis=0)}")

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(7)
x_labels = [f"{i}\u2192{i+1}" for i in range(7)]

# Individual lines (light blue, thin)
for i in range(all_rel_diffs.shape[0]):
    ax.plot(x, all_rel_diffs[i], color='steelblue', alpha=0.08, linewidth=0.8)

# Mean curve (red, thick, with markers)
mean_curve = all_rel_diffs.mean(axis=0)
ax.plot(x, mean_curve, color='red', linewidth=3, marker='o', markersize=8, label='Mean', zorder=10)

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=12)
ax.set_ylabel(r'Rel. Diff. $\|\mathbf{v}_{i+1} - \mathbf{v}_i\| / \|\mathbf{v}_i\|$ (%)', fontsize=13)
ax.set_ylim(bottom=0)
ax.legend(fontsize=13, loc='upper right')
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('action_cache_diff_plot.png', dpi=150)
plt.savefig('action_cache_diff_plot.pdf')
print("Saved to action_cache_diff_plot.png and .pdf")
