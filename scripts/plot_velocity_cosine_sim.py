import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import glob

random.seed(42)
np.random.seed(42)

files = sorted(glob.glob("action/all_steps_vs_*.pt"))

all_cos_sims = []

for f in files:
    data = torch.load(f, map_location='cpu')
    n_windows = len(data)
    indices = random.sample(range(n_windows), min(20, n_windows))

    for w_idx in indices:
        steps = data[w_idx]
        cos_sims = []
        for i in range(7):
            vi = steps[i].float().flatten()
            vi1 = steps[i + 1].float().flatten()
            cos = torch.nn.functional.cosine_similarity(vi.unsqueeze(0), vi1.unsqueeze(0)).item()
            cos_sims.append(cos)
        all_cos_sims.append(cos_sims)

all_cos_sims = np.array(all_cos_sims)

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(7)
x_labels = [f"{i}\u2192{i+1}" for i in range(7)]

for i in range(all_cos_sims.shape[0]):
    ax.plot(x, all_cos_sims[i], color='steelblue', alpha=0.08, linewidth=0.8)

mean_curve = all_cos_sims.mean(axis=0)
ax.plot(x, mean_curve, color='red', linewidth=3, marker='o', markersize=8, label='Mean', zorder=10)

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=12)
ax.set_ylabel(r'Cosine Similarity $\cos(\mathbf{v}_i, \mathbf{v}_{i+1})$', fontsize=13)
ax.set_ylim(0.8, 1.005)
ax.set_yticks([0.8, 0.9, 1.0])
ax.legend(fontsize=13, loc='lower right')
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('action_cache_cosine_sim_plot.png', dpi=150)
plt.savefig('action_cache_cosine_sim_plot.pdf')
print("Done")
