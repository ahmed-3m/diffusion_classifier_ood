import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Current results (0.02 still pending)
weights = [0.0,    0.001,  0.01,   0.05,   0.1]
aurocs  = [0.8025, 0.9732, 0.9869, 0.9851, 0.9667]
labels  = ['0.0', '0.001', '0.01', '0.05', '0.1']
x_pos   = [0, 1, 2, 3, 4]

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#0f1117')

# Line connecting points
ax.plot(x_pos, aurocs, color='#4fc3f7', linewidth=2.5, zorder=2)

# Scatter points
colors = ['#ef5350', '#66bb6a', '#ffd54f', '#42a5f5', '#ab47bc']
peak_idx = aurocs.index(max(aurocs))
for i, (x, y, lbl, c) in enumerate(zip(x_pos, aurocs, labels, colors)):
    ms = 200 if i == peak_idx else 100
    ax.scatter(x, y, color=c, s=ms, zorder=5, edgecolors='white', linewidths=1.5)
    offset = 0.009
    weight = 'bold' if i == peak_idx else 'normal'
    ax.annotate(f'{y:.4f}', (x, y + offset), ha='center', va='bottom',
                fontsize=11, color='white', fontweight=weight)

# Peak star annotation
ax.annotate('★ New Peak!', xy=(x_pos[peak_idx], aurocs[peak_idx]),
            xytext=(x_pos[peak_idx] + 0.7, aurocs[peak_idx] - 0.018),
            fontsize=10, color='#ffd54f', fontstyle='italic',
            arrowprops=dict(arrowstyle='->', color='#ffd54f', lw=1.5))

# Pending 0.02 marker
ax.scatter([2.5], [0.988], color='#888888', s=80, zorder=5,
           edgecolors='white', linewidths=1.2, alpha=0.5, marker='D')
ax.annotate('0.02\n(pending)', (2.5, 0.980), ha='center', va='top',
            fontsize=8.5, color='#888888', fontstyle='italic')

# Shaded optimal region
ax.axvspan(1.5, 2.5, alpha=0.10, color='#ffd54f')

# Baseline dashed line
ax.axhline(y=aurocs[0], color='#ef5350', linewidth=1,
           linestyle='--', alpha=0.5)
ax.annotate('No sep. loss baseline', (4.05, aurocs[0] + 0.003),
            fontsize=8, color='#ef5350', va='bottom')

# Axes
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=12, color='white')
ax.set_xlabel('Separation Loss Weight λ', fontsize=13, color='white', labelpad=10)
ax.set_ylabel('AUROC', fontsize=13, color='white', labelpad=10)
ax.set_title('Separation Loss Weight Ablation Study', fontsize=15,
             color='white', fontweight='bold', pad=15)
ax.set_ylim(0.77, 1.01)
ax.set_xlim(-0.3, 4.6)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#333')
ax.yaxis.set_tick_params(labelcolor='white', labelsize=11)
ax.grid(axis='y', color='#333', linewidth=0.8, linestyle='--', alpha=0.6)

out = 'results/figures/separation_loss_ablation_updated.png'
plt.tight_layout()
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {out}")
