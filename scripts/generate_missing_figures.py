"""
generate_missing_figures.py
============================
Generates the 2 critical missing figures for the CIFAR-10 CDM thesis chapter:
  1. Training convergence curves (AUROC + Score Sep vs epoch, 3 seeds)
  2. ROC curves (within-CIFAR + external OOD)

Run from: /system/user/studentwork/mohammed/2025/diffusion_classifier_ood/
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import torch
    from sklearn.metrics import roc_curve, auc
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

ROOT    = Path(__file__).parent.parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BG   = '#0f1117'
GRID = '#333333'
W    = 'white'
BLUE = '#4fc3f7'
RED  = '#ef5350'
GRN  = '#66bb6a'
GOLD = '#ffd54f'
ORG  = '#ff9800'
PRP  = '#ab47bc'
TEAL = '#26c6da'
PINK = '#f06292'

def dark(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=W)
    for s in ax.spines.values(): s.set_edgecolor(GRID)
    ax.grid(axis='y', color=GRID, lw=0.8, ls='--', alpha=0.6)


def parse_training_logs():
    """Parse per-epoch metrics from wandb output.log files for all 3 seeds."""
    seed_data = {}
    
    for seed in [42, 123, 456]:
        # Find ALL runs' output.log and merge
        pattern = str(ROOT / f"results/seed{seed}/*/wandb/run-*/files/output.log")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  WARNING: No training log for seed {seed}")
            continue
        
        epoch_data = {}  # epoch -> (auroc, fpr95, score_sep)
        
        for log_file in files:
            with open(log_file) as f:
                for line in f:
                    m = re.search(
                        r'Epoch (\d+) \| AUROC: ([\d.]+) \| FPR95: ([\d.]+) \| '
                        r'Accuracy: ([\d.]+) \| Score sep: ([-\d.]+)',
                        line
                    )
                    if m:
                        ep = int(m.group(1))
                        epoch_data[ep] = (
                            float(m.group(2)),
                            float(m.group(3)),
                            float(m.group(5)),
                        )
        
        if epoch_data:
            sorted_eps = sorted(epoch_data.keys())
            seed_data[seed] = {
                'epochs':     np.array(sorted_eps),
                'aurocs':     np.array([epoch_data[e][0] for e in sorted_eps]),
                'fpr95s':     np.array([epoch_data[e][1] for e in sorted_eps]),
                'score_seps': np.array([epoch_data[e][2] for e in sorted_eps]),
            }
            print(f"  Seed {seed}: {len(sorted_eps)} epochs parsed "
                  f"(ep {min(sorted_eps)}-{max(sorted_eps)}, from {len(files)} log files)")
    
    return seed_data


# ============================================================================
# Figure 1: Training Convergence Curves (3 seeds)
# ============================================================================
def fig_training_curves(seed_data):
    MAX_EPOCH = 49  # Clip to early stopping window for cleaner plot
    
    # Clip data
    clipped = {}
    for seed, data in seed_data.items():
        mask = data['epochs'] <= MAX_EPOCH
        clipped[seed] = {k: v[mask] if isinstance(v, np.ndarray) else v
                         for k, v in data.items()}
    
    # ── 2-panel (clean thesis version): AUROC + FPR ──────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    
    seed_colors = {42: TEAL, 123: BLUE, 456: ORG}
    
    for ax, key, label, ylo, yhi in [
        (ax1, 'aurocs', 'AUROC', 0.0, 1.05),
        (ax2, 'fpr95s', 'FPR@95%TPR', -0.02, 1.05),
    ]:
        ax.set_facecolor(BG)
        for seed, data in sorted(clipped.items()):
            vals, eps = data[key], data['epochs']
            c = seed_colors[seed]
            ax.plot(eps, vals, color=c, lw=2.2, marker='o', ms=6,
                    label=f'Seed {seed}', alpha=0.9)
            best_idx = np.argmax(vals) if 'auroc' in key else np.argmin(vals)
            ax.scatter(eps[best_idx], vals[best_idx], color=c, s=140,
                      zorder=5, edgecolors=W, linewidths=1.5)
            ax.annotate(f'{vals[best_idx]:.4f}',
                       (eps[best_idx], vals[best_idx]),
                       textcoords='offset points', xytext=(8, 6),
                       fontsize=9, color=c, fontweight='bold')
        
        # Early-stopping annotation
        ax.axvline(19, color=GOLD, lw=1.2, ls=':', alpha=0.6)
        ax.text(20, yhi * 0.05, 'early stop\n(all 3 seeds)',
                fontsize=7, color=GOLD, alpha=0.8, va='bottom')
        
        ax.set_xlabel('Epoch', fontsize=11, color=W, labelpad=6)
        ax.set_ylabel(label, fontsize=11, color=W, labelpad=6)
        ax.set_title(f'{label} vs Epoch', fontsize=12, color=W, fontweight='bold', pad=10)
        ax.set_ylim(ylo, yhi)
        ax.legend(fontsize=9, facecolor='#1e1e2e', labelcolor=W, edgecolor=GRID)
        dark(ax)
    
    fig.suptitle('Training Convergence — Binary CDM (3 seeds, λ=0.01)',
                 fontsize=14, color=W, fontweight='bold', y=1.0)
    
    out = FIG_DIR / 'training_curves.png'
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  => Saved: {out.name}')


# ============================================================================
# Figure 2: ROC Curves (within-CIFAR + external OOD)
# ============================================================================
def fig_roc_curves():
    if not HAS_TORCH:
        print("  WARNING: torch not available, skipping ROC curves")
        return
    
    fig, ax = plt.subplots(figsize=(7, 6.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    
    raw_dir = ROOT / "results" / "raw_scores"
    
    # Load ID scores (always seed42)
    id_path = raw_dir / "seed42_cifar10_id_scores.pt"
    if not id_path.exists():
        print(f"  WARNING: {id_path} not found, skipping ROC curves")
        return
    
    id_scores = torch.load(id_path, map_location='cpu').numpy()
    
    # Datasets to plot (OOD)
    ood_datasets = [
        ('seed42_cifar10_ood_scores.pt', 'CIFAR-10 (within)', RED, 2.5),
        ('seed42_svhn_scores.pt', 'SVHN', GRN, 2),
        ('seed42_textures_scores.pt', 'Textures (DTD)', BLUE, 2),
        ('seed42_cifar100_scores.pt', 'CIFAR-100', ORG, 2),
        ('seed42_fashionmnist_scores.pt', 'FashionMNIST', PINK, 2),
    ]
    
    plotted = 0
    for fname, label, color, lw in ood_datasets:
        fpath = raw_dir / fname
        if not fpath.exists():
            print(f"  Skipping {fname} (not found)")
            continue
        
        ood_scores = torch.load(fpath, map_location='cpu').numpy()
        
        # Labels: 0=ID, 1=OOD; scores: higher = more likely OOD
        labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        scores = np.concatenate([id_scores, ood_scores])
        
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, lw=lw,
                label=f'{label}  (AUROC={roc_auc:.4f})')
        plotted += 1
    
    if plotted == 0:
        print("  WARNING: No OOD score files found, skipping ROC curves")
        plt.close()
        return
    
    # Random baseline
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls='--', alpha=0.6, label='Random (0.50)')
    
    # TPR=95% line
    ax.axhline(0.95, color=GOLD, lw=0.8, ls=':', alpha=0.7)
    ax.text(0.02, 0.955, 'TPR=0.95', fontsize=8, color=GOLD, alpha=0.9)
    
    ax.set_xlabel('False Positive Rate', fontsize=12, color=W, labelpad=8)
    ax.set_ylabel('True Positive Rate', fontsize=12, color=W, labelpad=8)
    ax.set_title('ROC Curves — Binary CDM OOD Detection\n(seed=42, K=50 MC trials)',
                 fontsize=13, color=W, fontweight='bold', pad=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9.5, facecolor='#1e1e2e', labelcolor=W, edgecolor=GRID,
              loc='lower right')
    ax.grid(color=GRID, lw=0.8, ls='--', alpha=0.6)
    dark(ax)
    
    out = FIG_DIR / 'roc_curves_cifar10.png'
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  => Saved: {out.name}')


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print(f"\nGenerating missing CIFAR-10 CDM figures\n{'='*50}")
    
    print("\n[1] Training convergence curves...")
    seed_data = parse_training_logs()
    if seed_data:
        fig_training_curves(seed_data)
    else:
        print("  ERROR: No training data found!")
    
    print("\n[2] ROC curves...")
    fig_roc_curves()
    
    print(f"\n{'='*50}")
    print("Done.")
