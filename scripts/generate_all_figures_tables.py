"""
generate_all_figures_tables.py
================================
Generates ALL missing figures and fixes ALL broken LaTeX tables
for the CIFAR-10 CDM OOD thesis chapter.

Run from: /system/user/studentwork/mohammed/2025/diffusion_classifier_ood/
  conda run -p /system/apps/studentenv/mohammed/sdm python scripts/generate_all_figures_tables.py
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
FIG_DIR  = ROOT / "results" / "figures"
TEX_DIR  = ROOT / "results" / "latex_tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TEX_DIR.mkdir(parents=True, exist_ok=True)

# ── Style constants ─────────────────────────────────────────────────────────
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

def dark(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=W)
    for s in ax.spines.values(): s.set_edgecolor(GRID)
    ax.grid(axis='y', color=GRID, lw=0.8, ls='--', alpha=0.6)

# ── Verified data (from results_summary.md + experiments.md) ────────────────

# 1. Three-seed main results (λ=0.01, CIFAR-10 binary, within-CIFAR AUROC)
SEEDS  = [42,    123,    456]
AUROCS = [0.9873, 0.9886, 0.9887]
MEAN   = np.mean(AUROCS)   # 0.9882
STD    = np.std(AUROCS)    # 0.0006

# 2. Separation loss ablation (seed=42, CIFAR-10 within-CIFAR AUROC)
SEP_LAMBDAS = [0.0,    0.001,  0.01,   0.02,   0.05,   0.1]
SEP_AUROCS  = [0.8025, 0.9732, 0.9869, 0.9911, 0.9851, 0.9667]
SEP_EPOCHS  = [79,     19,     19,     29,     19,     149]

# 3. K-ablation (from k_ablation_results.json — verified)
K_VALS  = [1,      5,      10,     25,     50,     100]
K_AUROC = [0.9100, 0.9724, 0.9819, 0.9852, 0.9864, 0.9869]
K_FPR   = [0.408,  0.143,  0.094,  0.073,  0.066,  0.066]
K_TIME  = [97.9,   486.3,  972.9,  2431.8, 4861.1, 9723.6]

# 4. Scoring method comparison (from scoring_method_results.json — verified)
SCORE_METHODS = ['difference', 'ratio', 'id\_error']
SCORE_CIFAR   = [0.9869, 0.9862, 0.7830]
SCORE_FPR     = [0.063,  0.066,  0.670]
SCORE_SVHN    = [0.9413, 0.9606, 0.2023]


# ============================================================================
# Figure A: 3-Seed AUROC Bar Chart (main result)
# ============================================================================
def fig_three_seed_bar():
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    colors = [TEAL, BLUE, ORG]
    x = [0, 1, 2]
    for i, (s, a, c) in enumerate(zip(SEEDS, AUROCS, colors)):
        bar = ax.bar(i, a, color=c, alpha=0.85, width=0.5,
                     edgecolor=W, linewidth=0.8)
        ax.text(i, a + 0.0008, f'{a:.4f}', ha='center', va='bottom',
                fontsize=12, color=W, fontweight='bold')

    # Mean ± std line
    ax.axhline(MEAN, color=GOLD, lw=2, ls='--', alpha=0.9)
    ax.fill_between([-0.5, 2.5], MEAN - STD, MEAN + STD,
                    color=GOLD, alpha=0.12)
    ax.text(2.55, MEAN + 0.0001,
            f'Mean={MEAN:.4f}\n±{STD:.4f}',
            fontsize=9, color=GOLD, va='center')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Seed {s}' for s in SEEDS], fontsize=12, color=W)
    ax.set_ylabel('AUROC (CIFAR-10 Binary)', fontsize=12, color=W, labelpad=8)
    ax.set_title('CDM — 3-Seed Reliability (λ=0.01)',
                 fontsize=14, color=W, fontweight='bold', pad=12)
    ax.set_ylim(0.980, 0.993)
    ax.set_xlim(-0.5, 3.0)
    dark(ax)

    out = FIG_DIR / 'three_seed_auroc.png'
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'✓ {out.name}')


# ============================================================================
# Figure B: Separation Loss — AUROC + Convergence Epoch dual plot
# ============================================================================
def fig_sep_loss_dual():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)

    x = list(range(len(SEP_LAMBDAS)))
    xlabels = ['0.0', '0.001', '0.01', '0.02', '0.05', '0.1']
    colors = [RED, GRN, GOLD, ORG, BLUE, PRP]

    # Left: AUROC curve
    ax1.set_facecolor(BG)
    ax1.plot(x, SEP_AUROCS, color=TEAL, lw=2.5, zorder=2)
    peak = SEP_AUROCS.index(max(SEP_AUROCS))
    for i, (y, c) in enumerate(zip(SEP_AUROCS, colors)):
        ms = 220 if i == peak else 100
        ax1.scatter(i, y, color=c, s=ms, zorder=5, edgecolors=W, lw=1.5)
        ax1.annotate(f'{y:.4f}', (i, y + 0.005), ha='center', va='bottom',
                     fontsize=9, color=W, fontweight='bold' if i == peak else 'normal')
    ax1.axhline(SEP_AUROCS[0], color=RED, lw=1, ls='--', alpha=0.5)
    ax1.annotate('── baseline', (0.3, SEP_AUROCS[0] - 0.008),
                 fontsize=8, color=RED, va='top', fontweight='bold')
    ax1.axvspan(1.5, 3.5, alpha=0.08, color=GOLD)
    ax1.set_xticks(x); ax1.set_xticklabels(xlabels, fontsize=10, color=W)
    ax1.set_ylabel('AUROC', fontsize=12, color=W)
    ax1.set_title('AUROC vs λ', fontsize=13, color=W, fontweight='bold', pad=10)
    ax1.set_ylim(0.77, 1.01)
    for s in ax1.spines.values(): s.set_edgecolor(GRID)
    ax1.tick_params(colors=W)
    ax1.grid(axis='y', color=GRID, lw=0.8, ls='--', alpha=0.6)

    # Right: Best epoch bar (convergence speed)
    ax2.set_facecolor(BG)
    bars = ax2.bar(x, SEP_EPOCHS, color=colors, alpha=0.8, width=0.6, edgecolor=W, lw=0.8)
    for bar, e in zip(bars, SEP_EPOCHS):
        ax2.text(bar.get_x() + bar.get_width()/2, e + 1.5,
                 f'ep {e}', ha='center', va='bottom', fontsize=9, color=W)
    ax2.set_xticks(x); ax2.set_xticklabels(xlabels, fontsize=10, color=W)
    ax2.set_ylabel('Best Epoch (earlier = faster)', fontsize=11, color=W)
    ax2.set_title('Convergence Speed vs λ', fontsize=13, color=W, fontweight='bold', pad=10)
    ax2.set_ylim(0, 180)
    for s in ax2.spines.values(): s.set_edgecolor(GRID)
    ax2.tick_params(colors=W)
    ax2.grid(axis='y', color=GRID, lw=0.8, ls='--', alpha=0.6)

    ax1.set_xlabel('Separation Loss Weight λ', fontsize=12, color=W, labelpad=8)
    ax2.set_xlabel('Separation Loss Weight λ', fontsize=12, color=W, labelpad=8)

    out = FIG_DIR / 'sep_loss_dual.png'
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'✓ {out.name}')


# ============================================================================
# Figure C: Scoring Method Comparison
# ============================================================================
def fig_scoring_methods():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.patch.set_facecolor(BG)

    data = [
        ('CIFAR-10 AUROC ↑', SCORE_CIFAR, 0.70, 1.01),
        ('CIFAR-10 FPR@95TPR ↓', SCORE_FPR,   0.0,  0.80),
        ('SVHN AUROC ↑',     SCORE_SVHN,  0.0,  1.01),
    ]
    col = [TEAL, ORG, RED]

    for ax, (title, vals, ylo, yhi) in zip(axes, data):
        ax.set_facecolor(BG)
        bars = ax.bar([0,1,2], vals, color=col, alpha=0.85,
                      edgecolor=W, linewidth=0.8, width=0.55)
        best = vals.index(max(vals)) if 'AUROC' in title else vals.index(min(vals))
        for i, (bar, v) in enumerate(zip(bars, vals)):
            sign = '★ ' if i == best else ''
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + (yhi - ylo)*0.015,
                    f'{sign}{v:.4f}', ha='center', va='bottom',
                    fontsize=9, color=W,
                    fontweight='bold' if i == best else 'normal')
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(['difference', 'ratio', 'id\\_error'],
                           fontsize=9, color=W)
        ax.set_title(title, fontsize=11, color=W, fontweight='bold', pad=10)
        ax.set_ylim(ylo, yhi)
        for s in ax.spines.values(): s.set_edgecolor(GRID)
        ax.tick_params(colors=W)
        ax.grid(axis='y', color=GRID, lw=0.8, ls='--', alpha=0.6)

    out = FIG_DIR / 'scoring_methods_full.png'
    plt.tight_layout()
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'✓ {out.name}')


# ============================================================================
# LaTeX Table 1: Main Results (3-seed mean ± std)
# ============================================================================
def table_main_results():
    ext_ood_path = ROOT / "results" / "external_ood_results.json"

    # Try to load external OOD results if available
    ext = {}
    if ext_ood_path.exists():
        with open(ext_ood_path) as f:
            data = json.load(f)
        agg = data.get('aggregated', {})
        for ds, metrics in agg.items():
            ext[ds] = {
                'auroc': metrics.get('auroc_mean', None),
                'fpr95': metrics.get('fpr95_mean', None),
                'auroc_std': metrics.get('auroc_std', None),
            }

    def fmt(v, std=None, pct=True):
        if v is None: return '—'
        scale = 100 if pct else 1
        if std is not None:
            return f'{v*scale:.1f} $\\pm$ {std*scale:.1f}'
        return f'{v*scale:.1f}'

    rows = []
    # Within CIFAR (main result)
    rows.append(
        f"        CIFAR-10 (Airplane vs Others) & "
        f"{MEAN*100:.2f} $\\pm$ {STD*100:.2f} & "
        f"— & — \\\\"
    )

    # External OOD datasets (from evaluation if available)
    ds_display = {
        'svhn': 'SVHN',
        'cifar100': 'CIFAR-100',
        'textures': 'Textures (DTD)',
        'fashionmnist': 'FashionMNIST',
        'stl10': 'STL-10',
        'food101': 'Food-101',
        'places365': 'Places365',
    }
    for ds_key, ds_name in ds_display.items():
        m = ext.get(ds_key, {})
        auroc_v = m.get('auroc'); auroc_s = m.get('auroc_std')
        fpr_v   = m.get('fpr95')
        rows.append(
            f"        {ds_name} & "
            f"{fmt(auroc_v, auroc_s)} & "
            f"{fmt(fpr_v, pct=True)} & — \\\\"
        )

    tex = r"""% Main Results Table — CDM Binary OOD Detection
% Generated by scripts/generate_all_figures_tables.py
\begin{table}[htbp]
    \centering
    \caption{Binary CDM OOD detection performance (AUROC, FPR@95\%TPR).
             Within-dataset result is mean $\pm$ std over 3 seeds (42/123/456).
             External OOD is mean $\pm$ std evaluated on the seed-42 checkpoint
             (K=100 MC trials, difference scoring).}
    \label{tab:main-results}
    \begin{tabular}{lccc}
        \toprule
        OOD Dataset & AUROC (\%)$\uparrow$ & FPR95 (\%)$\downarrow$ & AUPR$\uparrow$ \\
        \midrule
""" + "\n".join(rows) + r"""
        \bottomrule
    \end{tabular}
\end{table}
"""
    out = TEX_DIR / 'main_results_table.tex'
    out.write_text(tex)
    print(f'✓ {out.name}  ({len(rows)} dataset rows)')


# ============================================================================
# LaTeX Table 2: Separation Loss Ablation — CIFAR AUROC (all 6 λ values)
# ============================================================================
def table_separation_loss():
    rows = []
    for lam, auroc, epoch in zip(SEP_LAMBDAS, SEP_AUROCS, SEP_EPOCHS):
        best_mark = r' \textbf{NEW BEST}' if lam == 0.02 else ''
        rows.append(
            f"        {lam} & {auroc*100:.2f} & {epoch} & — & —{best_mark} \\\\"
        )
    # Manually insert known SVHN from scoring_method_results for λ=0.01
    rows[2] = (
        f"        0.01 & {0.9869*100:.2f} & 19 & {0.9413*100:.1f} & {0.285*100:.1f} \\\\"
    )
    rows[3] = (
        r"        0.02 & \textbf{99.11} & 29 & — & — \\ % \textbf{Best AUROC}"
    )

    tex = r"""% Separation Loss Ablation Table
% Generated by scripts/generate_all_figures_tables.py
% Note: SVHN cols only populated for lambda=0.01 (from scoring_method_results.json).
%       Full external OOD evaluation pending completion.
\begin{table}[htbp]
    \centering
    \caption{Effect of separation loss weight $\lambda$ on CIFAR-10 OOD detection
             (seed=42, K=15 MC trials during training, K=100 for CIFAR final AUROC).
             Best epoch reports early stopping result.
             SVHN column only available for $\lambda=0.01$ (see ablation script).}
    \label{tab:separation-loss}
    \begin{tabular}{lcccc}
        \toprule
        $\lambda$ & CIFAR AUROC (\%) & Best Epoch & SVHN AUROC (\%) & SVHN FPR95 (\%) \\
        \midrule
""" + "\n".join(rows) + r"""
        \bottomrule
    \end{tabular}
\end{table}
"""
    out = TEX_DIR / 'separation_loss_table.tex'
    out.write_text(tex)
    print(f'✓ {out.name}')


# ============================================================================
# LaTeX Table 3: K-ablation (fully populated)
# ============================================================================
def table_k_ablation():
    rows = []
    for k, a, f, t in zip(K_VALS, K_AUROC, K_FPR, K_TIME):
        rec = r' $\leftarrow$ used in training' if k == 15 else ''
        mark = r'\textbf{' if k >= 25 else ''
        endmark = r'}' if k >= 25 else ''
        rows.append(
            f"        {k} & {mark}{a*100:.2f}{endmark} & "
            f"{mark}{f*100:.1f}{endmark} & {t:.1f}{rec} \\\\"
        )
    tex = r"""% K-Ablation Table (num_trials)
% Generated by scripts/generate_all_figures_tables.py
\begin{table}[htbp]
    \centering
    \caption{Effect of number of Monte Carlo trials $K$ on OOD scoring.
             Evaluated on CIFAR-10 binary test set using seed-42 checkpoint.
             Diminishing returns beyond $K=25$; $K=15$ used during training.}
    \label{tab:k-ablation}
    \begin{tabular}{lccc}
        \toprule
        $K$ (trials) & AUROC (\%)$\uparrow$ & FPR@95\%TPR$\downarrow$ & Time (s) \\
        \midrule
""" + "\n".join(rows) + r"""
        \bottomrule
    \end{tabular}
\end{table}
"""
    out = TEX_DIR / 'k_ablation_table.tex'
    out.write_text(tex)
    print(f'✓ {out.name}')


# ============================================================================
# LaTeX Table 4: Scoring Method Comparison (fully populated)
# ============================================================================
def table_scoring_methods():
    method_names = [r'difference', r'ratio', r'id\_error']
    rows = []
    for name, ca, cf, sa in zip(method_names, SCORE_CIFAR, SCORE_FPR, SCORE_SVHN):
        best_c = (name == 'difference')
        best_s = (name == 'ratio')
        tt   = r'\texttt{' + name + r'}'
        bold_open  = r'\textbf{' if best_c else ''
        bold_close = r'}'       if best_c else ''
        bold_sopen  = r'\textbf{' if best_s else ''
        bold_sclose = r'}'        if best_s else ''
        ca_str = f"{bold_open}{ca*100:.2f}{bold_close}"
        sa_str = f"{bold_sopen}{sa*100:.1f}{bold_sclose}"
        bs = r'\\'
        rows.append(
            f"        {tt} & {ca_str} & {cf*100:.1f} & {sa_str} {bs}"
        )
    tex = r"""% Scoring Method Comparison Table
% Generated by scripts/generate_all_figures_tables.py
\begin{table}[htbp]
    \centering
    \caption{Comparison of OOD scoring methods (seed-42 checkpoint, K=50 trials).
             \texttt{difference} and \texttt{ratio} perform similarly within CIFAR-10;
             \texttt{ratio} marginally better on external SVHN OOD.
             \texttt{id\_error} (ID-only scoring without class conditioning) is much worse,
             confirming binary conditioning is essential.}
    \label{tab:scoring-methods}
    \begin{tabular}{lccc}
        \toprule
        Scoring Method & CIFAR AUROC (\%)$\uparrow$ & CIFAR FPR95 (\%)$\downarrow$ & SVHN AUROC (\%)$\uparrow$ \\
        \midrule
""" + "\n".join(rows) + r"""
        \bottomrule
    \end{tabular}
\end{table}
"""
    out = TEX_DIR / 'scoring_method_table.tex'
    out.write_text(tex)
    print(f'✓ {out.name}')


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print(f"\nGenerating CIFAR-10 CDM figures & tables\n{'='*50}")

    print("\n[Figures]")
    fig_three_seed_bar()
    fig_sep_loss_dual()
    fig_scoring_methods()

    print("\n[LaTeX Tables]")
    table_main_results()
    table_separation_loss()
    table_k_ablation()
    table_scoring_methods()

    print(f"\n{'='*50}")
    print("Completed. Outputs:")
    print(f"  Figures → results/figures/")
    print(f"    three_seed_auroc.png     — 3-seed bar chart")
    print(f"    sep_loss_dual.png        — AUROC + convergence dual plot")
    print(f"    scoring_methods_full.png — 3-panel scoring comparison")
    print(f"  Tables  → results/latex_tables/")
    print(f"    main_results_table.tex       — fixed (was empty)")
    print(f"    separation_loss_table.tex    — fixed (all 6 λ, λ=0.02 added)")
    print(f"    k_ablation_table.tex         — verified & reformatted")
    print(f"    scoring_method_table.tex     — verified & reformatted")
