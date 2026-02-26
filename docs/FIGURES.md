# CIFAR-10 CDM — Figure Index

> Generated: 2026-02-26
> All figures in `results/figures/`

---

## Primary Thesis Figures (recommended)

### Fig. 1: Training Convergence
- **File:** `training_curves.png`
- **Caption:** Training convergence of the Binary CDM across three random seeds (42, 123, 456) with λ=0.01. Left: AUROC vs epoch. Right: FPR@95%TPR vs epoch. All three seeds converge to AUROC ≥ 0.987 by epoch 19, demonstrating stable training and reproducibility. The yellow dashed line marks the early stopping point (epoch 19).
- **Placement:** §Results → Training Stability

### Fig. 2: Separation Loss Ablation
- **File:** `separation_loss_ablation_final.png` (single panel) or `sep_loss_dual.png` (dual panel with convergence speed)
- **Caption:** Effect of separation loss weight λ on CIFAR-10 OOD detection AUROC. The optimal zone λ ∈ [0.01, 0.05] achieves AUROC ≥ 0.985. Without separation loss (λ=0), AUROC drops to 0.8025.  λ=0.02 achieves the best single-run AUROC of 0.9911. The dual-panel variant additionally shows convergence speed (best epoch), revealing that strong λ values accelerate convergence while λ=0.1 requires 149 epochs.
- **Placement:** §Results → Separation Loss Analysis

### Fig. 3: ROC Curves
- **File:** `roc_curves_cifar10.png`
- **Caption:** ROC curves for Binary CDM OOD detection evaluated on five datasets (seed=42, K=50 MC trials, difference scoring). Within-CIFAR AUROC=0.9898; external datasets: CIFAR-100 (0.9696), FashionMNIST (0.9403), Textures (0.9281), SVHN (0.9049). All curves substantially above the random baseline.
- **Placement:** §Results → OOD Detection Performance

### Fig. 4: 3-Seed Reliability
- **File:** `three_seed_auroc.png`
- **Caption:** Per-seed AUROC for the Binary CDM (λ=0.01, K=100 MC trials). Seeds 42/123/456 achieve 0.9873/0.9886/0.9887 respectively, with mean 0.9882 ± 0.0006, demonstrating high reproducibility.
- **Placement:** §Results → Reproducibility

### Fig. 5: Score Distributions
- **File:** `score_distributions.png`
- **Caption:** Distribution of OOD scores for ID (airplane) and OOD (other CIFAR-10 classes) samples. The clear separation between the two distributions enables reliable OOD detection with low FPR.
- **Placement:** §Results → Score Analysis

---

## Ablation Figures (appendix or supplementary)

### Fig. A1: K-Ablation
- **File:** `k_ablation.png`
- **Caption:** Effect of the number of Monte Carlo trials K on AUROC and inference time. AUROC saturates beyond K=25 (0.9852); K=100 reaches 0.9869 at the cost of ~97× more inference time than K=1. Grey bars show wall-clock inference time.
- **Placement:** §Ablations → MC Trial Budget

### Fig. A2: Scoring Method Comparison
- **File:** `scoring_methods_full.png`
- **Caption:** Comparison of three OOD scoring methods across CIFAR-10 AUROC, FPR@95%TPR, and SVHN AUROC. The `difference` method achieves the best CIFAR-10 performance (0.9869), while `ratio` is slightly better on external SVHN (0.9606). The `id_error` baseline (0.7830) confirms that binary conditioning is essential.
- **Placement:** §Ablations → Scoring Strategy

### Fig. A3: Timestep Strategy Comparison
- **File:** `timestep_strategy_comparison.png`
- **Caption:** Comparison of timestep sampling strategies (uniform, mid_focus, stratified). All strategies perform within 0.4% AUROC of each other on CIFAR-10, indicating robustness to this design choice.
- **Placement:** §Ablations → Timestep Sampling

### Fig. A4: Error vs Timestep
- **File:** `error_vs_timestep.png`
- **Caption:** Per-timestep MSE reconstruction error for ID vs OOD samples. The separation is most pronounced at intermediate timesteps (t ∈ [200, 600]), motivating the mid_focus sampling strategy.
- **Placement:** §Analysis → Timestep Sensitivity

### Fig. A5: Confusion Matrix
- **File:** `confusion_matrix.png`
- **Caption:** Binary classification confusion matrix at the optimal threshold.
- **Placement:** §Supplementary

### Fig. A6: Calibration Curves
- **File:** `calibration_curves.png`
- **Caption:** Calibration plot showing predicted probability vs observed frequency.
- **Placement:** §Supplementary

---

## LaTeX Tables

| Table | File | Description |
|-------|------|-------------|
| Main results (3-seed) | `latex_tables/main_results_table.tex` | AUROC, FPR95 across datasets |
| Separation loss ablation | `latex_tables/separation_loss_table.tex` | All 6 λ values with AUROC + best epoch |
| K-ablation | `latex_tables/k_ablation_table.tex` | K vs AUROC, FPR95, inference time |
| Scoring methods | `latex_tables/scoring_method_table.tex` | difference vs ratio vs id_error |
| Timestep strategies | `latex_tables/timestep_strategy_table.tex` | uniform vs mid_focus vs stratified |

---

## Duplicate / Intermediate Files (can be ignored)

- `separation_loss_ablation.png` — early version, superseded by `_final`
- `separation_loss_ablation_updated.png` — intermediate version
- `scoring_method_comparison.png` — early version, superseded by `scoring_methods_full.png`
- `per_class_performance.png` — only relevant if per-class breakdown needed
- `score_distributions_all.png` — alternate version of score distributions
