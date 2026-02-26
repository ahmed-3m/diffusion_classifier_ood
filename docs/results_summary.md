# Results Summary

> All final metrics for the Binary CDM OOD Detection project.
> Last verified: 2026-02-26 from checkpoints and JSON files.

---

## 1. Main Experiment — Binary CDM (3 Seeds, λ=0.01)

**Training config (identical for all seeds):**
```
model:             ConditionalUNet (35.7M params)
dataset:           CIFAR-10 binary (ID=class 0 airplane, OOD=classes 1-9)
image_size:        32×32×3
batch_size:        64 (effective 128 with grad accumulation ×2)
learning_rate:     1e-4 (cosine schedule, 5-epoch warmup)
weight_decay:      0.01
max_epochs:        200 (early stopping, patience=30 on val/auroc)
precision:         16-mixed
sep_loss_weight:   0.01
num_trials:        15 (MC diffusion passes for OOD scoring)
scoring_method:    difference
timestep_mode:     mid_focus
noise_schedule:    squaredcos_cap_v2 (1000 timesteps)
prediction_type:   epsilon
```

| Seed | Val AUROC | Best Epoch | Checkpoint Date |
|------|-----------|------------|-----------------|
| 42   | **0.9873** | 19        | 2026-02-19      |
| 123  | **0.9886** | 19        | 2026-02-19      |
| 456  | **0.9887** | 19        | 2026-02-19      |
| **Mean ± Std** | **0.9882 ± 0.0006** | — | — |

> 🏆 **Project best AUROC ever: 0.9911** achieved in the sep loss ablation (λ=0.02, seed 42, epoch 29).

---

## 2. Separation Loss Ablation (λ sweep, seed=42)

All runs: seed=42, batch=64, lr=1e-4, max_epochs=200, num_trials=15.

| λ (weight) | Best AUROC | Best Epoch | Δ vs baseline |
|------------|-----------|------------|---------------|
| 0.0 (baseline) | 0.8025 | 79 | —           |
| 0.001      | 0.9732    | 19         | +17.07%       |
| 0.01       | 0.9869    | 19         | +18.44%       |
| **0.02**   | **0.9911**| **29**     | **+18.86%**   |
| 0.05       | 0.9851    | 19         | +18.26%       |
| 0.1        | 0.9667    | 149        | +16.42%       |

**Key findings:**
- **Optimal: λ = 0.02** (AUROC = 0.9911) — new all-time best, exceeds seed runs (0.9887)
- **Robust range: λ ∈ [0.01, 0.05]** — all values give AUROC ≥ 0.9851
- Without separation loss (λ=0): AUROC drops to 0.8025 (−18.9% penalty)
- Too large λ=0.1: performance drops to 0.9667 and converges much later (epoch 149)

### λ=0.02 Multi-seed (in progress)
| Seed | AUROC | Epoch | Status |
|------|-------|-------|--------|
| 42   | 0.9911 | 29   | ✅ Done |
| 123  | 0.9840 | 29   | ✅ Done |
| 456  | —      | —    | ⏳ Not started yet |

---

## 3. External OOD Evaluation (seed 42 & 123 completed)

Evaluated on 7 datasets using trained checkpoints.
> ⚠️ Seed 456 external OOD results have suspicious threshold values (possible scoring unit mismatch) — use seeds 42 & 123 only until verified.

### Seed 42 Results (K=50, difference scoring)
| OOD Dataset | AUROC | FPR@95% | AUPR |
|-------------|-------|---------|------|
| CIFAR-10 (within) | **0.9898** | 4.7% | 0.9987 |
| Food-101 | **0.9927** | 3.4% | 0.9984 |
| CIFAR-100 | 0.9697 | 14.8% | 0.9965 |
| STL-10 | 0.9521 | 32.4% | 0.9906 |
| FashionMNIST | 0.9404 | 20.6% | 0.9916 |
| Textures (DTD) | 0.9284 | 30.1% | 0.9597 |
| SVHN | 0.9050 | 27.1% | 0.9938 |

### Seed 123 Results (K=50, difference scoring)
| OOD Dataset | AUROC | FPR@95% | AUPR |
|-------------|-------|---------|------|
| CIFAR-10 (within) | **0.9906** | 4.7% | 0.9989 |
| Food-101 | **0.9897** | 4.7% | 0.9977 |
| CIFAR-100 | 0.9647 | 15.7% | 0.9959 |
| STL-10 | 0.9512 | 33.9% | 0.9905 |
| SVHN | 0.9470 | 18.2% | 0.9971 |
| Textures (DTD) | 0.9310 | 31.7% | 0.9609 |
| FashionMNIST | 0.9287 | 23.8% | 0.9899 |

### Seeds 42+123 Average
| OOD Dataset | AUROC Mean | FPR@95% Mean |
|-------------|------------|--------------|
| CIFAR-10 (within) | **0.9902** | 4.7% |
| Food-101 | **0.9912** | 4.1% |
| CIFAR-100 | 0.9672 | 15.3% |
| STL-10 | 0.9516 | 33.2% |
| FashionMNIST | 0.9346 | 22.2% |
| Textures (DTD) | 0.9297 | 30.9% |
| SVHN | 0.9260 | 22.7% |

> General pattern: strong on semantically similar OOD (CIFAR-100, Food-101), weaker on
> distribution-shifted sets (SVHN, Textures). This is expected for diffusion-based methods.

---

## 4. Ablation — Number of MC Trials (K)

Source: `results/k_ablation_results.json` (verified)

| K | AUROC | FPR@95% | Time (s) | Time/sample (s) |
|---|-------|---------|----------|-----------------|
| 1 | 0.9100 | 40.8% | 97.9  | 0.010 |
| 5 | 0.9724 | 14.3% | 486.3 | 0.049 |
| 10 | 0.9819 | 9.4% | 972.9 | 0.097 |
| 25 | 0.9852 | 7.3% | 2431.8 | 0.243 |
| 50 | 0.9864 | 6.6% | 4861.1 | 0.486 |
| 100 | 0.9869 | 6.6% | 9723.6 | 0.972 |

> Recommended: K=15 (used in training evals) — good speed/accuracy tradeoff.

---

## 5. Ablation — Scoring Method

Source: `results/scoring_method_results.json` (verified)

| Method | CIFAR AUROC | CIFAR FPR95 | SVHN AUROC | SVHN FPR95 |
|--------|------------|-------------|------------|------------|
| **difference** | **0.9869** | **6.3%** | 0.9413 | 19.6% |
| ratio | 0.9862 | 6.6% | **0.9606** | **22.5%** |
| id_error | 0.7830 | 67.0% | 0.2023 | 96.5% |

---

## 6. Ablation — Timestep Strategy

Source: `results/timestep_strategy_results.json` (verified)

| Strategy | CIFAR AUROC | CIFAR FPR95 | SVHN AUROC |
|----------|------------|-------------|------------|
| **uniform** | **0.9887** | **5.4%** | **0.9544** |
| stratified | 0.9881 | 5.9% | 0.9498 |
| mid_focus | 0.9855 | 7.6% | 0.9380 |

---

## 7. Available Figures

| Figure | Content | Date |
|--------|---------|------|
| `separation_loss_ablation_final.png` | ⭐ Main ablation curve | Feb 25 |
| `sep_loss_dual.png` | Ablation curve + convergence speed bars | Feb 26 |
| `three_seed_auroc.png` | Bar chart: 3-seed AUROC reliability | Feb 26 |
| `training_curves.png` | AUROC+FPR95 vs epoch (3 seeds) | Feb 26 |
| `roc_curves_cifar10.png` | ROC curves for all 5 OOD datasets (seed 42) | Feb 26 |
| `scoring_methods_full.png` | Scoring method comparison (3 panels) | Feb 26 |
| `score_distributions_all.png` | Score distributions per seed | Feb 25 |
| `k_ablation.png` | K sensitivity curve | Feb 25 |
| `calibration_curves.png` | Calibration of OOD scores | Feb 25 |
| `timestep_strategy_comparison.png` | Timestep strategy comparison | Feb 25 |
| `confusion_matrix.png` | Confusion matrix on test set | Feb 25 |
| `scoring_method_comparison.png` | Scoring method comparison (older) | Feb 25 |
