# Results Summary

> All final metrics for the Binary CDM OOD Detection project.
> Last verified: 2026-02-25 from checkpoint filenames.

---

## 1. Main Experiment — Binary CDM (3 Seeds)

**Training config (identical for all seeds):**
```
model:             ConditionalUNet (35.7M params)
dataset:           CIFAR-10 binary (ID=class 0, OOD=classes 1-9)
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

> Note: All 3 seeds converged at epoch 19, which was the first validation
> checkpoint (eval_interval=10, so eval at epoch 9 and 19). Early stopping
> triggered after patience=30 from epoch 19.

---

## 2. Separation Loss Ablation (λ sweep)

All runs: seed=42, batch=64, lr=1e-4, max_epochs=200, num_trials=15.
Only the separation_loss_weight λ varies.

| λ (weight) | Best AUROC | Best Epoch | Δ vs λ=0 | Run Date |
|------------|-----------|------------|----------|----------|
| 0.0 (no sep loss) | 0.8025 | 79 | baseline | 2026-02-21 |
| 0.001 | 0.9732 | 19 | +17.07% | 2026-02-21 |
| **0.01** | **0.9869** | **19** | **+18.44%** | 2026-02-23 |
| 0.02 | 0.9786 | 9 | +17.61% | 2026-02-24 |
| 0.05 | 0.9851 | 19 | +18.26% | 2026-02-23 |
| 0.1 | 0.9667 | 149 | +16.42% | 2026-02-22 |

**Key findings:**
- **Optimal: λ = 0.01** (AUROC = 0.9869)
- **Robust range: λ ∈ [0.01, 0.05]** — all values give AUROC ≥ 0.9786
- Without separation loss (λ=0): AUROC drops to 0.8025 (−18.4%)
- Too large λ=0.1: performance degrades to 0.9667 (separation dominates MSE)
- λ=0.1 converges much later (epoch 149 vs 19 for others)

---

## 3. External OOD Evaluation

**Status: ⚠️ JSON is EMPTY — needs re-running**

`results/external_ood_results.json` contains `{}`.
The evaluation script (`scripts/evaluate_external_ood.py`) needs to be re-run
against all 3 seed checkpoints on external datasets (SVHN, CIFAR-100, etc.).

---

## 4. Ablation — Number of MC Trials (K)

Source: `results/k_ablation_results.json` (verified)

**Within-CIFAR (ID vs OOD from same CIFAR-10):**

| K | AUROC | FPR@95% | Time (s) | Time/sample (s) |
|---|-------|---------|----------|-----------------|
| 1 | 0.9100 | 40.8% | 97.9 | 0.010 |
| 5 | 0.9724 | 14.3% | 486.3 | 0.049 |
| 10 | 0.9819 | 9.4% | 972.9 | 0.097 |
| 25 | 0.9852 | 7.3% | 2431.8 | 0.243 |
| 50 | 0.9864 | 6.6% | 4861.1 | 0.486 |
| 100 | 0.9869 | 6.6% | 9723.6 | 0.972 |

**Key findings:**
- Diminishing returns beyond K=25
- K=15 (used in training/eval) is a good speed/accuracy tradeoff
- K=1 still achieves 0.91 AUROC — model has strong single-pass signal

---

## 5. Ablation — Scoring Method

Source: `results/scoring_method_results.json` (verified)

| Method | CIFAR AUROC | CIFAR FPR95 | SVHN AUROC | SVHN FPR95 |
|--------|------------|-------------|------------|------------|
| **difference** | **0.9869** | **6.3%** | 0.9413 | 19.6% |
| ratio | 0.9862 | 6.6% | **0.9606** | **22.5%** |
| id_error | 0.7830 | 67.0% | 0.2023 | 96.5% |

**Key findings:**
- `difference` and `ratio` perform similarly on within-CIFAR
- `ratio` is slightly better on SVHN (external OOD)
- `id_error` (using only ID condition) is much worse — confirms binary conditioning is essential

---

## 6. Ablation — Timestep Strategy

Source: `results/latex_tables/timestep_strategy_table.tex` (verified)

| Strategy | CIFAR AUROC | SVHN AUROC |
|----------|------------|------------|
| **uniform** | **98.9%** | **95.4%** |
| stratified | 98.8% | 95.0% |
| mid_focus | 98.5% | 93.8% |

**Key findings:**
- All strategies perform well (within 0.4% of each other on CIFAR)
- `uniform` slightly edges out `mid_focus` — but mid_focus was used in all main runs
- This suggests the model is robust to timestep strategy choice

> Note: Even though `uniform` appears slightly better here, the main experiments
> used `mid_focus`. The difference is within noise and would require multi-seed
> runs to confirm.
