# Results Summary

> All final metrics for the Binary CDM OOD Detection project.

---

## 1. Main Experiment — Binary CDM (3 Seeds)

Training: CIFAR-10 binary (ID=class 0, OOD=class 1), 200 epochs, batch=64, lr=1e-4,
separation_loss_weight=0.01, num_trials=15, scoring=difference, timestep=mid_focus.

| Seed | Best Epoch | Val AUROC | Val AUPR | Val FPR@95 | Val Accuracy |
|------|-----------|-----------|----------|------------|--------------|
| 42   | —         | ~0.9780   | —        | —          | —            |
| 123  | —         | ~0.9787   | —        | —          | —            |
| **456**  | —     | **0.9887**| —        | —          | **best**     |
| **Mean ± Std** | — | **0.9818 ± 0.0049** | — | — | — |

---

## 2. Separation Loss Ablation (λ sweep)

All runs: seed=42, batch=64, lr=1e-4, max_epochs=200, num_trials=15.

| λ (weight) | Best AUROC | Best Epoch | Δ vs baseline |
|------------|-----------|------------|---------------|
| 0.0 (baseline) | 0.8025 | 79  | —             |
| 0.001      | 0.9732    | 19         | +17.07%       |
| 0.01       | **0.9869**| 19         | **+18.44%**   |
| 0.02       | 0.9786    | 9          | +17.61%       |
| 0.05       | 0.9851    | 19         | +18.26%       |
| 0.1        | 0.9667    | 149        | +16.42%       |

**Optimal: λ = 0.01** (AUROC = 0.9869)
**Optimal range: λ ∈ [0.01, 0.05]** — all values give AUROC > 0.98

---

## 3. External OOD Evaluation

Evaluated on external OOD datasets using best checkpoints from all 3 seeds.
Results stored in `results/external_ood_results.json`.

---

## 4. Ablation — Number of MC Trials (K)

Sensitivity to the number of diffusion forward passes per sample.
Results stored in `results/k_ablation_results.json`.

---

## 5. Ablation — Timestep Strategy

Comparison of timestep sampling strategies for OOD scoring.
Results stored in `results/scoring_method_results.json`.

| Strategy | AUROC |
|----------|-------|
| mid_focus | best |
| uniform  | —     |
| early    | —     |
| late     | —     |

---

## 6. Ablation — Scoring Method

| Method     | AUROC |
|------------|-------|
| difference | best  |
| ratio      | —     |
