# Experiment Log

Full chronological log of all experiments run for this thesis.

---

## Model Architecture Summary

- **ConditionalUNet**: 35.7M parameters (all trainable)
- Block channels: (128, 256, 256, 256)
- 2 layers per block
- Attention at 3rd level (AttnDownBlock2D/AttnUpBlock2D)
- 2-class embedding (binary: ID=0, OOD=1)
- Noise schedule: squaredcos_cap_v2, 1000 timesteps, epsilon prediction

---

## Experiment 1 — Main Training (3 Seeds)

**Goal:** Establish baseline performance of the Binary CDM with separation loss.

**Config:**
```
model:               ConditionalUNet (35.7M params)
dataset:             CIFAR-10 binary (ID=class 0, OOD=classes 1-9)
image_size:          32×32×3
batch_size:          64 (effective 128 with grad accumulation ×2)
learning_rate:       1e-4 (cosine schedule, 5-epoch warmup)
weight_decay:        0.01
max_epochs:          200 (early stopping patience=30 on val/auroc)
precision:           16-mixed
sep_loss_weight:     0.01
num_trials:          15
scoring_method:      difference
timestep_mode:       mid_focus
eval_interval:       10 (validate every 10 epochs)
seeds:               42 / 123 / 456
```

**Results (verified from checkpoint filenames):**
| Seed | Val AUROC | Best Epoch | Checkpoint |
|------|-----------|------------|------------|
| 42   | 0.9873    | 19         | results/seed42/auroc=0.9873.ckpt |
| 123  | 0.9886    | 19         | results/seed123/auroc=0.9886.ckpt |
| 456  | 0.9887    | 19         | results/seed456/auroc=0.9887.ckpt |
| **Mean ± Std** | **0.9882 ± 0.0006** | — | — |

**Hardware:**
- Seed 42:  student10, GPU (P40 24GB or V100 16GB)
- Seed 123: student10, GPU (P40 24GB or V100 16GB)
- Seed 456: student10, GPU 1 (V100 16GB) — resumed after OOM on P40

**Issues encountered:**
- Seed 456 initially crashed OOM on P40 (24GB) due to concurrent processes
- Moved to V100 (16GB), resumed from checkpoint — completed successfully
- Multiple zombie processes from previous runs caused GPU contention — killed manually

---

## Experiment 2 — External OOD Evaluation

**Goal:** Evaluate trained checkpoints on external OOD datasets (SVHN, CIFAR-100, etc.)

**Script:** `scripts/evaluate_external_ood.py`
**Config:** num_trials=50, batch_size=64, scoring=difference, timestep=mid_focus

**Status: ✅ COMPLETE for seeds 42 and 123** (2026-02-26)

| Seed | CIFAR-10 within | SVHN | CIFAR-100 | Food-101 | Textures | FashionMNIST | STL-10 |
|------|-----------------|------|-----------|---------|---------|-----------|---------|
| 42   | 0.9898 | 0.9050 | 0.9697 | **0.9927** | 0.9284 | 0.9404 | 0.9521 |
| 123  | 0.9906 | 0.9470 | 0.9647 | **0.9897** | 0.9310 | 0.9287 | 0.9512 |
| **Mean** | **0.9902** | **0.9260** | **0.9672** | **0.9912** | **0.9297** | **0.9346** | **0.9516** |

> ⚠️ Seed 456 external results have suspicious threshold values (likely scoring unit mismatch)
> and need re-verification. Seeds 42 and 123 results are confirmed valid.

---

## Experiment 3 — Ablation Studies (K, Timestep, Scoring Method)

**Goal:** Study sensitivity to hyperparameters using seed42 checkpoint.

**Script:** `scripts/run_ablations.py`
**Hardware:** student10

### 3a. K Ablation (num_trials)
**Swept:** K = {1, 5, 10, 25, 50, 100}
**Results:** `results/k_ablation_results.json` ✅ (verified, contains real data)

Key: Diminishing returns beyond K=25. K=1 still gives AUROC=0.91.

### 3b. Scoring Method
**Compared:** difference, ratio, id_error
**Results:** `results/scoring_method_results.json` ✅ (verified)

Key: difference and ratio nearly identical; id_error much worse (0.78 AUROC).

### 3c. Timestep Strategy
**Compared:** uniform, mid_focus, stratified
**Results:** `results/latex_tables/timestep_strategy_table.tex` ✅

Key: All within 0.4% of each other. Uniform slightly best (98.9%).

---

## Experiment 4 — Separation Loss Ablation

**Goal:** Quantify the effect of the separation loss coefficient λ.

**Script:** `scripts/train.py` (repeated 6 times with different weights)
**Config:** Identical to Experiment 1 except separation_loss_weight varies and seed=42 fixed.

### Timeline

| Date | Event |
|------|-------|
| Feb 17–19 | First attempts at sep_0.0 (slow, competing processes on student10) |
| Feb 20 | sep_0.0 running properly on student10 GPU 3 |
| Feb 21 05:06 | **sep_0.0 DONE** — AUROC=0.8025 (epoch 79, full 200 epochs) |
| Feb 21 22:03 | sep_0.001 started automatically (student10 sequential for-loop) |
| Feb 22 ~14:00 | sep_0.05 started on student06 GPU0, sep_0.1 on student06 GPU1 |
| Feb 22 ~14:00 | DDP attempt on student06 — crashed with NCCL timeout after 30 min |
| Feb 22 | Fixed: `num_sanity_val_steps=0` added to Trainer; reverted to single-GPU |
| Feb 22 ~00:53 | **sep_0.001 DONE** — AUROC=0.9732 (epoch 19) |
| Feb 23 02:04 | **sep_0.05 DONE** — AUROC=0.9851 (epoch 19) |
| Feb 23 11:35 | **sep_0.1 DONE** — AUROC=0.9667 (epoch 149) |
| Feb 23 18:52 | sep_0.01 started on 32GB GPU server |
| Feb 23 21:41 | **sep_0.01 DONE** — AUROC=0.9869 (epoch 19, new peak!) |
| Feb 24 22:57 | sep_0.02 started |
| Feb 25 ~00:30 | **sep_0.02 first checkpoint** — AUROC=0.9911 (epoch 29) — NEW ALL-TIME BEST |
| Feb 25 ~01:30 | sep_0.02 training complete (epoch 200) — final best stays 0.9911 |

### Final Results (verified from checkpoint filenames)

| λ       | AUROC  | Best Epoch | Checkpoint Dir Timestamp |
|---------|--------|------------|--------------------------|
| 0.0     | 0.8025 | 79         | 2026-02-21_05-04-31 |
| 0.001   | 0.9732 | 19         | 2026-02-21_22-03-43 |
| 0.01    | 0.9869 | 19         | 2026-02-23_18-52-30 |
| **0.02**| **0.9911** | **29** | 2026-02-24_22-57-58 |
| 0.05    | 0.9851 | 19         | 2026-02-23_02-04-16 |
| 0.1     | 0.9667 | 149        | 2026-02-22_14-36-51 |

### Key Findings
1. Separation loss is **critical** — removing it drops AUROC by 18.4%
2. **Optimal λ = 0.02** (AUROC = 0.9911) — **new all-time best** (beats seed runs of 0.9887)
3. Robust range: λ ∈ [0.01, 0.05], all give AUROC ≥ 0.9851
4. Too-large λ (0.1) causes slight degradation and much slower convergence (epoch 149 vs 19)
5. λ=0.01 dip to 0.9786 in early eval (epoch 9) recovered to 0.9869 — model needed more epochs

---

## Experiment 5 — λ=0.02 Multi-seed (in progress)

**Goal:** Confirm the new peak AUROC (0.9911 at λ=0.02, seed=42) generalises across seeds.

**Script:** `scripts/run_seeds_lambda02.sh`
**Config:** Same as Experiment 4 (sep_0.02) but with seeds 123 and 456.

| Seed | AUROC | Best Epoch | Status | Date |
|------|-------|------------|--------|----- |
| 42   | 0.9911 | 29        | ✅ Done | Feb 25 |
| 123  | 0.9840 | 29        | ✅ Done | Feb 26 |
| 456  | —      | —         | ⏳ Not started | — |

> When seed 456 completes, the 3-seed mean for λ=0.02 can be computed and compared
> to λ=0.01's mean of 0.9882 ± 0.0006.

---

## Experiment 6 — Figure Generation

**Script:** `scripts/generate_missing_figures.py`, `scripts/generate_all_figures_tables.py`
**Date:** 2026-02-26

New figures generated:
- `training_curves.png` — AUROC + FPR95 vs epoch for 3 seeds (λ=0.01)
- `roc_curves_cifar10.png` — ROC curves across 5 OOD datasets (seed 42)
- `sep_loss_dual.png` — Ablation curve + convergence speed bar chart
- `three_seed_auroc.png` — 3-seed bar chart showing 0.9882 ± 0.0006
- `scoring_methods_full.png` — 3-panel scoring method comparison

---

## Known Issues / Status

| Item | Status |
|------|--------|
| External OOD results | ✅ Done (seeds 42 & 123); ⚠️ Seed 456 needs re-verification |
| main_results_table.tex | ✅ Regenerated (Feb 26) |
| separation_loss_table.tex | ✅ Updated with all 6 λ values |
| λ=0.02 multi-seed | ✅ Seed 123 done (0.9840); ⏳ Seed 456 not started |
| AUPR/FPR95 for sep ablation | ⚠️ Only AUROC in checkpoints; full metrics need re-eval |
| Seed 456 external OOD | ⚠️ Threshold values look wrong; re-run needed |

---

## GPU & Server Log

See [gpu_log.md](gpu_log.md) for detailed GPU allocation history.
