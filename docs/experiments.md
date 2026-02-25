# Experiment Log

Full chronological log of all experiments run for this thesis.

---

## Experiment 1 — Main Training (3 Seeds)

**Goal:** Establish baseline performance of the Binary CDM with separation loss.

**Config:**
```
model:      ConditionalUNet (68.8M params)
dataset:    CIFAR-10 binary (50k train, 10k val)
batch_size: 64
lr:         1e-4 (cosine schedule)
epochs:     200
precision:  16-mixed
sep_weight: 0.01
num_trials: 15
scoring:    difference
timestep:   mid_focus
seed:       42 / 123 / 456
```

**Results:**
- Seed 42:  ~0.9780 AUROC
- Seed 123: ~0.9787 AUROC
- Seed 456: 0.9887 AUROC (best checkpoint: epoch ~150+)
- **Mean: 0.9818 ± 0.0049**

**Hardware:**
- Seed 42:  student10, GPU 2 (Tesla P40, 24GB)
- Seed 123: student10, GPU 2 (Tesla P40, 24GB)
- Seed 456: student10, GPU 1 (V100, 16GB) — resumed after OOM on P40

**Issues encountered:**
- Seed 456 initially crashed OOM on P40 (24GB) due to concurrent processes
- Moved to V100 (16GB), resumed from checkpoint — completed successfully
- Multiple zombie processes from previous runs caused GPU contention — killed manually

---

## Experiment 2 — External OOD Evaluation

**Goal:** Evaluate trained checkpoints (seeds 42, 123, 456) on external OOD datasets.

**Script:** `scripts/evaluate_external_ood.py`
**Config:** num_trials=50, batch_size=64, scoring=difference, timestep=mid_focus
**Datasets:** SVHN, CIFAR-100, and others
**Hardware:** student10, GPU 1 (V100, 16GB) — relaunched after initial crash on GPU 2

**Results:** Saved to `results/external_ood_results.json`

---

## Experiment 3 — Ablation Studies (K, Timestep, Scoring Method)

**Goal:** Study sensitivity to hyperparameters using best seed42 checkpoint.

**Script:** `scripts/run_ablations.py`

### 3a. K Ablation (num_trials)
Swept K = {1, 5, 10, 15, 25, 50}
Results: `results/k_ablation_results.json`

### 3b. Timestep Strategy
Compared: uniform, mid_focus, early, late
Results: `results/scoring_method_results.json`

### 3c. Scoring Method
Compared: difference vs ratio
Results: `results/scoring_method_results.json`

---

## Experiment 4 — Separation Loss Ablation

**Goal:** Quantify the effect of the separation loss coefficient λ.

**Script:** `scripts/train.py` (repeated 6 times with different weights)
**Config:** Identical to Experiment 1 except separation_loss_weight varies.

### Timeline

| Date | Event |
|------|-------|
| Feb 17 | First attempts at sep_0.0, 0.001, 0.05, 0.1 (slow, competing processes) |
| Feb 19–20 | Multiple restarts due to GPU contention on student10 |
| Feb 20 | sep_0.0 completed full training (200 epochs), AUROC=0.8025 |
| Feb 21 22:03 | sep_0.001 started automatically (student10 sequential for-loop) |
| Feb 22 | sep_0.05 started on student06 GPU0, sep_0.1 on student06 GPU1 |
| Feb 22 | DDP (2-GPU) attempted for sep_0.1 — crashed with NCCL timeout (30min sanity check) |
| Feb 22 | Fixed: `num_sanity_val_steps=0` added to Trainer; reverted to single-GPU |
| Feb 23 | sep_0.05 DONE (AUROC=0.9851), sep_0.1 DONE (AUROC=0.9667) |
| Feb 23 21:41 | sep_0.01 completed (AUROC=0.9869, new peak!) |
| Feb 24 | sep_0.02 completed (AUROC=0.9786) |
| Feb 25 | All 6 weights complete — final figure generated |

### Final Results

| λ       | AUROC  | Notes         |
|---------|--------|---------------|
| 0.0     | 0.8025 | No sep loss   |
| 0.001   | 0.9732 |               |
| **0.01**| **0.9869** | **Peak** |
| 0.02    | 0.9786 |               |
| 0.05    | 0.9851 |               |
| 0.1     | 0.9667 |               |

### Key Findings
1. Separation loss is **critical** — removing it drops AUROC by 18.4%
2. Optimal λ ≈ 0.01, but the range 0.01–0.05 is robust
3. Too-large λ (0.1) causes slight degradation
4. The non-monotonicity at 0.02 suggests single-seed variance; multi-seed would smooth this

---

## GPU & Server Log

See [gpu_log.md](gpu_log.md) for detailed GPU allocation history.
