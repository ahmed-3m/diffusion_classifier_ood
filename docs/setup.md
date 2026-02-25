# Environment Setup & Reproducibility Guide

---

## Prerequisites

- Linux (tested on student10, student06, student07 university servers)
- NVIDIA GPU with CUDA 12.x
- Conda / Miniconda

## Conda Environment

```bash
conda activate /system/apps/studentenv/mohammed/sdm/
# or
source /system/user/mohammed/miniconda3/etc/profile.d/conda.sh
conda activate /system/apps/studentenv/mohammed/sdm/
```

## Project Directory

```
/system/user/studentwork/mohammed/2025/diffusion_classifier_ood/
```

---

## Reproducing All Experiments

### Step 1 — Main Training (3 Seeds)

```bash
cd /system/user/studentwork/mohammed/2025/diffusion_classifier_ood

# Seed 42
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --seed 42 --batch_size 64 --learning_rate 1e-4 \
    --max_epochs 200 --num_trials 15 \
    --separation_loss_weight 0.01 \
    --scoring_method difference --timestep_mode mid_focus \
    --eval_interval 10 --experiment_tag "seed42" \
    --output_dir results/seed42 --wandb_mode online

# Seed 123 (change --seed and --output_dir)
# Seed 456 (change --seed and --output_dir)
```

### Step 2 — External OOD Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_external_ood.py \
    --results_dir results/ --data_dir ./data \
    --num_trials 50 --batch_size 64
```

### Step 3 — Ablation Studies (K, Timestep, Scoring)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_ablations.py \
    --results_dir results/ --data_dir ./data --batch_size 64
```

### Step 4 — Separation Loss Ablation

```bash
for W in 0.0 0.001 0.01 0.02 0.05 0.1; do
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
        --seed 42 --batch_size 64 --learning_rate 1e-4 \
        --max_epochs 200 --num_workers 8 --num_trials 15 \
        --separation_loss_weight ${W} \
        --scoring_method difference --timestep_mode mid_focus \
        --eval_interval 10 --experiment_tag "sep_${W}" \
        --output_dir results/sep_loss_ablation \
        --wandb_mode online
done
```

### Step 5 — Generate All Figures & Tables

```bash
python scripts/generate_all_figures.py --results_dir results/

# Optional: regenerate separation loss ablation figure
MPLBACKEND=Agg python scripts/plot_sep_ablation.py
```

---

## Important Notes

### GPU Recommendations

| GPU | Architecture | FP16 Tensor Cores | Training Speed | Recommended |
|-----|-------------|-------------------|----------------|-------------|
| Tesla P40 (24GB) | Pascal | ❌ No | ~0.35 it/s | ❌ Avoid |
| GTX 1080 Ti (11GB) | Pascal | ❌ No | ~0.5 it/s | ❌ Avoid |
| Titan V (12GB) | Volta | ✅ Yes | ~3.0 it/s | ✅ Good |
| V100 (16/32GB) | Volta | ✅ Yes | ~3.0 it/s | ✅ Best |

> ⚠️ Always use `CUDA_VISIBLE_DEVICES` to pin to a specific GPU.
> The model uses `precision="16-mixed"` — requires Tensor Cores for speed benefit.

### Multi-GPU (DDP) Notes

- **Safe pairs**: GPU0+GPU1 (PHB-connected), GPU2+GPU3 (PHB-connected) on dual-CPU servers
- **Avoid**: Cross-NUMA pairs (SYS topology) — NCCL timeout risk
- **Critical fix**: Set `num_sanity_val_steps=0` in Trainer — OOD sanity check takes 30+ min
  and causes NCCL timeout in DDP mode
- **Recommendation**: For this model, single-GPU per experiment is most reliable

### Resume Training

```bash
python scripts/train.py \
    --checkpoint_path results/TIMESTAMP_tag/last.ckpt \
    ... (same args as original run)
```

### WandB

```bash
export WANDB_MODE=online   # log to wandb.ai
export WANDB_MODE=offline  # save locally, sync later
export WANDB_MODE=disabled # no logging
```

---

## Key Files

```
scripts/
├── train.py                      # Main training script
├── evaluate.py                   # Single-model evaluation
├── evaluate_external_ood.py      # Multi-seed external OOD eval
├── evaluate_separation_ablation.py # Ablation evaluation
├── run_ablations.py              # K / timestep / scoring ablations
├── generate_all_figures.py       # All thesis figures
├── plot_sep_ablation.py          # Separation loss ablation curve
├── run_all_experiments.sh        # Master orchestration script
└── run_sep_parallel.sh           # Parallel sep-loss training

src/
├── model.py           # ConditionalUNet architecture
├── lightning_module.py # PyTorch Lightning wrapper
├── data.py            # CIFAR-10 binary DataModule
├── scoring.py         # Diffusion classifier score
├── metrics.py         # AUROC, FPR, AUPR etc.
└── utils.py           # Callbacks, logging, checkpointing

configs/
└── default.py         # Default hyperparameter dataclasses
```
