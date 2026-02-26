#!/bin/bash
# ============================================================================
# Run 2 remaining seeds (123, 456) with λ=0.02 to complete the 3-seed study.
# Seed 42 already done: AUROC=0.9911
#
# Usage:
#   CUDA_VISIBLE_DEVICES=3 nohup bash scripts/run_seeds_lambda02.sh > results/seeds_lambda02.log 2>&1 &
#
# Expected time: ~6-8 hours per seed ≈ 12-16 hours total
# GPU memory: ~6-8 GB per run
# ============================================================================

set -e

CONDA_ENV="/system/apps/studentenv/mohammed/sdm"
PROJECT_DIR="/system/user/studentwork/mohammed/2025/diffusion_classifier_ood"
cd "$PROJECT_DIR"

echo "=============================================="
echo "λ=0.02 — 3-Seed Completion (seeds 123 + 456)"
echo "Started: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

for SEED in 123 456; do
    echo ""
    echo "=============================================="
    echo "[$(date)] Starting seed $SEED (λ=0.02)"
    echo "=============================================="
    
    conda run --no-capture-output -p "$CONDA_ENV" python scripts/train.py \
        --seed $SEED \
        --separation_loss_weight 0.02 \
        --experiment_tag "sep_0.02_seed${SEED}" \
        --output_dir "results/sep_loss_ablation/sep_0.02_seed${SEED}" \
        --batch_size 64 \
        --learning_rate 1e-4 \
        --max_epochs 200 \
        --accumulate_grad_batches 2 \
        --num_trials 15 \
        --eval_interval 10 \
        --scoring_method difference \
        --timestep_mode mid_focus \
        --num_workers 4 \
        --wandb_mode offline
    
    echo "[$(date)] Seed $SEED DONE"
    echo ""
done

echo "=============================================="
echo "All seeds complete: $(date)"
echo "=============================================="
echo ""
echo "Next step: Update the 3-seed bar chart:"
echo "  conda run -p $CONDA_ENV python scripts/generate_all_figures_tables.py"
