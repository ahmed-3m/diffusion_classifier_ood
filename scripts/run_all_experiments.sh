#!/bin/bash
# ==============================================================================
# MASTER EXPERIMENT RUNNER - Binary CDM for OOD Detection (Optimized Parallel)
# ==============================================================================
# This script runs all 6 experiment sets for the thesis.
# Uses GPU 2 and GPU 0 in parallel to speed up execution.
#
# GPU 2: Seed 42, Seed 123, Sep Loss (0.0, 0.001)
# GPU 0: Seed 456, Sep Loss (0.05, 0.1)
#
# Usage: bash scripts/run_all_experiments.sh
# ==============================================================================

set -e

# Configuration
CONDA_ENV="/system/apps/studentenv/mohammed/sdm/"
PROJECT_DIR="/system/user/studentwork/mohammed/2025/diffusion_classifier_ood"
RESULTS_DIR="${PROJECT_DIR}/results"
WANDB_MODE="online"

# Create directory structure
mkdir -p "${RESULTS_DIR}"/{seed42,seed123,seed456,raw_scores,figures,latex_tables}
mkdir -p "${RESULTS_DIR}/sep_loss_ablation"

# Initialize conda
# Using absolute path to ensure availability in non-interactive shells
source /system/user/mohammed/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

echo "============================================================"
echo " DIFFUSION CLASSIFIER OOD - PARALLEL EXPERIMENT RUNNER"
echo " Project: ${PROJECT_DIR}"
echo " GPU Strategy: GPU 2 (Seed 42, 123) + GPU 0 (Seed 456)"
echo " Results: ${RESULTS_DIR}"
echo "============================================================"

cd "${PROJECT_DIR}"

# ==============================================================================
# EXPERIMENT SET 1: Train Main Model (Parallel Execution)
# ==============================================================================
echo ""
echo "[EXP 1] Launching parallel training..."

# --- GPU 2 JOB (Seeds 42, 123) ---
(
    set -e
    echo "[GPU 2] Starting Seed 42..."
    CUDA_VISIBLE_DEVICES=2 python scripts/train.py \
        --seed 42 \
        --batch_size 64 \
        --learning_rate 1e-4 \
        --max_epochs 200 \
        --num_trials 15 \
        --separation_loss_weight 0.01 \
        --scoring_method difference \
        --timestep_mode mid_focus \
        --eval_interval 5 \
        --experiment_tag "seed42" \
        --output_dir "${RESULTS_DIR}/seed42" \
        --wandb_mode "${WANDB_MODE}" \
        2>&1 | tee "${RESULTS_DIR}/seed42/training.log"
    
    echo "[GPU 2] Seed 42 Done. Starting Seed 123..."
    CUDA_VISIBLE_DEVICES=2 python scripts/train.py \
        --seed 123 \
        --batch_size 64 \
        --learning_rate 1e-4 \
        --max_epochs 200 \
        --num_trials 15 \
        --separation_loss_weight 0.01 \
        --scoring_method difference \
        --timestep_mode mid_focus \
        --eval_interval 5 \
        --experiment_tag "seed123" \
        --output_dir "${RESULTS_DIR}/seed123" \
        --wandb_mode "${WANDB_MODE}" \
        2>&1 | tee "${RESULTS_DIR}/seed123/training.log"
        
    echo "[GPU 2] Seed 123 Done."
) &
PID_GPU2=$!

# --- GPU 0 JOB (Seed 456) ---
(
    set -e
    echo "[GPU 0] Starting Seed 456..."
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
        --seed 456 \
        --batch_size 64 \
        --learning_rate 1e-4 \
        --max_epochs 200 \
        --num_trials 15 \
        --separation_loss_weight 0.01 \
        --scoring_method difference \
        --timestep_mode mid_focus \
        --eval_interval 5 \
        --experiment_tag "seed456" \
        --output_dir "${RESULTS_DIR}/seed456" \
        --wandb_mode "${WANDB_MODE}" \
        2>&1 | tee "${RESULTS_DIR}/seed456/training.log"
        
    echo "[GPU 0] Seed 456 Done."
) &
PID_GPU0=$!

# Wait for both chains to finish
echo "Waiting for training jobs to complete..."
wait $PID_GPU2
wait $PID_GPU0
echo "[EXP 1] COMPLETE - All seeds trained."


# ==============================================================================
# EXPERIMENT SET 2: Evaluate on External OOD Datasets
# ==============================================================================
echo ""
echo "[EXP 2] Evaluating on External OOD Datasets..."
# Using GPU 2 for evaluation since training is done
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_external_ood.py \
    --results_dir "${RESULTS_DIR}" \
    --data_dir ./data \
    --num_trials 50 \
    2>&1 | tee "${RESULTS_DIR}/external_ood_eval.log"

echo "[EXP 2] COMPLETE"


# ==============================================================================
# EXPERIMENT SET 6: Separation Loss Ablation (Parallel Execution)
# ==============================================================================
echo ""
echo "[EXP 6] Running Separation Loss Ablation (Parallel)..."

# --- GPU 2 JOB (Weights 0.0, 0.001) ---
(
    set -e
    for WEIGHT in 0.0 0.001; do
        echo "[GPU 2] Training with sep_loss=${WEIGHT}..."
        CUDA_VISIBLE_DEVICES=2 python scripts/train.py \
            --seed 42 \
            --batch_size 64 \
            --learning_rate 1e-4 \
            --max_epochs 200 \
            --num_trials 15 \
            --separation_loss_weight ${WEIGHT} \
            --scoring_method difference \
            --timestep_mode mid_focus \
            --eval_interval 10 \
            --experiment_tag "sep_${WEIGHT}" \
            --output_dir "${RESULTS_DIR}/sep_loss_ablation" \
            --wandb_mode "${WANDB_MODE}" \
            2>&1 | tee "${RESULTS_DIR}/sep_loss_ablation/sep_${WEIGHT}.log"
    done
    echo "[GPU 2] Ablation tasks done."
) &
PID_ABLATION_G2=$!

# --- GPU 0 JOB (Weights 0.05, 0.1) ---
# Note: 0.01 is skipped as it corresponds to main Seed 42 run (default)
(
    set -e
    for WEIGHT in 0.05 0.1; do
        echo "[GPU 0] Training with sep_loss=${WEIGHT}..."
        CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
            --seed 42 \
            --batch_size 64 \
            --learning_rate 1e-4 \
            --max_epochs 200 \
            --num_trials 15 \
            --separation_loss_weight ${WEIGHT} \
            --scoring_method difference \
            --timestep_mode mid_focus \
            --eval_interval 10 \
            --experiment_tag "sep_${WEIGHT}" \
            --output_dir "${RESULTS_DIR}/sep_loss_ablation" \
            --wandb_mode "${WANDB_MODE}" \
            2>&1 | tee "${RESULTS_DIR}/sep_loss_ablation/sep_${WEIGHT}.log"
    done
    echo "[GPU 0] Ablation tasks done."
) &
PID_ABLATION_G0=$!

echo "Waiting for ablation jobs to complete..."
wait $PID_ABLATION_G2
wait $PID_ABLATION_G0
echo "[EXP 6] Training COMPLETE"


# ==============================================================================
# EXPERIMENT SET 6 Evaluation
# ==============================================================================
echo ""
echo "[EXP 6 Evaluation] Evaluating separation loss models..."

CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_separation_ablation.py \
    --results_dir "${RESULTS_DIR}" \
    --data_dir ./data \
    2>&1 | tee "${RESULTS_DIR}/sep_ablation_eval.log"

echo "[EXP 6 Evaluation] COMPLETE"


# ==============================================================================
# EXPERIMENT SETS 3-5: Ablation Studies (evaluation only, using best seed42 ckpt)
# ==============================================================================
echo ""
echo "[EXP 3-5] Running K, Timestep, Scoring Ablations..."

CUDA_VISIBLE_DEVICES=2 python scripts/run_ablations.py \
    --results_dir "${RESULTS_DIR}" \
    --data_dir ./data \
    2>&1 | tee "${RESULTS_DIR}/ablations.log"

echo "[EXP 3-5] COMPLETE"


# ==============================================================================
# FIGURE GENERATION
# ==============================================================================
echo ""
echo "[FIGURES] Generating Thesis Figures & Tables..."

python scripts/generate_all_figures.py \
    --results_dir "${RESULTS_DIR}" \
    2>&1 | tee "${RESULTS_DIR}/figures_generation.log"

echo "============================================================"
echo " ALL EXPERIMENTS COMPLETE!"
echo " Results available in: ${RESULTS_DIR}"
echo "============================================================"
