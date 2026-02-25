#!/bin/bash
# ============================================================
# Run sep_0.05 (GPUs 0,1) and sep_0.1 (GPUs 2,3) in parallel
# Usage: bash scripts/run_sep_parallel.sh
# ============================================================

set -e

PROJECT_DIR="/system/user/studentwork/mohammed/2025/diffusion_classifier_ood"
RESULTS_DIR="${PROJECT_DIR}/results/sep_loss_ablation"
CONDA_SH="/system/user/mohammed/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="/system/apps/studentenv/mohammed/sdm/"

cd "$PROJECT_DIR"
mkdir -p "$RESULTS_DIR"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

COMMON_ARGS="--seed 42 --batch_size 64 --learning_rate 1e-4 --max_epochs 200
    --num_trials 15 --scoring_method difference --timestep_mode mid_focus
    --eval_interval 10 --output_dir ${RESULTS_DIR} --wandb_mode online"

echo "============================================================"
echo "  Starting sep_0.05 on GPUs 0,1"
echo "  Starting sep_0.1  on GPUs 2,3"
echo "============================================================"

# --- Job 1: sep_0.05 on GPUs 0,1 ---
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py \
    $COMMON_ARGS \
    --separation_loss_weight 0.05 \
    --experiment_tag "sep_0.05" \
    > "${RESULTS_DIR}/sep_0.05.log" 2>&1 &
PID1=$!
echo "[sep_0.05] Started with PID=$PID1 on GPUs 0,1"

# Small delay so GPU memory init doesn't collide
sleep 5

# --- Job 2: sep_0.1 on GPUs 2,3 ---
CUDA_VISIBLE_DEVICES=2,3 python scripts/train.py \
    $COMMON_ARGS \
    --separation_loss_weight 0.1 \
    --experiment_tag "sep_0.1" \
    > "${RESULTS_DIR}/sep_0.1.log" 2>&1 &
PID2=$!
echo "[sep_0.1]  Started with PID=$PID2 on GPUs 2,3"

echo ""
echo "Monitor progress:"
echo "  tail -f ${RESULTS_DIR}/sep_0.05.log"
echo "  tail -f ${RESULTS_DIR}/sep_0.1.log"
echo ""
echo "Waiting for both jobs to finish..."

# Wait for both and capture exit codes
wait $PID1
STATUS1=$?
echo "[sep_0.05] Finished with exit code $STATUS1"

wait $PID2
STATUS2=$?
echo "[sep_0.1]  Finished with exit code $STATUS2"

# --- Run evaluation + figures once both are done ---
if [ $STATUS1 -eq 0 ] && [ $STATUS2 -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "  Both trainings complete! Running evaluation..."
    echo "============================================================"
    python scripts/evaluate_separation_ablation.py \
        --results_dir results/ \
        --data_dir ./data

    echo "Generating figures..."
    python scripts/generate_all_figures.py --results_dir results/
    echo "ALL DONE — figures saved to results/figures/"
else
    echo "One or both jobs failed (codes: $STATUS1, $STATUS2). Check logs."
    exit 1
fi
