#!/bin/bash
# Full pipeline execution script

set -e

echo "=========================================="
echo "NeRF Teaching Scene Reconstruction Pipeline"
echo "=========================================="

# Setup
export CUDA_VISIBLE_DEVICES=0

# Configuration
CONFIG_DIR="config"
DATA_DIR="data"
CKPT_DIR="experiments/checkpoints"
LOG_DIR="experiments/logs"
RESULT_DIR="experiments/results"

# Create directories
mkdir -p ${CKPT_DIR} ${LOG_DIR} ${RESULT_DIR}

# Step 1: Train NeRF with Instant-NGP
echo ""
echo "[Step 1/4] Training NeRF with multi-resolution hash encoding..."
python src/train_nerf.py --config ${CONFIG_DIR}/train_nerf.yaml

# Step 2: Distill Gaussian primitives
echo ""
echo "[Step 2/4] Distilling Gaussian primitives from NeRF..."
python src/distill_gaussian.py --config ${CONFIG_DIR}/distill_gaussian.yaml

# Step 3: Run VR interaction
echo ""
echo "[Step 3/4] Running VR interaction..."
python src/vr_interact.py --config ${CONFIG_DIR}/vr_interact.yaml

# Step 4: Evaluate
echo ""
echo "[Step 4/4] Evaluating results..."
python src/run_evaluation.py --config ${CONFIG_DIR}/default.yaml

echo ""
echo "=========================================="
echo "Pipeline completed!"
echo "Results saved to: ${RESULT_DIR}"
echo "=========================================="