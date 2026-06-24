# Instant-NGP Accelerated NeRF for 3D Animation Teaching Scene Reconstruction

## Overview
This repository contains the official implementation of the paper:
"Reconstruction of 3D Animated Teaching Scenes and Optimization of Virtual Reality Interactive Teaching Model Based on Instant-NGP Acceleration of NeRF"

## Key Features
- Multi-resolution hash encoding for fast NeRF training
- Distillation from implicit radiance fields to explicit 3D Gaussian primitives
- Real-time VR interaction with view-frustum culling and dynamic attribute updates
- Multi-view geometric consistency and perceptual detail enhancement

## Requirements
- Python 3.10+
- PyTorch 2.0+
- CUDA 12.1
- NVIDIA GPU with 24GB+ VRAM (RTX 4090 recommended)
- OpenXR runtime (for VR deployment)

## Quick Start
```bash
# Install dependencies
conda env create -f environment.yaml
conda activate nerf_teaching

# Download datasets
bash scripts/download_datasets.sh

# Train NeRF with Instant-NGP
python src/train_nerf.py --config config/train_nerf.yaml

# Distill to Gaussian primitives
python src/distill_gaussian.py --config config/distill_gaussian.yaml

# Run VR interaction
python src/vr_interact.py --config config/vr_interact.yaml

# Evaluate
python src/run_evaluation.py --checkpoint experiments/checkpoints/full_model.pth

