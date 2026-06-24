#!/bin/bash
# Dataset download script

set -e

DATA_DIR="./data"
mkdir -p ${DATA_DIR}

echo "Downloading NeRF-4Scenes dataset..."
# Replace with actual download URL
# wget -P ${DATA_DIR} https://example.com/nerf_4scenes.zip
# unzip ${DATA_DIR}/nerf_4scenes.zip -d ${DATA_DIR}

echo "Downloading Mip-NeRF360 dataset..."
# wget -P ${DATA_DIR} https://example.com/mipnerf360.zip
# unzip ${DATA_DIR}/mipnerf360.zip -d ${DATA_DIR}

echo "Downloading DONeRF Classroom dataset..."
# wget -P ${DATA_DIR} https://example.com/donerf_classroom.zip
# unzip ${DATA_DIR}/donerf_classroom.zip -d ${DATA_DIR}

echo "Dataset download complete."