#!/bin/bash

# Check if the rosbag file path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/nas/rosbag.bag"
    exit 1
fi

# Get the full path of the rosbag and extract the base name without extension
ROSBAG_NAS_PATH=$1
ROSBAG_NAME=$(basename "$ROSBAG_NAS_PATH" .bag)
ROSBAG_DETINATION_PATH="/home/model01/perception/pointpillar/jac_data"

ROSBAG_RESULTS_PATH="tools/projects/jac/$ROSBAG_NAME"
EXTRACTED_TOPICS_PATH="${ROSBAG_RESULTS_PATH}/extracted_topics"
STITCHED_PATH="${ROSBAG_RESULTS_PATH}/stitched"
STITCHED_PATH_BIN="${ROSBAG_RESULTS_PATH}/stitched_bin"

EXPORT_DIR_ONNX_V5="${ROSBAG_RESULTS_PATH}/onnx_results_v5"
EXPORT_DIR_PYTORCH_V5="${ROSBAG_RESULTS_PATH}/pytorch_results_v5"
EXPORT_DIR_ONNX_V6="${ROSBAG_RESULTS_PATH}/onnx_results_v6"
EXPORT_DIR_PYTORCH_V6="${ROSBAG_RESULTS_PATH}/pytorch_results_v6"
VIDEOS_DIR="${ROSBAG_RESULTS_PATH}/videos"

CONDA_ENV_NAME="shahryarenv"

echo "Step 1: Creating necessary directories..."
mkdir -p "$ROSBAG_RESULTS_PATH" && echo "Created $ROSBAG_RESULTS_PATH."
mkdir -p "$STITCHED_PATH" && echo "Created $STITCHED_PATH."
mkdir -p "$STITCHED_PATH_BIN" && echo "Created $STITCHED_PATH_BIN."
mkdir -p "$EXPORT_DIR_ONNX_V5" && echo "Created $EXPORT_DIR_ONNX_V5."
mkdir -p "$EXPORT_DIR_PYTORCH_V5" && echo "Created $EXPORT_DIR_PYTORCH_V5."
mkdir -p "$EXPORT_DIR_ONNX_V6" && echo "Created $EXPORT_DIR_ONNX_V6."
mkdir -p "$EXPORT_DIR_PYTORCH_V6" && echo "Created $EXPORT_DIR_PYTORCH_V6."
mkdir -p "$VIDEOS_DIR" && echo "Created $VIDEOS_DIR."


echo "Step 2: Copying the rosbag to the destination directory..."
# Copy the rosbag to the destination directory
cp "$ROSBAG_NAS_PATH" "$ROSBAG_DETINATION_PATH"
if [ $? -eq 0 ]; then
    echo "Rosbag copied successfully to $ROSBAG_DETINATION_PATH."
else
    echo "Failed to copy rosbag."
    exit 1
fi


echo "Step 3: Activating the conda environment..."
# Activate the conda environment and run rosbag parsing script
source ~/miniconda3/etc/profile.d/conda.sh  # Ensure conda is sourced correctly
conda activate "$CONDA_ENV_NAME"
if [ $? -eq 0 ]; then
    echo "Conda environment '$CONDA_ENV_NAME' activated."
else
    echo "Failed to activate conda environment '$CONDA_ENV_NAME'."
    exit 1
fi

chmod +x tools/projects/jac/run_preprocess.sh

echo "Step 4: Running preprocessing rosbag script..."
tools/projects/jac/run_preprocess.sh "$ROSBAG_DETINATION_PATH/$ROSBAG_NAME.bag" "$EXTRACTED_TOPICS_PATH" "$STITCHED_PATH"

if [ $? -eq 0 ]; then
    echo "run_preprocess.sh completed successfully."
else
    echo "Failed to run v5 run_models.sh."
    exit 1
fi

# Move all .bin files from STITCHED_PATH to STITCHED_PATH_BIN
mv "$STITCHED_PATH"/*.bin "$STITCHED_PATH_BIN"

echo "Step 5: Running v5 run_models.sh with stitched path..."
tools/projects/jac/run_models.sh "$STITCHED_PATH_BIN" "$EXPORT_DIR_ONNX_V5" "$EXPORT_DIR_PYTORCH_V5" v5
if [ $? -eq 0 ]; then
    echo "v5 run_models.sh completed successfully."
else
    echo "Failed to run v5 run_models.sh."
    exit 1
fi


echo "Step 6: Running v6 run_models.sh with stitched path..."
tools/projects/jac/run_models.sh "$STITCHED_PATH_BIN" "$EXPORT_DIR_ONNX_V6" "$EXPORT_DIR_PYTORCH_V6" v6
if [ $? -eq 0 ]; then
    echo "v6 run_models.sh completed successfully."
else
    echo "Failed to run v6 run_models.sh."
    exit 1
fi


echo "Step 7: Running video creation script..."

# Run the first command and check if it succeeded
if tools/projects/jac/run_create_video.sh "$EXPORT_DIR_ONNX_V5" "$VIDEOS_DIR/onnx_results_v5.mp4"; then
    echo "Video creation succeeded for ONNX V5."
else
    echo "Video creation failed for ONNX V5."
fi

# Run the second command and check if it succeeded
if tools/projects/jac/run_create_video.sh "$EXPORT_DIR_ONNX_V6" "$VIDEOS_DIR/onnx_results_v6.mp4"; then
    echo "Video creation succeeded for ONNX V6."
else
    echo "Video creation failed for ONNX V6."
fi

# Run the third command and check if it succeeded
if tools/projects/jac/run_create_video.sh "$EXPORT_DIR_PYTORCH_V5/vis_lidar" "$VIDEOS_DIR/pytorch_results_v5.mp4"; then
    echo "Video creation succeeded for PyTorch V5."
else
    echo "Video creation failed for PyTorch V5."
fi

# Run the fourth command and check if it succeeded
if tools/projects/jac/run_create_video.sh "$EXPORT_DIR_PYTORCH_V6/vis_lidar" "$VIDEOS_DIR/pytorch_results_v6.mp4"; then
    echo "Video creation succeeded for PyTorch V6."
else
    echo "Video creation failed for PyTorch V6."
fi

if [ $? -eq 0 ]; then
    echo "Video creation script completed successfully."
else
    echo "Failed to run video creation script."
    exit 1
fi


echo "Step 8: Transfering files to NAS script..."
tools/projects/jac/transfer_data_to_nas.sh "$ROSBAG_NAME" "$ROSBAG_RESULTS_PATH"
if [ $? -eq 0 ]; then
    echo "Transfering files to NAS script completed successfully."
else
    echo "Failed to run Transfering files to NAS script."
    exit 1
fi

