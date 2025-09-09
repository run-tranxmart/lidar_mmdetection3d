#!/bin/bash
# Check if the required arguments are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
    echo "Usage: $0 /path/to/bin_stitched_data /path/to/export_dir_onnx /path/to/export_dir_pytorch version"
    exit 1
fi

# Get the paths and version from the arguments
BIN_STITCHED_PATH=$1
EXPORT_DIR_ONNX=$2
EXPORT_DIR_PYTORCH=$3
VERSION=$4

# Print out the arguments for confirmation (optional)
echo "BIN_STITCHED_PATH: $BIN_STITCHED_PATH"
echo "EXPORT_DIR_ONNX: $EXPORT_DIR_ONNX"
echo "EXPORT_DIR_PYTORCH: $EXPORT_DIR_PYTORCH"
echo "VERSION: $VERSION"


# Set paths based on the version provided
if [ "$VERSION" == "v5" ]; then
    ONNX_PATH_1="/home/model01/perception/pointpillar/onnx/v2_5_new/pfns.onnx"
    ONNX_PATH_2="/home/model01/perception/pointpillar/onnx/v2_5_new/detector.onnx"
    CHECKPOINT_PATH="/home/model01/perception/pointpillar/checkpoints/v2_5_epoch_22.pth"
    CONFIG_PATH="configs/users/mahdi/configs/batch6_pointpillars_v2_5_light_4class_onnx.py"
elif [ "$VERSION" == "v6" ]; then
    ONNX_PATH_1="/home/model01/perception/pointpillar/onnx/v2_6_epoch40/pfns.onnx"
    ONNX_PATH_2="/home/model01/perception/pointpillar/onnx/v2_6_epoch40/detector.onnx"
    CHECKPOINT_PATH="/home/model01/perception/pointpillar/checkpoints/v2_6_epoch40.pth"
    CONFIG_PATH="configs/users/mahdi/configs/batch6_narrow_dataset_pointpillars_v2_5_light_3dim.py"
else
    echo "Invalid version specified. Use 'v5' or 'v6'."
    exit 1
fi


# Activate the conda environment
echo "Activating the conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sheheryar
if [ $? -eq 0 ]; then
    echo "Conda environment 'sheheryar' activated."
else
    echo "Failed to activate conda environment."
    exit 1
fi

# Run the ONNX demo script
echo "Running ONNX demo script for $VERSION with stitched binary path: $BIN_STITCHED_PATH"
python tools/onnx_utils/demo.py \
    --cloud "$BIN_STITCHED_PATH" \
    --onnx1 "$ONNX_PATH_1" \
    --onnx2 "$ONNX_PATH_2" \
    --export_dir "$EXPORT_DIR_ONNX" \
    --config "$CONFIG_PATH" \
    --visual

if [ $? -eq 0 ]; then
    echo "$VERSION ONNX processing completed successfully."
else
    echo "Failed to run $VERSION ONNX script."
    exit 1
fi

# Run the PyTorch demo script
echo "Running PyTorch demo script for $VERSION with stitched binary path: $BIN_STITCHED_PATH"
python demo/pcd_demo.py \
    "$BIN_STITCHED_PATH" \
    "$CONFIG_PATH" \
    "$CHECKPOINT_PATH" \
    --out-dir "$EXPORT_DIR_PYTORCH" \
    --show

if [ $? -eq 0 ]; then
    echo "$VERSION PyTorch processing completed successfully."
else
    echo "Failed to run $VERSION PyTorch script."
    exit 1
fi
