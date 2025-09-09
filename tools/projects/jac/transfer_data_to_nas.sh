#!/bin/bash

# Get the variables from the passed arguments
ROSBAG_NAME=$1
ROSBAG_RESULTS_PATH=$2

# Define the directories for v5, v6, and stitched
stitched_dir="${ROSBAG_RESULTS_PATH}/stitched"
stitched_bin_dir="${ROSBAG_RESULTS_PATH}/stitched_bin"
onnx_results_v5_dir="$ROSBAG_RESULTS_PATH/onnx_results_v5"
onnx_results_v6_dir="$ROSBAG_RESULTS_PATH/onnx_results_v6"
pytorch_results_v5_dir="$ROSBAG_RESULTS_PATH/pytorch_results_v5/vis_lidar"
pytorch_results_v6_dir="$ROSBAG_RESULTS_PATH/pytorch_results_v6/vis_lidar"

# Define the video files
onnx_results_v5_video="$ROSBAG_RESULTS_PATH/videos/onnx_results_v5.mp4"
onnx_results_v6_video="$ROSBAG_RESULTS_PATH/videos/onnx_results_v6.mp4"
pytorch_results_v5_video="$ROSBAG_RESULTS_PATH/videos/pytorch_results_v5.mp4"
pytorch_results_v6_video="$ROSBAG_RESULTS_PATH/videos/pytorch_results_v6.mp4"

# Function to check directory existence
check_directory_existence() {
  if [ ! -d "$1" ]; then
    echo "Error: $1 does not exist."
    exit 1
  else
    echo "$1 exists."
  fi
}

# Check existence of directories
check_directory_existence "$onnx_results_v5_dir"
check_directory_existence "$onnx_results_v6_dir"
check_directory_existence "$pytorch_results_v5_dir"
check_directory_existence "$pytorch_results_v6_dir"
check_directory_existence "$stitched_dir"
check_directory_existence "$stitched_bin_dir"

# Define the base paths and include ROSBAG_NAME in the subdirectories
nas_pytorch_results="/mnt/CAEA_NAS/Perception_Model/Pointpillars/JAC_Project/pytorch_results/$ROSBAG_NAME"
nas_onnx_results="/mnt/CAEA_NAS/Perception_Model/Pointpillars/JAC_Project/onnx_results/$ROSBAG_NAME"
nas_cloud_results="/mnt/CAEA_NAS/Perception_Model/Pointpillars/JAC_Project/cloud_pcd/$ROSBAG_NAME"

# Create directories on NAS
sudo mkdir -p "$nas_pytorch_results"
sudo mkdir -p "$nas_onnx_results"
sudo mkdir -p "$nas_cloud_results"
echo "Directories created for ROSBAG_NAME: $ROSBAG_NAME"

# Create zip files for STITCHED_PATH and STITCHED_PATH_BIN
zip -j "$stitched_dir.zip" $(find "$stitched_dir" -type f) && echo "Zipped stitched directory without nesting."
zip -j "$stitched_bin_dir.zip" $(find "$stitched_bin_dir" -type f) && echo "Zipped stitched_bin directory without nesting."

# Copy the zip files to NAS cloud directory
sudo cp "$stitched_dir.zip" "$nas_cloud_results" && echo "Copied stitched.zip to $nas_cloud_results"
sudo cp "$stitched_bin_dir.zip" "$nas_cloud_results" && echo "Copied stitched_bin.zip to $nas_cloud_results"

# Creating zip files for PNG files
echo "Creating zip files for PNG files..."

# ONNX V5
zip -j "${onnx_results_v5_dir}/onnx_v5_pngs.zip" $(find "$onnx_results_v5_dir" -type f -name "*.png") && echo "Zipped ONNX V5 PNGs without nesting."
sudo cp "${onnx_results_v5_dir}/onnx_v5_pngs.zip" "$nas_onnx_results" && echo "Copied ONNX V5 zip to $nas_onnx_results."

# ONNX V6
zip -j "${onnx_results_v6_dir}/onnx_v6_pngs.zip" $(find "$onnx_results_v6_dir" -type f -name "*.png") && echo "Zipped ONNX V6 PNGs without nesting."
sudo cp "${onnx_results_v6_dir}/onnx_v6_pngs.zip" "$nas_onnx_results" && echo "Copied ONNX V6 zip to $nas_onnx_results."

# PyTorch V5
zip -j "${pytorch_results_v5_dir}/pytorch_v5_pngs.zip" $(find "$pytorch_results_v5_dir" -type f -name "*.png") && echo "Zipped PyTorch V5 PNGs without nesting."
sudo cp "${pytorch_results_v5_dir}/pytorch_v5_pngs.zip" "$nas_pytorch_results" && echo "Copied PyTorch V5 zip to $nas_pytorch_results."

# PyTorch V6
zip -j "${pytorch_results_v6_dir}/pytorch_v6_pngs.zip" $(find "$pytorch_results_v6_dir" -type f -name "*.png") && echo "Zipped PyTorch V6 PNGs without nesting."
sudo cp "${pytorch_results_v6_dir}/pytorch_v6_pngs.zip" "$nas_pytorch_results" && echo "Copied PyTorch V6 zip to $nas_pytorch_results."

echo "All zip files created and copied successfully."

# Function to copy video files to NAS
copy_video() {
  local source_video=$1
  local nas_dir=$2
  local video_name=$(basename "$source_video")

  # Copy the video file to the NAS location
  sudo cp "$source_video" "$nas_dir/$video_name"

  # Check if the copy operation was successful
  if [ $? -eq 0 ]; then
    echo "Successfully copied $video_name to $nas_dir"
  else
    echo "Error: Failed to copy the video file to $nas_dir."
    exit 1
  fi
}

# Copy video files for ONNX and PyTorch results (v5 and v6)
copy_video "$onnx_results_v5_video" "$nas_onnx_results"
copy_video "$onnx_results_v6_video" "$nas_onnx_results"
copy_video "$pytorch_results_v5_video" "$nas_pytorch_results"
copy_video "$pytorch_results_v6_video" "$nas_pytorch_results"
