#!/bin/bash

# Get the variables from the passed arguments
ROSBAG_DETINATION_PATH=$1
EXTRACTED_TOPICS_PATH=$2
STITCHED_PATH=$3

echo "Running the rosbag parsing script..."
python tools/projects/jac/rosbagParsing/rosbagParsing.py "$ROSBAG_DETINATION_PATH" "$EXTRACTED_TOPICS_PATH" -l
if [ $? -eq 0 ]; then
    echo "Rosbag parsing completed successfully."
else
    echo "Failed to run rosbag parsing script."
    exit 1
fi

echo "Running the Lidar preprocessing script..."
python tools/projects/jac/LidarPreprocess.py --dir_path "$EXTRACTED_TOPICS_PATH" --save_path "$STITCHED_PATH"
if [ $? -eq 0 ]; then
    echo "Lidar preprocessing completed successfully."
else
    echo "Failed to run Lidar preprocessing script."
    exit 1
fi
