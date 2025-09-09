#!/bin/bash
# Check if both required arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 /path/to/images_path /path/to/output_video"
    exit 1
fi

# Get the input arguments
IMAGES_PATH=$1
OUTPUT_VIDEO_PATH=$2

# Print out the arguments for confirmation (optional)
echo "IMAGES_PATH: $IMAGES_PATH"
echo "OUTPUT_VIDEO_PATH: $OUTPUT_VIDEO_PATH"

# Step 2: Run the video_demo.py script for ONNX results v5
echo "Generating video for $IMAGES_PATH..."
python tools/projects/jac/video_demo.py --image_dir "$IMAGES_PATH" --output_video "$OUTPUT_VIDEO_PATH"
if [ $? -eq 0 ]; then
    echo "Video generated successfully $IMAGES_PATH."
else
    echo "Failed to generate video $IMAGES_PATH."
    exit 1
fi

