import argparse
import copy
import torch
import numpy as np
import os
from os import path as osp
import pickle as pkl
from pathlib import Path
import time
from tqdm import tqdm
import warnings
import cv2

from mmengine.structures import InstanceData
from mmdet3d.datasets.transforms.loading import LoadPointsFromFile
from mmdet3d.visualization import (Det3DLocalVisualizer, BEVLocalVisualizer)
from mmdet3d.structures import (LiDARInstance3DBoxes, Det3DDataSample)

import json

def create_images(cloud_dir, json_files, save_dir):
    bev_visualizer = BEVLocalVisualizer()
    cloud_loader = LoadPointsFromFile(
            coord_type='LIDAR', load_dim=4, use_dim=4)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for path in os.listdir(json_files):
        cloud_pth = Path(cloud_dir).joinpath(Path(path))
        cloud_pth = os.path.join(cloud_dir, path.replace('.json', '.bin'))
        result_path = os.path.join(json_files, path)
        print(result_path)

        with open(result_path, 'r') as f:
            cloud_result = json.load(f)

        # breakpoint()
        frame_points = cloud_loader.load_points(cloud_pth)
        frame_input = dict(points=frame_points)
        frame_name = osp.splitext(osp.basename(cloud_pth))[0]
        save_pth = osp.join(save_dir, frame_name + '.png')

        frame_instances_3d = InstanceData()
        frame_instances_3d.bboxes_3d = LiDARInstance3DBoxes(
            torch.tensor(cloud_result['bboxes_3d']))
        frame_instances_3d.scores_3d = torch.tensor(cloud_result['scores_3d'])
        frame_instances_3d.labels_3d = torch.tensor(cloud_result['labels_3d'])
        frame_dets = Det3DDataSample(pred_instances_3d=frame_instances_3d)

        bev_visualizer.add_datasample(
            name='onnx_infer',
            data_sample=frame_dets,
            data_input=frame_input,
            show=False,
            save_file=save_pth,
            vis_task='lidar_det',
            draw_gt=False,
            draw_pred=True,
            draw_distance=True,
            draw_det_scope=False)

        # Add filename text to image
        add_filename_to_image(save_pth, frame_name)

def add_filename_to_image(image_path, filename_text):
    """
    Add the filename as a text overlay on the image.
    Args:
        image_path (str): Path to the image file.
        filename_text (str): Text to overlay on the image.
    """
    image = cv2.imread(image_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, filename_text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(image_path, image)

def create_video_from_images(image_dir, fps=10):
    """
    Create a video from a sequence of images in the specified directory.
    Args:
        image_dir (str): Directory containing the images.
        fps (int): Frames per second for the output video.
    """
    # Get list of image files
    images = [img for img in os.listdir(image_dir) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()  # Sort images by name to maintain the correct sequence

    if not images:
        print("No images found in directory.")
        return

    # Read the first image to get the frame size
    first_image_path = os.path.join(image_dir, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the video output path in the same directory
    output_video = osp.join(image_dir, "output_video.mp4")

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for .mp4 output
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write each image to the video
    for image in images[:300]:  # Cap at 300 images for demonstration purposes
        image_path = os.path.join(image_dir, image)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video saved at {output_video}")

def main():
    parser = argparse.ArgumentParser(description="Create images and videos from cloud and JSON data.")
    parser.add_argument('--cloud', type=str, required=True, help='Directory containing cloud files (bin format).')
    parser.add_argument('--json_files', type=str, required=True, help='Directory containing JSON result files.')
    parser.add_argument('--save', type=str, required=True, help='Directory to save generated images.')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for the output video.')

    args = parser.parse_args()

    create_images(args.cloud, args.json_files, args.save)
    create_video_from_images(args.save, args.fps)

if __name__ == "__main__":
    main()
