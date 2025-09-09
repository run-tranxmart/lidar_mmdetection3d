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

def visual_cloud_with_results(cloud_dir, result_dir, save_dir):
    bev_visualizer = BEVLocalVisualizer(
        # Default parameters for AICC project
        score_thresh=[0.4, 0.4, 0.4, 0.4, 0.4],
        class_names=['Pedestrian', 'Cyclist', 'Car', 'Truck', 'Misc'],
        det_scope=[-100, -28.8, -4, 156, 28.8, 3],
        area_scope=[[-72, 156], [-72, 72], [-5, 5]],
    )

    cloud_list = sorted(os.listdir(cloud_dir))
    cloud_suffix = '.bin'
    if len(cloud_list) > 0:
        cloud_suffix = cloud_list[0].rsplit('.')[-1]

    cloud_loader = LoadPointsFromFile(coord_type='LIDAR', load_dim=4, use_dim=4)

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    for i, path in enumerate(tqdm(os.listdir(result_dir))):
        cloud_pth = Path(cloud_dir).joinpath(Path(path))
        cloud_pth = osp.join(cloud_dir, path.replace('json', cloud_suffix))
        result_path = osp.join(result_dir, path)

        with open(result_path, 'r') as f:
            cloud_result = json.load(f)

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
            name='model_infer',
            data_sample=frame_dets,
            data_input=frame_input,
            show=False,
            save_file=save_pth,
            vis_task='lidar_det',
            draw_gt=False,
            draw_pred=True,
            draw_distance=True,
            draw_det_scope=True,
            draw_score=True,    
        )


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
    first_image_path = osp.join(image_dir, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the video output path in the same directory
    output_video = osp.join(image_dir, "output_video.mp4")

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for .mp4 output
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write each image to the video
    for image in images[:300]:  # Cap at 300 images for demonstration purposes
        image_path = osp.join(image_dir, image)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video saved at {output_video}")

def main():
    parser = argparse.ArgumentParser(description="Create images and videos from cloud and JSON data.")
    parser.add_argument('--cloud', type=str, required=True, help='Directory of cloud files (bin or pcd format).')
    parser.add_argument('--result', type=str, required=True, help='Directory of result files (json format).')
    parser.add_argument('--save', type=str, default="save_image", help='Directory of saved images.')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for the output video.')
    args = parser.parse_args()

    visual_cloud_with_results(args.cloud, args.result, args.save)
    create_video_from_images(args.save, args.fps)

if __name__ == "__main__":
    main()
