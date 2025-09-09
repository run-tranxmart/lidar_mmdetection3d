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

from mmengine.structures import InstanceData
from mmdet3d.datasets.transforms.loading import LoadPointsFromFile
from mmdet3d.visualization import (Det3DLocalVisualizer, BEVLocalVisualizer)
from mmdet3d.structures import (LiDARInstance3DBoxes, Det3DDataSample)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize the points and bbox')
    parser.add_argument(
        '--cloud', type=str, default=None, help='Path of cloud')
    parser.add_argument(
        '--pkl_path',
        type=str,
        default=None,
        help='specify the pkl path of dataset')
    parser.add_argument(
        '--save',
        type=str,
        default="visual",
        help='Save path of visualization')
    parser.add_argument(
        "--bev",
        action="store_true",
        default=False,
        help="Use BEVLocalVisualizer",
    )
    parser.add_argument(
        "--open3d",
        action="store_true",
        default=False,
        help="Use Det3DLocalVisualizer",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="aicc",
        help='Project (AICC or JAC)'
    )
    parser.add_argument(
        "--save_pkl",
        action="store_true",
        default=False,
        help="Save bboxes_3d lables_3d point_cloud for local visualization with open3d",
    )
    args = parser.parse_args()
    return args


def get_cloud_list(cloud_pth=None):
    cloud_list = []
    if osp.isdir(cloud_pth):
        frame_list = sorted(os.listdir(cloud_pth))
        cloud_list = [osp.join(cloud_pth, fn) for fn in frame_list]
    elif osp.isfile(cloud_pth):
        cloud_list = [cloud_pth]
    else:
        raise FileExistsError(f'Cannot find {cloud_pth}')
    return cloud_list


def get_pkl_info(pkl_path=None):
    '''
        Load the pkl infos and transform the list to dict 
        in which the keys are sample_idx
    '''
    pkl_list = []
    if osp.exists(pkl_path):
        print(f'Loading pkl: {pkl_path}')
        with open(pkl_path, 'rb') as f:
            pkl_label = pkl.load(f)
            pkl_list = pkl_label['data_list']
    pkl_dict = {}
    for frame_info in pkl_list:
        frame_name = frame_info['sample_idx']
        if frame_name.endswith('.bin') or frame_name.endswith('.pcd'):
            frame_name = frame_name.rsplit('.', 1)[0]
            if "/" in frame_name:
                frame_name = frame_name.rsplit('/', 1)[-1]
        pkl_dict[frame_name] = frame_info['instances']
    return pkl_list, pkl_dict


def open3d_visual(frame_name, cloud_points, pkl_annos, visualizer, save_dir):
    print("If switch to next frame, press Right")
    visualizer.set_points(cloud_points)
    bboxes_3d = []
    if frame_name in pkl_annos:
        frame_instances = pkl_annos[frame_name]
        for obj in frame_instances:
            bboxes_3d.append(obj['bbox_3d'])
    else:
        print(f'Cannot find {frame_name} in pkl annos')
    if len(bboxes_3d) > 0:
        bboxes_3d = LiDARInstance3DBoxes(torch.from_numpy(np.array(bboxes_3d)))
        
        visualizer.draw_bboxes_3d(bboxes_3d)
    save_pth = osp.join(save_dir, frame_name + '.png')
    visualizer.show(save_path=save_pth, win_name=frame_name)


def bev_visual(frame_name, cloud_points, pkl_annos, visualizer, save_dir):
    frame_labels = []
    frame_bboxes = []
    frame_scores = []
    frame_crowd = []
    if frame_name in pkl_dict:
        cloud_annos = pkl_annos[frame_name]
        for obj in cloud_annos:
            frame_labels.append(obj['bbox_label_3d'])
            frame_bboxes.append(obj['bbox_3d'])
            frame_scores.append(1.0)
            if 'crowd' in obj.keys():
                frame_crowd.append(obj['crowd'])
    else:
        print(f'Cannot find {frame_name} in pkl annos')

    frame_labels = np.array(frame_labels, dtype=np.int16)
    frame_bboxes = np.array(frame_bboxes, dtype=np.float32)
    frame_scores = np.array(frame_scores, dtype=np.float32)

    frame_instances_3d = InstanceData()
    frame_instances_3d.labels_3d = torch.tensor(frame_labels)
    frame_instances_3d.bboxes_3d = LiDARInstance3DBoxes(
        torch.tensor(frame_bboxes))
    frame_instances_3d.scores_3d = torch.tensor(frame_scores)
    frame_gts = Det3DDataSample(gt_instances_3d=frame_instances_3d)
    frame_input = dict(points=cloud_points)
    save_pth = osp.join(save_dir, frame_name + '.png')
    visualizer.add_datasample(
        name='Visual GT',
        data_sample=frame_gts,
        data_input=frame_input,
        data_crowd=frame_crowd,
        show=False,
        save_file=save_pth,
        vis_task='lidar_det',
        draw_gt=True,
        draw_pred=False,
        draw_distance=True,
        draw_det_scope=False)
    if args.save_pkl :
        if frame_name in pkl_dict :
            temp = {'bboxes_3d':frame_gts.gt_instances_3d.bboxes_3d.numpy(),
                    'labels_3d':frame_gts.gt_instances_3d.labels_3d.numpy(),
                    'scores_3d':frame_gts.gt_instances_3d.scores_3d.numpy(),
                    'points':cloud_points}
            with open(osp.join(save_dir, frame_name + '.pkl'), 'wb') as f:
                pkl.dump(temp,f)
                f.close()
    


if __name__ == '__main__':
    args = parse_args()

    if args.project == "aicc":
        vis_range = [[-72, 160], [-72, 72], [-5, 5]]
        det_range = [[-100, 156], [-28.8, 28.8]]
    elif args.project == "jac":
        vis_range = [[-72, 92], [-72, 72], [-5, 5]]
        det_range = [[-16, 80], [-16, 16]]
    
    if args.pkl_path is not None:
        pkl_list, pkl_dict = get_pkl_info(args.pkl_path)

    cloud_list = get_cloud_list(args.cloud)
    cloud_loader = LoadPointsFromFile(
        coord_type='LIDAR', load_dim=4, use_dim=4)
    os.makedirs(args.save, exist_ok=True)

    if args.bev:
        visualizer = BEVLocalVisualizer(
            area_scope = vis_range,
            det_scope = det_range,
        )
    elif args.open3d:
        visualizer = Det3DLocalVisualizer()

    for i, cloud_pth in enumerate(tqdm(cloud_list)):
        cloud_name = cloud_pth.rsplit('/')[-1]
        frame_name = cloud_name.rsplit('.', 1)[0]
        cloud_points = cloud_loader.load_points(cloud_pth)
        if args.bev:
            bev_visual(frame_name, cloud_points, pkl_dict, visualizer,
                       args.save)
        elif args.open3d:
            open3d_visual(frame_name, cloud_points, pkl_dict, visualizer,
                          args.save)
        else:
            print('Only support BEV or Open3D Visualization')
 