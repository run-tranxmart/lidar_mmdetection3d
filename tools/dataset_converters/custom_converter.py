# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from pathlib import Path
import os.path as osp

import mmcv
import mmengine
import numpy as np
from nuscenes.utils.geometry_utils import view_points

from mmdet3d.structures import points_cam2img
from mmdet3d.structures.ops import box_np_ops
from .custom_data_utils import get_custom_frame_info
# from .nuscenes_converter import post_process_coords

# custom_categories = ('Pedestrian', 'Cyclist', 'Car')


def _read_split_file(path):
    if not osp.exists(path):
        return []
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def _calculate_num_points_in_gt(data_path,
                                infos,
                                relative_path,
                                num_features=4):
    for info in mmengine.track_iter_progress(infos):
        pc_info = info['point_cloud']
        if relative_path:
            v_path = str(Path(data_path) / pc_info['velodyne_path'])
        else:
            v_path = pc_info['velodyne_path']
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape([-1, num_features])

        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)


def create_custom_info_file(data_path,
                            pkl_prefix='custom',
                            save_path=None,
                            relative_path=True):
    """Create info file of Custom dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str, optional): Prefix of the info file to be generated.
            Default: 'custom'.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
        save_path (str, optional): Path to save the info file.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
    """
    split_folder = Path(data_path) / 'split_set'
    train_frame_ids = _read_split_file(str(split_folder / 'train.txt'))
    val_frame_ids = _read_split_file(str(split_folder / 'val.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)

    # Generate train info
    if len(train_frame_ids) > 0:
        custom_infos_train = get_custom_frame_info(
            data_path, frame_list=train_frame_ids, relative_path=relative_path)
        _calculate_num_points_in_gt(data_path, custom_infos_train,
                                    relative_path)
        filename = save_path / f'{pkl_prefix}_infos_train.pkl'
        print(f'Custom info train file is saved to {filename}')
        mmengine.dump(custom_infos_train, filename)
    else:
        print('No frames in training set')

    # Generate validation info
    if len(val_frame_ids) > 0:
        custom_infos_val = get_custom_frame_info(
            data_path, frame_list=val_frame_ids, relative_path=relative_path)
        _calculate_num_points_in_gt(data_path, custom_infos_val, relative_path)
        filename = save_path / f'{pkl_prefix}_infos_val.pkl'
        print(f'Custom info val file is saved to {filename}')
        mmengine.dump(custom_infos_val, filename)

        filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
        print(f'Custom info trainval file is saved to {filename}')
        mmengine.dump(custom_infos_train + custom_infos_val, filename)
    else:
        print('No frame in val set')
