import pickle
from os import path as osp
import numpy as np
import json
from shapely.geometry import Polygon, Point
import mmengine
from mmengine import init_default_scope, track_iter_progress
from mmdet3d.registry import DATASETS
from mmdet3d.structures.ops import box_np_ops
import argparse

init_default_scope('mmdet3d')

def extract_area_from_mask(data_path: str,
                           info_path: str,
                           mask: Polygon,
                           save_path: str,):
    """Given the raw data, generate the ground truth database.

    Args:
        data_path (str): Path of the data.
        info_path (str, optional): Path of the info file.
            Default: None.
        mask(Polygon): A polygon mask to mask data.
        save_path(str): Path to save data.
    """
    dataset_cfg = dict(
        type='Kitti_5Class_Dataset', data_root=data_path, ann_file=info_path)

    backend_args = None
    dataset_cfg.update(
        modality=dict(
            use_lidar=True,
            use_camera=False,
        ),
        data_prefix=dict(pts=''),
        convert_cam_to_lidar=False,
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                backend_args=backend_args),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                backend_args=backend_args)
        ])

    dataset = DATASETS.build(dataset_cfg)

    database_save_path = osp.join(save_path, 'short_range_cloud_database')
    db_info_save_path = osp.join(save_path, 'short_range_cloud_infos_train.pkl')

    mmengine.mkdir_or_exist(database_save_path)
    data_list = []

    for j in track_iter_progress(list(range(len(dataset)))):

        data_info = dataset.get_data_info(j)
        example = dataset.pipeline(data_info)
        annos = example['ann_info']
        points = example['points'].numpy()
        gt_boxes_3d = annos['gt_bboxes_3d'].numpy()
        labels = annos['gt_labels_3d']
        alpha = annos['alpha']
        index = annos['index']
        group_id = annos['group_id']
        gt_ori_labels = annos['gt_ori_labels']
        crowd = annos['crowd']

        name =example['lidar_path'].split("/")[-1][:-4]
        filename = f'{name}_shortrange.bin'
        abs_filepath = osp.join(database_save_path, filename)
        rel_filepath = osp.join('short_range_cloud_database', filename)

        points_inside_mask = [Point(p[:2]).within(mask) for p in points]
        masked_points = points[points_inside_mask]

        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)
        instances = []

        for i, box in enumerate(gt_boxes_3d):

            b_center = Point(box[0], box[1])
            if b_center.within(mask):

                points_in_box = points[point_indices[:, i]]
                masked_points = np.concatenate((masked_points, points_in_box), axis=0)
                masked_points = np.unique(masked_points, axis=0)
                instances.append({
                    "bbox": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    "bbox_label": labels[i],
                    "bbox_3d": box,
                    "bbox_label_3d": labels[i],
                    "depth": 0.0, 
                    "center_2d": [0.0, 0.0],  
                    "num_lidar_pts": len(points[point_indices[:, i]]),
                    "difficulty": 0,
                    "truncated": 0,
                    "occluded": 0,
                    "alpha": alpha[i],
                    "score": 0.0,
                    "index": index[i],
                    "gt_ori_labels": gt_ori_labels[i],
                    "crowd": crowd[i],
                    "group_id": group_id[i],
                })
        data_list.append({
            "instances": instances,
            "images": {},
            "lidar_points": {'num_pts_feats': 4,
                            'lidar_path': rel_filepath},
            "sample_idx": rel_filepath,
        },)

        with open(abs_filepath, 'w') as f:
            masked_points.tofile(f)

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(data_list, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract the area in the mask as a new dataset.")   
    parser.add_argument('data_path', type=str, help='Path to the input data.')
    parser.add_argument('pkl_path', type=str, help='Path to the input pickle file.')
    parser.add_argument('save_path', type=str, help='Directory to save the outputs.')
    parser.add_argument('mask_path', type=str, help='Path to the mask json file.')
    args = parser.parse_args()

    with open(args.mask_path, "r") as f:
        mask = json.load(f)
    mask_polygon = Polygon(mask["mask_points"])

    extract_area_from_mask(data_path = args.data_path,
                           info_path = args.pkl_path,
                           mask = mask_polygon,
                           save_path = args.save_path)