from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
import math
import json
import pickle
from mmdet3d.structures.bbox_3d.lidar_box3d import LiDARInstance3DBoxes
import torch
import argparse
from tqdm import tqdm
from mmdet3d.structures.ops import box_np_ops
from pathlib import Path
from glob import glob

from typing import List

META_INFO = {
    'categories':
        {
            'Pedestrian': 0,
            'Cyclist': 1,
            'Car': 2,
            'Truck': 3,
            'Static': 4,
        },
    'dataset': 'custom',
    'info_version': '1.0'
}
CLASS_MAP = {
    # IGNORE
    'unknown/unknown': -1,
    'animal/samll_animal': -1,
    'Non_vehicle/stroller': -1,
    'traffic_facility/barrier_gate': -1,

    # Pedestrian
    'pedestrian/pedestrian': 0,
    'pedestrian/stroller_user': 0,
    'pedestrian/wheelchair_user': 0,

    # Cyclist
    'Non_vehicle/bicycles': 1,
    'Non_vehicle/wheelchair': 1,
    'Non_vehicle/bicycle': 1,
    'Non_vehicle/tricycle': 1,
    'rider/bicyclist': 1,
    'rider/tricyclist': 1,

    # Car (Consider the length bigger than 6 will convert to Truck
    'service vehicle/service vehicle': 2,
    'vehicle/van': 2,
    'vehicle/car': 2,
    'vehicle/bus': 2,
    'vehicle/Campus Logistics Vehicle': 2,
    'vehicle/construction_vehicle': 2,
    'vehicle/emergency_vehicle': 2,
    'vehicle/truck': 2,
    # Misc
    'traffic_facility/road_barrier': 4,
    'traffic_facility/road_triangle': 4,
    'traffic_facility/crash_barrel': 4,
    'traffic_facility/cone': 4,
    'traffic_facility/bollard': 4,
    'traffic_facility/water_horse': 4
}

CAMERA_INFOS = {
    'CAM0': {'cam2img': [[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]],
             'lidar2img': [[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]]},
    'CAM1': {'cam2imFg': [[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]],
             'lidar2img': [[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]]},
    'CAM2': {'img_path': 'demo.jpg',
             'height': 427,
             'width': 640,
             'cam2img': [[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]],
             'lidar2img': [[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]],
             'lidar2cam': [[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]]},
    'CAM3': {'cam2img': [[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]],
             'lidar2img': [[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]]},
    'R0_rect': [[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]]
}

LIDAR_INFOS = {
    'num_pts_feats': 4,
    # 'lidar_path': [],
    # 'Tr_velo_to_cam': [[1.0, 0.0, 0.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0],
    # [0.0, 0.0, 1.0, 0.0],
    # [0.0, 0.0, 0.0, 1.0]],
    # 'Tr_imu_to_velo': [[1.0, 0.0, 0.0, 0.0],
    # [0.0, 1.0, 0.0, 0.0],
    # [0.0, 0.0, 1.0, 0.0],
    # [0.0, 0.0, 0.0, 1.0]]
}


def quaternion_to_yaw(qw, qx, qy, qz):
    # Calculate yaw from quaternion
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    # yaw += np.pi / 2
    return yaw


def _calculate_num_points_in_gt(pcd_path: Path,
                                annos: List[dict],
                                num_features=4,
                                ):
    points_v = np.fromfile(
        pcd_path, dtype=np.float32, count=-1).reshape([-1, num_features])

    bboxes_3d = np.array([ann['bbox_3d'] for ann in annos])

    indices = box_np_ops.points_in_rbbox(points_v[:, :3], bboxes_3d)
    num_points_in_gt = indices.sum(0)

    for i, ann in enumerate(annos):
        ann.update({'num_lidar_pts': num_points_in_gt[i]})
    return annos


def convert_to_bbox3d_format(data: dict, index: int):
    """
    Convert a single sample dictionary into the desired format.

    Args:
        pcd_path: Path to the point cloud file.
        data (dict): A dictionary containing bounding box information.
        index (int): The index of the sample (for `index` field in output).
        group_id (int): The group ID for the sample.

    Returns:
        dict: A dictionary in the desired output format.
    """
    # Convert quaternion to yaw
    bbox3d = data['bbox3d']
    yaw = bbox3d["yaw"]
    # Extract the required fields
    # x, y, z = bbox3d['position_x'], bbox3d['position_y'], bbox3d['position_z'] + 0.8 - bbox3d["height"] / 2
    x, y, z = bbox3d['position_y'], - bbox3d['position_x'], bbox3d['position_z'] - 1.3 - bbox3d["height"] / 2
    # dx, dy, dz = bbox3d["length"], bbox3d["width"], bbox3d["height"]
    dy, dx, dz = bbox3d["length"], bbox3d["width"], bbox3d["height"]

    # z = z - (dz / 2)
    # Create 3D bounding box array
    yaw = yaw - 1.57
    bbox_3d = [x, y, z, dy, dx, dz, yaw]
    bbox_3d = LiDARInstance3DBoxes(torch.tensor([bbox_3d]),
                                   with_yaw=True, origin=(0.5, 0.5, 0)
                                   )

    # print(len9(bbox_3d))
    # print("Before rotation", bbox_3d)
    bbox_3d.rotate(angle=1.57)
    # bbox_3d.flip()
    bbox_3d = bbox_3d.numpy()[0]
    # Get category label
    data["bbox_3d"] = bbox_3d
    category = data['category']  # Extract class name from category string
    bbox_label_3d = CLASS_MAP.get(category, -1)  # Map category to label index
    if bbox_label_3d == 2 and bbox_3d[3] >= 6:
        bbox_label_3d = 3
    depth = z  # Typically, depth is the z-coordinate in many systems
    # Compute the bearing angle between the camera and the object
    bearing_angle = np.arctan2(x, y)  # Notice we use (x, z)

    # Calculate alpha (alpha = yaw - bearing_angle)
    alpha = yaw - bearing_angle
    # Normalize alpha to the range [-pi, pi]
    alpha = (alpha - np.pi / 2) % (2 * np.pi) - np.pi

    num_lidar_pts = 0

    # occluded = data['Occlusion']
    # truncated = data['Truncation']
    group_id = index
    # Assemble the final dictionary
    result = {
        'bbox': [0.0, 0.0, 0.0, 0.0],  # 2D bounding box set to zero since we're focusing on 3D
        'bbox_label': bbox_label_3d,  # Assume bbox_label is the same as bbox_label_3d
        'bbox_3d': bbox_3d,
        'bbox_label_3d': bbox_label_3d,
        'depth': depth,
        'center_2d': [0.0, 0.0],  # Placeholder for 2D center
        'num_lidar_pts': num_lidar_pts,
        'difficulty': 0,  # Placeholder for difficulty
        'truncated': 0,
        'occluded': 0,
        'alpha': alpha,
        'score': 0.0,  # Assuming score is 0 as it's not provided
        'index': index,
        'gt_ori_labels': data['category'],
        'group_id': group_id
    }

    return result


def parse_json_infos(json_path, pcd_path, crowd=False):
    with open(json_path, 'r') as f:
        data = json.load(f)
    objects = []
    gt_info = {}
    if not pcd_path.exists():
        return None

    for idx, obj in enumerate(data['items']):
        obj["category"] = f"{obj['category']}/{obj['subcategory']}"

        gt_data_info = convert_to_bbox3d_format(data=obj, index=idx)
        if gt_data_info['bbox_label_3d'] == -1:
            continue
        gt_data_info.update({'crowd': 0})
        objects.append(gt_data_info)

    if not len(objects):
        return None

    try:
        objects = _calculate_num_points_in_gt(pcd_path=pcd_path, annos=objects)
    except Exception as e:
        pass

    sample_idx = pcd_path.name  # split name with '/'
    print(sample_idx)
    gt_info['instances'] = objects
    gt_info['images'] = CAMERA_INFOS
    lidar_infos = LIDAR_INFOS.copy()
    lidar_infos['lidar_path'] = sample_idx
    gt_info['lidar_points'] = lidar_infos
    gt_info['sample_idx'] = sample_idx

    return gt_info


def process_paths(pcd_path, json_path):
    return parse_json_infos(json_path=json_path, pcd_path=pcd_path, crowd=False)


def create_pickle_file(args):
    json_dir = Path(args.json_dir)
    output_dir = Path(args.out_dir)

    split_sets = Path(args.split_sets)
    split_sets = list(split_sets.glob('*.txt'))

    for split in split_sets:
        print("Processing", split.stem)
        with open(split, 'r') as f:
            pcd_paths = f.read()

        pcd_paths = pcd_paths.split('\n')
        pcd_paths = list(filter(lambda x: x.endswith('.bin'), pcd_paths))
        pcd_abs_paths = list(map(lambda x: Path(args.pcd_dir).joinpath(x), pcd_paths))
        json_abs_paths = list(map(lambda x: json_dir.joinpath(x.replace('.bin', '.json')), pcd_paths))
        data_list = []
        for pcd_path, json_path in tqdm(zip(pcd_abs_paths, json_abs_paths), total=len(pcd_abs_paths)):
             ann = parse_json_infos(json_path=json_path, pcd_path=pcd_path, crowd=False)
             if ann:
                data_list.append(ann)
        # with ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(process_paths, pcd, json) for pcd, json in zip(pcd_abs_paths, json_abs_paths)]
        #     for future in tqdm(futures):
        #         result = future.result()
        #         if len(result):
        #             data_list.append(result)
        #with ProcessPoolExecutor() as executor:
        #    futures = [executor.submit(process_paths, pcd, json) for pcd, json in zip(pcd_abs_paths, json_abs_paths)]
        #    for future in tqdm(futures):
        #        result = future.result()
        #        if result is not None:
        #            data_list.append(result)

        data_infos = {
            'metainfo': META_INFO,
            'data_list': data_list
        }

        output_file = output_dir.joinpath(split.stem + '_data_infos.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(data_infos, f)
        saved_name = output_file.resolve().absolute().as_posix()
        print(f"{split.stem} data information saved on {saved_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Get directories for training, validation, and data root.")

    parser.add_argument('--out_dir', '-o', type=str, required=False, help='Path to the output directory.')
    parser.add_argument('--json_dir', '-j', type=str, required=False, help='Path to the root json directory.')
    parser.add_argument('--split_sets', '-s', type=str, required=True, help='Path the json root')
    parser.add_argument('--pcd_dir', '-p', type=str, required=True, help='Path the point cloud directory')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("Batch 2 data converting")
    args = parse_args()
    print(args)
    data_infos = create_pickle_file(args)
