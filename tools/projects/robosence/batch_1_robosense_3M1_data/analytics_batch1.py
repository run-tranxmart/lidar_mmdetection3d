import argparse
import math
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
from mmdet3d.structures.bbox_3d.lidar_box3d import LiDARInstance3DBoxes
from mmdet3d.structures.ops import box_np_ops
from tqdm import tqdm

META_INFO = {
    "categories": {
        "Pedestrian": 0,
        "Cyclist": 1,
        "Car": 2,
        "Truck": 3,
        "Misc": 4,
    },
    "dataset": "custom",
    "info_version": "1.0",
}
CLASS_MAP = {
    # IGNORE
    "unknown/unknown": -1,
    "animal/samll_animal": -1,
    "Non_vehicle/stroller": -1,
    "traffic_facility/barrier_gate": -1,
    # Pedestrian
    "pedestrian/pedestrian": 0,
    "pedestrian/stroller_user": 0,
    "pedestrian/wheelchair_user": 0,
    # Cyclist
    "Non_vehicle/bicycles": 1,
    "Non_vehicle/wheelchair": 1,
    "Non_vehicle/bicycle": 1,
    "Non_vehicle/tricycle": 1,
    "rider/bicyclist": 1,
    "rider/tricyclist": 1,
    # Car (Consider the length bigger than 6 will convert to Truck
    "service vehicle/service vehicle": 2,
    "vehicle/van": 2,
    "vehicle/car": 2,
    "vehicle/bus": 2,
    "vehicle/Campus Logistics Vehicle": 2,
    "vehicle/construction_vehicle": 2,
    "vehicle/emergency_vehicle": 2,
    "vehicle/truck": 2,
    # Misc
    "traffic_facility/road_barrier": 4,
    "traffic_facility/road_triangle": 4,
    "traffic_facility/crash_barrel": 4,
    "traffic_facility/cone": 4,
    "traffic_facility/bollard": 4,
    "traffic_facility/water_horse": 4,
}

CAMERA_INFOS = {
    "CAM0": {
        "cam2img": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "lidar2img": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    "CAM1": {
        "cam2imFg": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "lidar2img": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    "CAM2": {
        "img_path": "demo.jpg",
        "height": 427,
        "width": 640,
        "cam2img": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "lidar2img": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "lidar2cam": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    "CAM3": {
        "cam2img": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "lidar2img": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    "R0_rect": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
}

LIDAR_INFOS = {
    "num_pts_feats": 4,
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


def _calculate_num_points_in_gt(
    pcd_path: Path,
    annos: List[dict],
    num_features=4,
):
    points_v = np.fromfile(pcd_path, dtype=np.float32, count=-1).reshape(
        [-1, num_features]
    )

    bboxes_3d = np.array([ann["bbox_3d"] for ann in annos])

    indices = box_np_ops.points_in_rbbox(points_v[:, :3], bboxes_3d)
    num_points_in_gt = indices.sum(0)

    for i, ann in enumerate(annos):
        ann.update({"num_lidar_pts": num_points_in_gt[i]})
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
    bbox3d = data["bbox3d"]
    dimensions = data["dimensions"]
    yaw = data["rotation_y"]
    # Extract the required fields
    x, y, z = (
        bbox3d[0],
        bbox3d[1],
        bbox3d[2],
    )
    # dx, dy, dz = bbox3d["length"], bbox3d["width"], bbox3d["height"]
    dx, dy, dz = dimensions[0], dimensions[1], dimensions[2]

    # z = z - (dz / 2)
    # Create 3D bounding box array
    yaw = yaw + 1.57
    bbox_3d = [x, y, z, dx, dy, dz, yaw]
    bbox_3d = LiDARInstance3DBoxes(
        torch.tensor([bbox_3d]), with_yaw=True, origin=(0.5, 0.5, 0)
    )
    bbox_3d = bbox_3d.numpy()[0]
    # Get category label
    data["bbox_3d"] = bbox_3d
    category = data["category"]  # Extract class name from category string
    bbox_label_3d = data["bbox_label_3d"]  # Map category to label index
    if bbox_label_3d == 2 and bbox_3d[3] >= 6:
        bbox_label_3d = 3
    depth = z  # Typically, depth is the z-coordinate in many systems
    # Compute the bearing angle between the camera and the object
    bearing_angle = np.arctan2(x, y)  # Notice we use (x, z)

    # Calculate alpha (alpha = yaw - bearing_angle)
    alpha = yaw - bearing_angle
    # Normalize alpha to the range [-pi, pi]
    alpha = (alpha - np.pi / 2) % (2 * np.pi) - np.pi

    group_id = index
    # Assemble the final dictionary
    result = {
        "bbox": [
            0.0,
            0.0,
            0.0,
            0.0,
        ],  # 2D bounding box set to zero since we're focusing on 3D
        "bbox_label": bbox_label_3d,  # Assume bbox_label is the same as bbox_label_3d
        "bbox_3d": bbox_3d,
        "bbox_label_3d": bbox_label_3d,
        "depth": depth,
        "center_2d": [0.0, 0.0],  # Placeholder for 2D center
        "num_lidar_pts": data["num_points_in_gt"],
        "difficulty": data["difficulty"],  # Placeholder for difficulty
        "truncated": data["truncated"],
        "occluded": data["occluded"],
        # "alpha": data["alpha"],
        "alpha": alpha,
        "score": data["score"],  # Assuming score is 0 as it's not provided
        "index": index,
        "gt_ori_labels": category,
        "group_id": group_id,
    }

    return result


def parse_dataset_instance_info(dataset_instance, pcd_path, crowd=False):
    objects = []
    gt_info = {}
    if not pcd_path.exists():
        return None

    for idx in dataset_instance["annos"]["index"]:
        data = {}
        label = META_INFO["categories"].get(dataset_instance["annos"]["name"][idx], -1)
        if label == 3:
            label = 2
        data["bbox_label_3d"] = label
        data["bbox3d"] = dataset_instance["annos"]["location"][idx]
        data["dimensions"] = dataset_instance["annos"]["dimensions"][idx]
        data["rotation_y"] = dataset_instance["annos"]["rotation_y"][idx]
        data["num_points_in_gt"] = dataset_instance["annos"]["num_points_in_gt"][idx]
        data["alpha"] = dataset_instance["annos"]["alpha"][idx]
        data["truncated"] = dataset_instance["annos"]["truncated"][idx]
        data["occluded"] = dataset_instance["annos"]["occluded"][idx]
        data["score"] = dataset_instance["annos"]["score"][idx]
        data["difficulty"] = dataset_instance["annos"]["difficulty"][idx]
        data["category"] = dataset_instance["annos"]["name"][idx]
        gt_data_info = convert_to_bbox3d_format(data=data, index=idx)
        if gt_data_info["bbox_label_3d"] == -1:
            continue
        gt_data_info.update({"crowd": 0})
        objects.append(gt_data_info)

    if not len(objects):
        return None

    sample_idx = pcd_path.name  # split name with '/'
    gt_info["instances"] = objects
    gt_info["images"] = CAMERA_INFOS
    lidar_infos = LIDAR_INFOS.copy()
    lidar_infos["lidar_path"] = sample_idx
    gt_info["lidar_points"] = lidar_infos
    gt_info["sample_idx"] = sample_idx

    return gt_info


def process_paths(pcd_path, dataset_instance):
    return parse_dataset_instance_info(
        dataset_instance=dataset_instance, pcd_path=pcd_path, crowd=False
    )


def create_pickle_file(args):
    output_dir = Path(args.out_dir)
    raw_label_path = Path(args.raw_pickle_path)
    with open(raw_label_path, "rb") as f:
        raw_dataset = pickle.load(f)

    for i, data in enumerate(raw_dataset):
        raw_dataset[i]["velodyne_path"] = Path(args.pcd_dir).joinpath(
            Path(data["velodyne_path"]).name
        )
    raw_dataset = list(raw_dataset)
    split_sets = Path(args.split_sets)
    split_sets = list(split_sets.glob("*.txt"))

    for split in split_sets:
        # print("Processing", split.stem)
        with open(split, "r") as f:
            pcd_paths = f.read()

        pcd_paths = pcd_paths.split("\n")
        pcd_paths = list(filter(lambda x: x.endswith(".bin"), pcd_paths))
        pcd_abs_paths = list(map(lambda x: Path(args.pcd_dir).joinpath(x), pcd_paths))
        data_list = []
        for raw_dataset_instance in tqdm(raw_dataset):
            raw_dataset_pcd_path = raw_dataset_instance["velodyne_path"]
            try:
                index = pcd_abs_paths.index(raw_dataset_pcd_path)
                pcd_path = pcd_abs_paths[index]
            except ValueError:
                continue
            raw_dataset_instance["velodyne_path"] = pcd_path
            ann = parse_dataset_instance_info(
                dataset_instance=raw_dataset_instance, pcd_path=pcd_path, crowd=False
            )
            if ann:
                data_list.append(ann)
        data_infos = {"metainfo": META_INFO, "data_list": data_list}
        output_file = output_dir.joinpath(split.stem + "_data_infos.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(data_infos, f)
        saved_name = output_file.resolve().absolute().as_posix()
        print(f"{split.stem} data information saved on {saved_name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Get directories for training, validation, and data root."
    )

    parser.add_argument(
        "--out_dir",
        "-o",
        type=str,
        required=False,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--raw_pickle_path",
        "-r",
        type=str,
        required=False,
        help="Path to the root json directory.",
    )
    parser.add_argument(
        "--split_sets", "-s", type=str, required=True, help="Path the json root"
    )
    parser.add_argument(
        "--pcd_dir",
        "-p",
        type=str,
        required=True,
        help="Path the point cloud directory",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("Batch 1 data converting")
    args = parse_args()
    print(args)
    data_infos = create_pickle_file(args)
