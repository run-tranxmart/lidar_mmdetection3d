# Copyright (c) Softmotion. All rights reserved.
import argparse
import glob
import json
import numpy as np
import onnxruntime
import os
import os.path as osp
import pickle
from preprocess import Preprocess
from postprocess import Postprocess
import warnings
import torch
import tqdm
from mmengine.structures import InstanceData
from mmdet3d.models.middle_encoders import PointPillarsScatter
from mmdet3d.visualization import BEVLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes, Det3DDataSample
from mmdet3d.datasets.transforms.loading import LoadAnnotations3D, LoadPointsFromFile
from pathlib import Path
from typing import List
from mmdet3d.structures.ops import box_np_ops


from mmengine.config import Config


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

ID_TO_NAME = {v: k for k, v in META_INFO["categories"].items()}
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
}


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


def parse_json_infos(result_dict, pcd_path, crowd=False):
    objects = []
    gt_info = {}
    pcd_path = Path(pcd_path)
    if not pcd_path.exists():
        return None
    idx = 0
    for bbox_3d, label_id, score, direction in zip(
        result_dict["bboxes_3d"],
        result_dict["labels_3d"],
        result_dict["scores_3d"],
        result_dict["dir"],
    ):
        if score < 0.5:
            continue

        obj = {
            "category": ID_TO_NAME[label_id],
            "bbox3d": bbox_3d,
            "label": label_id,
        }

        gt_data_info = convert_to_bbox3d_format(data=obj, index=idx)
        if gt_data_info["bbox_label_3d"] == -1:
            continue
        # gt_data_info.update({"crowd": 0})
        objects.append(gt_data_info)
        idx += 1
    if not len(objects):
        return None

    try:
        objects = _calculate_num_points_in_gt(pcd_path=pcd_path, annos=objects)
    except Exception:
        pass

    sample_idx = pcd_path.name  # split name with '/'
    gt_info["instances"] = objects
    gt_info["images"] = CAMERA_INFOS
    lidar_infos = LIDAR_INFOS.copy()
    lidar_infos["lidar_path"] = sample_idx
    gt_info["lidar_points"] = lidar_infos
    gt_info["sample_idx"] = sample_idx

    return gt_info


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
    bbox3d = data["bbox3d"]
    yaw = bbox3d[6]

    bbox_3d = LiDARInstance3DBoxes(
        torch.tensor([bbox3d]), with_yaw=True, origin=(0.5, 0.5, 0)
    )

    bbox_3d = bbox_3d.numpy()[0]
    # Get category label
    bbox_label_3d = data["label"]  # Map category to label index
    if bbox_label_3d == 2 and bbox_3d[3] >= 6:
        bbox_label_3d = 3
    depth = bbox3d[2]  # Typically, depth is the z-coordinate in many systems
    bearing_angle = np.arctan2(bbox3d[0], bbox3d[1])  # Notice we use (x, z)
    alpha = yaw - bearing_angle
    # Normalize alpha to the range [-pi, pi]
    alpha = (alpha - np.pi / 2) % (2 * np.pi) - np.pi

    num_lidar_pts = 0

    group_id = index
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
        "num_lidar_pts": num_lidar_pts,
        "difficulty": 0,  # Placeholder for difficulty
        "truncated": 0,
        "occluded": 0,
        "alpha": alpha,
        "score": 0.0,  # Assuming score is 0 as it's not provided
        "index": index,
        "gt_ori_labels": data["category"],
        "crowd": 1 if data["category"] == "Non_vehicle/bicycles" else 0,
        "group_id": group_id,
    }

    return result


def create_pickle_file(cloud_list, result_list, output_dir):
    output_dir = Path(output_dir)
    data_list = []
    for pcd_path, result in tqdm.tqdm(
        zip(cloud_list, result_list), total=len(cloud_list)
    ):
        ann = parse_json_infos(result_dict=result, pcd_path=pcd_path, crowd=False)
        if ann:
            data_list.append(ann)

    data_infos = {"metainfo": META_INFO, "data_list": data_list}

    output_file = output_dir.joinpath("result_data_infos.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(data_infos, f)
    saved_name = output_file.resolve().absolute().as_posix()
    print(f"data information saved on {saved_name}")


class PointCloudParameters:
    def __init__(
        self, max_num_points, max_voxels, load_dim, use_dim, cloud_range, voxel_size
    ) -> None:
        # define the point cloud parameters here
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.cloud_range = cloud_range
        self.voxel_size = voxel_size
        self.dx = self.cloud_range[3] - self.cloud_range[0]
        self.dy = self.cloud_range[4] - self.cloud_range[1]
        self.feat_size = [
            round(self.dy / self.voxel_size[1]),
            round(self.dx / self.voxel_size[0]),
            1,
        ]

    def __str__(self):
        return f"Cloud Range: {self.cloud_range} \
                 Voxel Size: {self.voxel_size}  \
                 Feat Size: {self.feat_size}"


def load_data(args):
    # load the cloud
    cloud_list = []
    if osp.isdir(args.cloud):
        cloud_list = sorted(glob.glob(args.cloud + "/*.bin"))
        if len(cloud_list) == 0:
            raise FileNotFoundError(f"Cannot find clouds in {args.cloud}")
    elif osp.isfile(args.cloud):
        if args.cloud.endswith(".bin"):
            cloud_list = [args.cloud]
        else:
            raise TypeError("Only support .bin format")
    else:
        raise FileExistsError(f"Cloud {args.cloud_list} does not exists!")
    if not osp.isdir(args.export_dir):
        raise FileExistsError(f"Export directory {args.export_dir} does not exists!")
    if osp.exists(args.onnx):
        onnx_pth = args.onnx
    else:
        onnx_pth = osp.join(args.expport_dir, args.onnx)
        if not osp.exists(onnx_pth):
            raise FileExistsError(f"ONNX {onnx_pth} does not exist!")
    return cloud_list, onnx_pth


def infer_onnx_results(
    cloud_pth, onnx_session, onnx_results_pth, cfg, cloud_loader=None
):
    # Pre-processing
    preprocessor = Preprocess(points_path=cloud_pth, cfg=cfg, cloud_loader=cloud_loader)
    onnx_pre_data = preprocessor.preprocess()

    input_feed = {
        "inputs1": onnx_pre_data[
            "pfns_input"
        ].numpy(),  # Convert input_pfns to a NumPy array
        "coors": onnx_pre_data["coors"].numpy(),  # Convert coors to a NumPy array
    }

    # Run inference
    try:
        ort_output_part1 = onnx_session.run(None, input_feed)
    except Exception as e:
        print(f"Error during inference: {e}")

    ort_topk_inds, ort_cls_score, ort_bbox, ort_dir = ort_output_part1[0:4]

    postprogressor = Postprocess(cfg)
    ort_final_output = postprogressor.predict_by_feat(
        torch.Tensor(ort_cls_score),
        torch.Tensor(ort_bbox),
        torch.Tensor(ort_dir),
        ort_topk_inds,
    )

    # Dump the outputs
    onnx_results_dir = osp.dirname(onnx_results_pth)
    os.makedirs(onnx_results_dir, exist_ok=True)

    return ort_final_output


def parse_config():
    parser = argparse.ArgumentParser(description="Test the ONNX Model")
    parser.add_argument(
        "--cloud", type=str, help="Cloud path", default="demo/data/batch_0"
    )
    parser.add_argument("--input_pkl", type=str, default="")
    parser.add_argument(
        "--onnx", type=str, help="Single-stage ONNX path", default="single_stage.onnx"
    )
    parser.add_argument(
        "--export_dir", type=str, help="Export directory of onnx", default="export"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the main config file."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    Usage: python tools/onnx_utils/demo.py --cloud export
    """
    args = parse_config()
    cloud_list, onnx_pth = load_data(args)

    cfg = Config.fromfile(args.config)

    # Access the test pipeline from the dataset config file
    test_pipeline = cfg.get("test_pipeline", None)

    for step in test_pipeline:
        if step["type"] == "LoadPointsFromFile":
            load_dim = step.get("load_dim", None)
            use_dim = step.get("use_dim", None)

    cloud_loader = LoadPointsFromFile(
        coord_type="LIDAR", load_dim=load_dim, use_dim=use_dim
    )

    # Extract 'voxel_layer' into a new variable
    voxel_layer_params = cfg["model"]["data_preprocessor"]["voxel_layer"]

    cloud_params = PointCloudParameters(
        max_num_points=voxel_layer_params["max_num_points"],
        max_voxels=voxel_layer_params["max_voxels"][1],
        use_dim=use_dim,
        load_dim=load_dim,
        cloud_range=voxel_layer_params["point_cloud_range"],
        voxel_size=voxel_layer_params["voxel_size"],
    )

    result_list = []
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    ort_session_part1 = onnxruntime.InferenceSession(
        onnx_pth, sess_options=options, providers=["CPUExecutionProvider"]
    )
    for cloud_pth in tqdm.tqdm(cloud_list):
        frame_name = osp.basename(cloud_pth)

        # set onnx and align results path
        onnx_results_pth = osp.join(args.export_dir, frame_name + ".json")

        frame_results = infer_onnx_results(
            cloud_pth=cloud_pth,
            onnx_session=ort_session_part1,
            onnx_results_pth=onnx_results_pth,
            cfg=cfg,
            cloud_loader=cloud_loader,
        )

        result_list.append(frame_results)
    create_pickle_file(
        cloud_list=cloud_list, result_list=result_list, output_dir=args.export_dir
    )
