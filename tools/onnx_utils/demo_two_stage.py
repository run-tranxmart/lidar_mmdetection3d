# Copyright (c) Softmotion. All rights reserved.
import argparse
import glob
import json
import numpy as np
import onnxruntime
import os
import os.path as osp
import pickle as pkl
from preprocess import Preprocess
from postprocess import Postprocess
import warnings
import torch

from mmengine.config import Config
from mmengine.structures import InstanceData
from mmdet3d.models.middle_encoders import PointPillarsScatter
from mmdet3d.visualization import BEVLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes, Det3DDataSample
from mmdet3d.datasets.transforms.loading import LoadAnnotations3D, LoadPointsFromFile


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


def mtx_similar1(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """
    A method for calculating matrix similarity. Flatten the matrix into a vector and
    calculate the product of the vectors divided by the modulus length.
    Note the flattening operation.
    :param arr1: matrix 1
    :param arr2: matrix 2
    :return:cosine of the included angle
    """
    farr1 = arr1.ravel()
    farr2 = arr2.ravel()
    len1 = len(farr1)
    len2 = len(farr2)
    if len1 > len2:
        farr1 = farr1[:len2]
    else:
        farr2 = farr2[:len1]

    numer = np.sum(farr1 * farr2)
    denom = np.sqrt(np.sum(farr1**2) * np.sum(farr2**2))
    similar = numer / denom  # This is actually the cosine of the angle
    return similar


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
    # check the onnx models and pytorch results
    if not osp.isdir(args.export_dir):
        raise FileExistsError(f"Export directory {args.export_dir} does not exists!")
    onnx1_pth = osp.join(args.export_dir, args.onnx1)
    if not osp.exists(onnx1_pth):
        raise FileExistsError("ONNX part1 does not exist!")
    onnx2_pth = osp.join(args.export_dir, args.onnx2)
    if not osp.exists(onnx2_pth):
        raise FileExistsError(f"ONNX part2does not exist!")
    return cloud_list, onnx1_pth, onnx2_pth


def infer_onnx_results(
    cloud_pth,
    cloud_params,
    onnx1_pth,
    onnx2_pth,
    onnx_results_pth,
    alignment_results_pth,
    cfg,
    cloud_loader=None,
    step=1024,
    pytorch_results_pth=None,
    dump_results_cpp=True,
):

    # Pre-processing
    preprocessor = Preprocess(points_path=cloud_pth, cfg=cfg, cloud_loader=cloud_loader)

    onnx_pre_data = preprocessor.preprocess()
    # PFN
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    ort_session_part1 = onnxruntime.InferenceSession(
        onnx1_pth, sess_options=options, providers=["CPUExecutionProvider"]
    )
    onnx_pfn_outputs = ort_session_part1.run(
        None, {"inputs": onnx_pre_data["pfns_input"].numpy()}
    )[0]

    # Scatter
    scatter_layer = PointPillarsScatter(
        in_channels=64,
        output_shape=[cloud_params.feat_size[0], cloud_params.feat_size[1]],
    )
    onnx_scatter_output = scatter_layer.forward(
        torch.Tensor(onnx_pfn_outputs), torch.Tensor(onnx_pre_data["coors"])
    )

    # Detector
    pytorch_results = {}
    if pytorch_results_pth is not None and osp.exists(pytorch_results_pth):
        with open(pytorch_results_pth, "rb") as f:
            pytorch_results = pkl.load(f)

    ort_session2 = onnxruntime.InferenceSession(
        onnx2_pth, sess_options=options, providers=["CPUExecutionProvider"]
    )
    onnx_detector_input = onnx_scatter_output
    ort_detector_output = ort_session2.run(
        None, {"inputs": onnx_detector_input.numpy()}
    )
    ort_topk_inds, ort_cls_score, ort_bbox, ort_dir = ort_detector_output[0:4]

    # Post-Precessing
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
    with open(onnx_results_pth, "w") as ff:
        ff.write(json.dumps(ort_final_output))

    if dump_results_cpp:
        cpp_results_dir = osp.dirname(onnx_results_pth)
        os.makedirs(cpp_results_dir, exist_ok=True)
        # cpp_results_name, _, tmp = osp.basename(onnx_results_pth).split('.')
        # cpp_results_name, _, tmp = osp.basename(onnx_results_pth).split('.')
        parts = osp.basename(onnx_results_pth).split(".")
        if len(parts) == 4:
            cpp_results_name, _, _, _ = parts
        else:
            cpp_results_name = parts[0]  # Or handle this case based on your needs
        cpp_results_name_pre = cpp_results_dir + "/" + cpp_results_name + "_pre.bin"
        cpp_results_name_pfns = cpp_results_dir + "/" + cpp_results_name + "_pfns.bin"
        cpp_results_name_scatter = (
            cpp_results_dir + "/" + cpp_results_name + "_scatter.bin"
        )
        cpp_results_name_detector_ind = (
            cpp_results_dir + "/" + cpp_results_name + "_detector_ind.bin"
        )
        cpp_results_name_detector_score = (
            cpp_results_dir + "/" + cpp_results_name + "_detector_score.bin"
        )
        cpp_results_name_detector_bbox = (
            cpp_results_dir + "/" + cpp_results_name + "_detector_bbox.bin"
        )
        cpp_results_name_detector_dir = (
            cpp_results_dir + "/" + cpp_results_name + "_detector_dir.bin"
        )
        cpp_results_name_post_cls = (
            cpp_results_dir + "/" + cpp_results_name + "_post_cls.bin"
        )
        cpp_results_name_post_bbox = (
            cpp_results_dir + "/" + cpp_results_name + "_post_bbox.bin"
        )
        cpp_results_name_post_scores = (
            cpp_results_dir + "/" + cpp_results_name + "_post_score.bin"
        )

        np.array(onnx_pre_data["pfns_input"]).tofile(cpp_results_name_pre)
        np.array(onnx_pfn_outputs).tofile(cpp_results_name_pfns)
        np.array(onnx_scatter_output).tofile(cpp_results_name_scatter)
        # np.array(ort_topk_inds).tofile(cpp_results_name_detector_ind)
        np.array(ort_cls_score).tofile(cpp_results_name_detector_score)
        np.array(ort_bbox).tofile(cpp_results_name_detector_bbox)
        np.array(ort_dir).tofile(cpp_results_name_detector_dir)
        np.array(ort_final_output["labels_3d"]).tofile(cpp_results_name_post_cls)
        np.array(ort_final_output["scores_3d"]).tofile(cpp_results_name_post_scores)
        np.array(ort_final_output["bboxes_3d"]).tofile(cpp_results_name_post_bbox)

    print(f"Cloud frame: {cloud_pth} --> ONNX results: {onnx_results_pth}")

    # Measure the similarity between PyTorch and ONNX
    sim_pfn_input = 0.0
    sim_pfn_output = 0.0
    sim_scatter_output = 0.0
    sim_detector_output = 0.0
    if pytorch_results_pth is not None and osp.exists(pytorch_results_pth):
        pytorch_results = {}
        with open(pytorch_results_pth, "rb") as f:
            pytorch_results = pkl.load(f)
        # Pre-process similarity
        if "pfns_input" in pytorch_results:
            pytorch_pfn_input = pytorch_results["pfns_input"]
            sim_pfn_input = mtx_similar1(
                onnx_pre_data["pfns_input"].numpy(), pytorch_pfn_input.numpy()
            )
        else:
            warnings.warn("NOT find pre-processing data in pytorch results")

        # PFN similarity
        if "pfns_output" in pytorch_results:
            pytorch_pfn_output = pytorch_results["pfns_output"]
            sim_pfn_output = mtx_similar1(onnx_pfn_outputs, pytorch_pfn_output.numpy())
        else:
            warnings.warn("NOT find PFN data in pytorch results")

        # Detector similarity
        sim_detector_output = {}
        if "detector_output" in pytorch_results:
            pytorch_detector_output = pytorch_results["detector_output"]
            topk_num = ort_bbox.shape[0]
            if topk_num >= step:
                for select_num in range(step, topk_num, step):
                    sim_detector_output[select_num] = mtx_similar1(
                        ort_bbox[:select_num, :],
                        pytorch_detector_output[2].numpy()[:select_num, :],
                    )
            sim_detector_output[topk_num] = mtx_similar1(
                ort_bbox[:topk_num, :],
                pytorch_detector_output[2].numpy()[:topk_num, :],
            )
        else:
            warnings.warn("NOT find detector data in pytorch results")

        print(f"    Similarity of Pre-processing    : {sim_pfn_input}")
        print(f"    Similarity of PFN               : {sim_pfn_output}")
        print(f"    Similarity of Scatter           : {sim_scatter_output}")
        for select_num, select_sim in sim_detector_output.items():
            print(f"    Similarity of Detector Top-{select_num} : {select_sim}")
        with open(alignment_results_pth, "w") as f:
            f.write(f"Cloud frame: {cloud_pth}\n")
            f.write(f"  Similarity of Pre-processing    : {sim_pfn_input}\n")
            f.write(f"  Similarity of PFN               : {sim_pfn_output}\n")
            f.write(f"  Similarity of Scatter           : {sim_scatter_output}\n")
            for select_num, select_sim in sim_detector_output.items():
                f.write(f"    Similarity of Detector Top-{select_num} : {select_sim}\n")
    else:
        print("    PyTorch results is not provided")

    return ort_final_output


def visualization(
    cloud_list,
    result_list,
    cloud_loader,
    save_dir,
    show=False,
    score_thresh=None,
    class_names=None,
    voxel_size=None,
    vis_range=None,
):
    bev_visualizer = BEVLocalVisualizer(
        score_thresh=score_thresh,
        class_names=class_names,
        voxel_size=voxel_size,
        area_scope=vis_range,
    )

    os.makedirs(save_dir, exist_ok=True)
    for cloud_pth, cloud_result in zip(cloud_list, result_list):
        frame_points = cloud_loader.load_points(cloud_pth)
        frame_input = dict(points=frame_points)
        frame_name = cloud_pth.rsplit("/")[-1]
        save_pth = osp.join(save_dir, frame_name + ".png")
        frame_instances_3d = InstanceData()
        frame_instances_3d.bboxes_3d = LiDARInstance3DBoxes(
            torch.tensor(cloud_result["bboxes_3d"])
        )
        frame_instances_3d.scores_3d = torch.tensor(cloud_result["scores_3d"])
        frame_instances_3d.labels_3d = torch.tensor(cloud_result["labels_3d"])
        frame_dets = Det3DDataSample(pred_instances_3d=frame_instances_3d)
        bev_visualizer.add_datasample(
            name="onnx_infer",
            data_sample=frame_dets,
            data_input=frame_input,
            show=show,
            save_file=save_pth,
            vis_task="lidar_det",
            draw_gt=False,
            draw_pred=True,
            draw_distance=True,
            draw_det_scope=False,
        )


def parse_config():
    parser = argparse.ArgumentParser(description="Test the ONNX Model")
    parser.add_argument(
        "--cloud", type=str, help="Cloud path", default="demo/data/batch_0"
    )
    parser.add_argument("--input_pkl", type=str, default="")
    parser.add_argument(
        "--onnx1", type=str, help="ONNX Part1 model name of PFN", default="pfns.onnx"
    )
    parser.add_argument(
        "--onnx2",
        type=str,
        help="ONNX Part2 model name of detector",
        default="detector.onnx",
    )
    parser.add_argument(
        "--export_dir", type=str, help="Export directory of onnx", default="export"
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        default=False,
        help="Visualize the inference results",
    )
    parser.add_argument(
        "--show", action="store_true", default=False, help="Show the visual results"
    )
    parser.add_argument(
        "--pytorch_results",
        type=str,
        help="Inference results of PyTorch model",
        default=None,
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the main config file."
    )
    parser.add_argument(
        "--step",
        type=int,
        required=False,
        default=1024,
        help="Steps to compare similarity of detector",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    Usage: python tools/onnx_utils/demo.py --cloud export
    """
    args = parse_config()
    cloud_list, onnx1_pth, onnx2_pth = load_data(args)

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
    for i, cloud_pth in enumerate(cloud_list):
        frame_name = osp.basename(cloud_pth)
        # set pytorch results if found
        pytorch_results_pth = None
        if args.pytorch_results is not None:
            pytorch_results_pth = args.pytorch_results
        else:
            auto_pth = osp.join(args.export_dir, frame_name + ".pkl")
            if osp.exists(auto_pth):
                pytorch_results_pth = auto_pth
        # set onnx and align results path
        onnx_results_pth = osp.join(args.export_dir, frame_name + ".json")
        align_results_pth = osp.join(args.export_dir, frame_name + "_align.txt")
        frame_results = infer_onnx_results(
            cloud_pth,
            cloud_params,
            onnx1_pth,
            onnx2_pth,
            onnx_results_pth,
            align_results_pth,
            cfg,
            cloud_loader,
            args.step,
            pytorch_results_pth,
        )
        result_list.append(frame_results)

    # Visualization
    if args.visual:
        # Extract visualization parameters from cfg
        visualizer_cfg = cfg.get("visualizer", {})
        class_names = visualizer_cfg.get(
            "class_names", ["Pedestrian", "Cyclist", "Car", "Truck", "Misc"]
        )
        score_thresh = visualizer_cfg.get("score_thresh", [0.5] * len(class_names))
        vis_range = visualizer_cfg.get("area_scope", [[-72, 92], [-72, 72], [-5, 5]])
        voxel_size = voxel_layer_params["voxel_size"]
        visualization(
            cloud_list,
            result_list,
            cloud_loader,
            args.export_dir,
            args.show,
            score_thresh=score_thresh,
            class_names=class_names,
            voxel_size=voxel_size,
            vis_range=vis_range,
        )
