# Copyright (c) Softmotion. All rights reserved.
import argparse
from copy import deepcopy
from gc import collect
import glob
import json
import mmengine
import numpy as np
import onnx
import onnxruntime
import os
import os.path as osp
import pdb
import pickle as pkl
import shutil
import torch
from tqdm import tqdm
from typing import Dict, List, Optional, Sequence, Union

from mmcv.transforms.base import BaseTransform
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope, TRANSFORMS
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from mmdet3d.datasets import KittiDataset
from mmdet3d.models.detectors.single_stage import SingleStage3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Box3DMode, Det3DDataSample, get_box_type
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from onnx_utils.postprocess import Postprocess

# from mmdet3d.apis import inference_detector, init_model
# from mmdet3d.datasets.transforms import LoadPointsFromFile, Pack3DDetInputs
# from mmdet3d.models.dense_heads import Base3DDenseHead
# from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
# from mmdet3d.models.voxel_encoders.pillar_encoder import PillarFeatureNet
# from mmdet3d.models.voxel_encoders.utils import PFNLayer


class VoxelNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(
        self,
        voxel_encoder: ConfigType,
        middle_encoder: ConfigType,
        backbone: ConfigType,
        neck: OptConfigType = None,
        bbox_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)
        self.pfn_layers = self.voxel_encoder.pfn_layers
        self.use_sigmoid_cls = True
        self.num_cls = bbox_head.num_classes
        self.topk_num = test_cfg.nms_pre

    def get_input_pfns(self, voxel_list):
        input_pfns_feature = self.voxel_encoder.extract_input_pfns(
            voxel_list[1], voxel_list[0], voxel_list[2]
        )
        return input_pfns_feature

    def get_output_pfns(self, voxel_list):
        output_pfns_feature = self.voxel_encoder(
            voxel_list[1], voxel_list[0], voxel_list[2]
        )
        return output_pfns_feature

    def get_scatter_feat(self, voxel_list):
        voxel_features = self.voxel_encoder(voxel_list[1], voxel_list[0], voxel_list[2])
        batch_size = voxel_list[2][-1, 0] + 1
        x = self.middle_encoder(voxel_features, voxel_list[2], batch_size)
        return x

    def get_bbox_head_feat(self, voxel_list):
        voxel_features = self.voxel_encoder(voxel_list[1], voxel_list[0], voxel_list[2])
        batch_size = voxel_list[2][-1, 0] + 1
        x = self.middle_encoder(voxel_features, voxel_list[2], batch_size)
        x = self.backbone(x)
        x = self.neck(x)
        cls, bbox, dir = self.bbox_head(x)
        return x, cls, bbox, dir

    def forward(self, x, coors):

        for layer in self.pfn_layers:
            x = layer(x)
        voxel_features = x.squeeze(1)
        batch_size = coors[-1, 0] + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(x)
        x = self.neck(x)
        cls, bbox, dir = self.bbox_head(x)

        cls = torch.permute(cls[0], [0, 2, 3, 1])
        cls = torch.reshape(cls, [-1, self.num_cls])
        if self.use_sigmoid_cls:
            cls = cls.sigmoid()
        else:
            cls = cls.softmax(-1)

        cls_score_max, _ = torch.max(cls, 1)
        _, cls_topk_ind = torch.topk(cls_score_max, self.topk_num)

        cls = cls[cls_topk_ind, :]

        bbox = torch.permute(bbox[0], [0, 2, 3, 1])
        bbox = torch.reshape(bbox, [-1, 7])
        bbox = bbox[cls_topk_ind, :]

        dir = torch.permute(dir[0], [0, 2, 3, 1])
        dir = torch.reshape(dir, [-1, 2])
        dir = dir[cls_topk_ind, :]

        return cls_topk_ind, cls, bbox, dir

    def forward_pfns(self, x):
        for layer in self.pfn_layers:
            x = layer(x)
        return x.squeeze(1)

    def forward_detector(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        cls, bbox, dir = self.bbox_head(x)
        cls = torch.permute(cls[0], [0, 2, 3, 1])
        cls = torch.reshape(cls, [-1, self.num_cls])
        cls_score_max, _ = torch.max(cls, 1)
        _, cls_topk_ind = torch.topk(cls_score_max, self.topk_num)
        cls = cls[cls_topk_ind, :]
        if self.use_sigmoid_cls:
            cls = cls.sigmoid()
        else:
            cls = cls.softmax(-1)
        bbox = torch.permute(bbox[0], [0, 2, 3, 1])
        bbox = torch.reshape(bbox, [-1, 7])
        bbox = bbox[cls_topk_ind, :]

        dir = torch.permute(dir[0], [0, 2, 3, 1])
        dir = torch.reshape(dir, [-1, 2])
        dir = dir[cls_topk_ind, :]
        return cls_topk_ind, cls, bbox, dir


def mtx_similar1(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """
    A method for calculating matrix similarity. Flatten the matrix into
    a vector and calculate the product of the vectors divided by the
    modulus length.
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


def get_models(config: ConfigType):
    data_preprocessor = config.model.data_preprocessor
    pts_voxel_encoder = config.model.voxel_encoder
    pts_middle_encoder = config.model.middle_encoder
    pts_backbone = config.model.backbone
    pts_neck = config.model.neck
    pts_bbox_head = config.model.bbox_head
    train_cfg = config.model.train_cfg
    test_cfg = config.model.test_cfg

    pointpillar = VoxelNet(
        voxel_encoder=pts_voxel_encoder,
        data_preprocessor=data_preprocessor,
        backbone=pts_backbone,
        middle_encoder=pts_middle_encoder,
        neck=pts_neck,
        bbox_head=pts_bbox_head,
        train_cfg=train_cfg,
        test_cfg=test_cfg,
    )
    return pointpillar


def infer_pytorch_model(
    model_config: ConfigType,
    checkpoint_pth: str,
    cloud_list: List,
    save_dir: str,
    device: str = "cpu",
):
    # create the save directory
    os.makedirs(save_dir, exist_ok=True)

    # initialize the model and load the weights from checkpoint
    model_config.model.train_cfg = None
    init_default_scope(model_config.get("default_scope", "mmdet3d"))
    initialized_model = MODELS.build(model_config.model).to(device)

    # initialize the pre-processor and test pipeline from dataset config
    data_preprocessor_obj = initialized_model.data_preprocessor
    test_pipeline = deepcopy(model_config.test_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(
        model_config.test_dataloader.dataset.box_type_3d
    )

    # Initiate the model
    model_pointpillar = get_models(model_config)
    load_checkpoint(model_pointpillar, checkpoint_pth, map_location=device)

    model_pointpillar.eval()
    model_pointpillar.to(device)

    for i, cloud_pth in enumerate(tqdm(cloud_list)):
        cloud_dict = dict(
            lidar_points=dict(lidar_path=cloud_pth),
            timestamp=1,
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d,
        )
        cloud_pipeline = test_pipeline(cloud_dict)
        data = [cloud_pipeline]
        collate_data = pseudo_collate(data)

        with torch.no_grad():
            model_inputs = data_preprocessor_obj(collate_data)
            del model_inputs["inputs"]["points"]
            del model_inputs["inputs"]["voxels"]["voxel_centers"]

            voxel_input = list(model_inputs["inputs"]["voxels"].values())
            output_detector = model_pointpillar(voxel_input)

            feat_data = {"detector_output": output_detector}
            frame_name = osp.basename(cloud_pth)
            save_torch_results(feat_data, frame_name, save_dir, False)


def export_onnx(
    model_config: ConfigType,
    checkpoint_pth: str,
    cloud_pth: str,
    save_dir: str,
    device: str = "cpu",
):
    # create the save directory
    os.makedirs(save_dir, exist_ok=True)

    # initialize the model and load the weights from checkpoint
    model_config.model.train_cfg = None
    init_default_scope(model_config.get("default_scope", "mmdet3d"))
    initialized_model = MODELS.build(model_config.model).to(device)

    # initialize the pre-processor and test pipeline from dataset config
    data_preprocessor_obj = initialized_model.data_preprocessor
    test_pipeline = deepcopy(model_config.test_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(
        model_config.test_dataloader.dataset.box_type_3d
    )

    cloud_dict = dict(
        lidar_points=dict(lidar_path=cloud_pth),
        timestamp=1,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
    )
    cloud_pipeline = test_pipeline(cloud_dict)
    data = [cloud_pipeline]
    collate_data = pseudo_collate(data)

    # Initiate the model
    model_pointpillar = get_models(model_config)
    load_checkpoint(model_pointpillar, checkpoint_pth, map_location=device)

    model_pointpillar.eval()
    model_pointpillar.to(device)

    with torch.no_grad():
        model_inputs = data_preprocessor_obj(collate_data)
        del model_inputs["inputs"]["points"]
        del model_inputs["inputs"]["voxels"]["voxel_centers"]

        # copy the inputs for the collection of outputs from different modules
        voxel_input = list(model_inputs["inputs"]["voxels"].values())

        voxel_input_copy0 = deepcopy(voxel_input)  # to collect pfn input
        voxel_input_copy1 = deepcopy(voxel_input)  # to collect pfn output
        voxel_input_copy2 = deepcopy(voxel_input)  # to collect scatter output
        voxel_input_copy3 = deepcopy(voxel_input)  # to collect detector output
        voxel_input_copy4 = deepcopy(voxel_input)  # copy for additional processing
        coors = deepcopy(voxel_input[2])

        coors = voxel_input_copy3[2]

        input_pfns = model_pointpillar.get_input_pfns(voxel_input_copy0)
        output_pfns = model_pointpillar.get_output_pfns(voxel_input_copy1)
        output_scatter = model_pointpillar.get_scatter_feat(voxel_input_copy2)
        output_detector = model_pointpillar(deepcopy(input_pfns), deepcopy(coors))

        feat_data = {
            "pfns_input": input_pfns,
            "pfns_output": output_pfns,
            "detector_input": output_scatter,
            "coors": coors,
            "detector_output": output_detector,
        }

        frame_name = osp.basename(cloud_pth)
        save_torch_results(feat_data, frame_name, save_dir, save_feat=True)

        # Export ONNX part1 model
        onnx1_path = osp.join(save_dir, "single_stage.onnx")

        pytorch_output = model_pointpillar(deepcopy(input_pfns), deepcopy(coors))

        torch.onnx.export(
            model_pointpillar,
            (deepcopy(input_pfns), deepcopy(coors)),
            onnx1_path,
            do_constant_folding=False,
            export_params=True,
            input_names=["inputs1", "coors"],
            output_names=["cls_topk_ind", "cls", "bbox", "dir"],
            verbose=False,
            opset_version=11,
            dynamic_axes={
                "inputs1": {0: "voxel_num"},  # dynamic batch size for input pfns
                "coors": {0: "voxel_num"},  # dynamic batch size for voxel list
            },
        )

        onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx1_path)), onnx1_path)

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        ort_session_part1 = onnxruntime.InferenceSession(
            onnx1_path, sess_options=options, providers=["CPUExecutionProvider"]
        )

        # Combine the input tensors into a single input dictionary
        input_feed = {
            "inputs1": input_pfns.numpy(),  # Convert input_pfns to a NumPy array
            "coors": voxel_input_copy4[2].numpy(),  # Convert coors to a NumPy array
        }

        # Run inference
        try:
            ort_output_part1 = ort_session_part1.run(None, input_feed)
        except Exception as e:
            print(f"Error during inference: {e}")

        cls_topk_ind_onnx = ort_output_part1[0]
        cls_onnx = ort_output_part1[1]
        bbox_onnx = ort_output_part1[2]
        dir_onnx = ort_output_part1[3]

        cls_topk_ind_pt = pytorch_output[0].numpy()
        cls_pt = pytorch_output[1].numpy()
        bbox_pt = pytorch_output[2].numpy()
        dir_pt = pytorch_output[3].numpy()

        # printing similarities before post processing
        similarity2 = mtx_similar1(cls_pt, cls_onnx)
        similarity3 = mtx_similar1(bbox_pt, bbox_onnx)
        similarity4 = mtx_similar1(dir_pt, dir_onnx)

        print("\nMatrix similarity between PyTorch and ONNX outputs 1: ", similarity2)
        print("\nMatrix similarity between PyTorch and ONNX outputs 2: ", similarity3)
        print("\nMatrix similarity between PyTorch and ONNX outputs 3: ", similarity4)

        cfg = Config.fromfile(args.model_cfg)

        postprogressor = Postprocess(cfg)
        ort_final_output = postprogressor.predict_by_feat(
            torch.Tensor(cls_onnx),
            torch.Tensor(bbox_onnx),
            torch.Tensor(dir_onnx),
            cls_topk_ind_onnx,
        )

        pytorch_final_output = postprogressor.predict_by_feat(
            torch.Tensor(cls_pt),
            torch.Tensor(bbox_pt),
            torch.Tensor(dir_pt),
            cls_topk_ind_pt,
        )

        bbox_postprocess_sim = mtx_similar1(
            np.array(ort_final_output["bboxes_3d"]),
            np.array(pytorch_final_output["bboxes_3d"]),
        )
        labels_postprocess_sim = mtx_similar1(
            np.array(ort_final_output["labels_3d"]),
            np.array(pytorch_final_output["labels_3d"]),
        )
        scores_postprocess_sim = mtx_similar1(
            np.array(ort_final_output["scores_3d"]),
            np.array(pytorch_final_output["scores_3d"]),
        )
        dir_postprocess_sim = mtx_similar1(
            np.array(ort_final_output["dir"]), np.array(pytorch_final_output["dir"])
        )

        print(
            "\nSimilarity of bounding boxes after post processing are: ",
            bbox_postprocess_sim,
        )
        print(
            "\nSimilarity of scores after post processing are: ", scores_postprocess_sim
        )
        print(
            "\nSimilarity of labels after post processing are: ", labels_postprocess_sim
        )
        print("\nSimilarity of dir after post processing are: ", dir_postprocess_sim)

    return onnx1_path


def save_torch_results(
    feat_data: Dict, frame_name: str, save_dir: str, save_feat: bool = False
):
    if save_feat:
        feat_pth = osp.join(save_dir, osp.basename(frame_name) + ".pkl")
        with open(feat_pth, "wb") as f:
            pkl.dump(feat_data, f)
    result_pth = osp.join(save_dir, osp.basename(frame_name) + ".json")
    if len(feat_data["detector_output"]) == 4:
        obj_top = feat_data["detector_output"][0]
        obj_score = feat_data["detector_output"][1]
        obj_bbox = feat_data["detector_output"][2]
        obj_dir = feat_data["detector_output"][3]
        cfg = Config.fromfile(args.model_cfg)
        postprogressor = Postprocess(cfg)
        obj_results = postprogressor.predict_by_feat(
            obj_score, obj_bbox, obj_dir, obj_top
        )
        mmengine.dump(obj_results, result_pth)


def diff_results(pipeline_pkl, infer_dir, diff_num_threshold=1):
    pipeline_results = {}
    with open(pipeline_pkl, "rb") as f:
        pipeline_dets = pkl.load(f)
        for i, frame_det in enumerate(pipeline_dets):
            frame_id = frame_det["lidar_path"].split(".")[0]
            pipeline_results[frame_id] = frame_det

    infer_list = sorted(os.listdir(infer_dir))
    align_num = 0
    disalign_num = 0
    total_num = len(infer_list)
    for i, frame_fn in enumerate(infer_list):
        frame_id = frame_fn.split(".")[0]
        json_pth = osp.join(infer_dir, frame_fn)
        with open(json_pth, "r") as f:
            infer_frame_dets = json.load(f)
            pipeline_frame_dets = pipeline_results[frame_id]
            infer_num = len(infer_frame_dets["scores_3d"])
            pipeline_num = pipeline_frame_dets["score"].shape[0]
            if np.abs(infer_num - pipeline_num) > diff_num_threshold:
                disalign_num += 1
                print(f"{frame_id} : pipeline {pipeline_num} != infer {infer_num}")
            else:
                align_num += 1

    print(
        f"Align num : {align_num} / {total_num}, \
            Disalign num: {disalign_num} / {total_num}"
    )


def get_frame_list(
    use_config_cloud: bool = False,
    cloud_pth: Optional[str] = None,
    model_config: ConfigType = None,
) -> None:
    # get the frame list
    frame_list = []
    if not use_config_cloud:
        if osp.isdir(cloud_pth):
            frame_list = glob.glob(cloud_pth + "/*.bin")
        elif osp.isfile(cloud_pth):
            frame_list = [cloud_pth]
        else:
            raise FileExistsError("Cloud does not exist: ", cloud_pth)
    else:
        print("Not given cloud, use the cloud from config instead")
        if "Identity" not in TRANSFORMS:

            @TRANSFORMS.register_module()
            class Identity(BaseTransform):

                def transform(self, info):
                    if "ann_info" in info:
                        info["gt_labels_3d"] = info["ann_info"]["gt_labels_3d"]
                    data_sample = Det3DDataSample()
                    gt_instances_3d = InstanceData()
                    gt_instances_3d.labels_3d = info["gt_labels_3d"]
                    data_sample.gt_instances_3d = gt_instances_3d
                    info["data_samples"] = data_sample
                    return info

        pipeline = [
            dict(type="Identity"),
        ]
        dataset_cfg = model_config.test_dataloader.dataset
        kitti_dataset = KittiDataset(
            data_root=dataset_cfg.data_root,
            ann_file=dataset_cfg.ann_file,
            modality=dataset_cfg.modality,
            test_mode=dataset_cfg.test_mode,
            convert_cam_to_lidar=False,
            data_prefix=dataset_cfg.data_prefix,
            pipeline=pipeline,
            pcd_limit_range=model_config.point_cloud_range,
            metainfo=dict(classes=model_config.class_names),
        )
        dataset_list = kitti_dataset.load_data_list()
        frame_list = [
            dataset_list[i]["lidar_points"]["lidar_path"]
            for i in range(len(dataset_list))
        ]
    return frame_list


def parse_config():
    parser = argparse.ArgumentParser(description="Convert PointPillar to ONNX")
    parser.add_argument(
        "--model_cfg",
        type=str,
        help="Path of config",
        default="configs/users/run.yang/configs/aicc_pointpillars_heavy_5class_finetune_single_stage_export.py",
    )
    parser.add_argument("--ckpt", type=str, help="Path of checkpoint", default="")
    parser.add_argument("--device", type=str, help="Used Device", default="cpu")
    parser.add_argument(
        "--lidar_point",
        type=str,
        help="Path of LiDAR point (only support bin format)",
        default="demo/data/aicc/cloud/2690_1714282795192.bin",
    )
    parser.add_argument(
        "--use_config_cloud",
        action="store_true",
        default=False,
        help="Use the clouds indicated by config instead of given lidar points",
    )
    parser.add_argument(
        "--infer_all", action="store_true", default=False, help="Inference all frames"
    )
    parser.add_argument(
        "--diff_pipeline",
        action="store_true",
        default=False,
        help="Compare results with test pipeline",
    )
    parser.add_argument(
        "--pipeline_results", type=str, help="Test pipeline results path", default=None
    )
    parser.add_argument(
        "--diff_num_threshold",
        type=int,
        help="Threshold of object number difference",
        default=1,
    )
    parser.add_argument("--export_dir", type=str, default="export/")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
        Example: 
            python tools/convert_onnx_single_stage.py --ckpt path_to_checkpoint \
                --model_cfg path_to_config \
                --dataset_cfg path_to_dataset_config \
                --lidar_point path_to_cloud \
    """
    args = parse_config()

    # check the model and configs
    if not osp.exists(args.ckpt):
        raise FileExistsError("Checkpoint does not exist: ", args.ckpt)
    if not osp.exists(args.model_cfg):
        raise FileExistsError("Model config does not exist: ", args.model_cfg)
    model_config = Config.fromfile(args.model_cfg)
    os.makedirs(args.export_dir, exist_ok=True)

    frame_list = get_frame_list(args.use_config_cloud, args.lidar_point, model_config)

    # Use the first frame to export onnx models
    frame_pth = frame_list[0]
    frame_name = osp.basename(frame_pth)
    shutil.copyfile(frame_pth, osp.join(args.export_dir, frame_name))
    onnx1_pth = export_onnx(
        model_config=model_config,
        checkpoint_pth=args.ckpt,
        cloud_pth=frame_pth,
        save_dir=args.export_dir,
        device=args.device,
    )
    print("Export ONNX model -> {}".format(onnx1_pth))

    # Inference all frames by PyTorch model
    if args.infer_all:
        infer_dir = osp.join(args.export_dir, "pytorch_infer")
        infer_pytorch_model(
            model_config=model_config,
            checkpoint_pth=args.ckpt,
            cloud_list=frame_list,
            save_dir=infer_dir,
            device=args.device,
        )

    # Compare the results between the above inference and test pipeline
    if args.diff_pipeline:
        infer_dir = osp.join(args.export_dir, "pytorch_infer")
        diff_results(args.pipeline_results, infer_dir, args.diff_num_threshold)
