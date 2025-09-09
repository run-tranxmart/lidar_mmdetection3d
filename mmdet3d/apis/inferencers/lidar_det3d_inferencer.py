# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List, Optional, Sequence, Union, Tuple

import mmengine
import numpy as np
from pypcd import pypcd
from rich.progress import track
import torch
from tqdm import tqdm

from mmengine import dump, print_log
from mmengine.dataset import Compose
from mmengine.fileio import get_file_backend, isdir, join_path, list_dir_or_file
from mmengine.infer.infer import ModelType
from mmengine.structures import InstanceData

from mmdet3d.registry import INFERENCERS
from mmdet3d.structures import (
    CameraInstance3DBoxes,
    DepthInstance3DBoxes,
    Det3DDataSample,
    LiDARInstance3DBoxes,
)
from mmdet3d.utils import ConfigType
from mmdet3d.visualization import Det3DLocalVisualizer, BEVLocalVisualizer

from .base_3d_inferencer import Base3DInferencer

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


@INFERENCERS.register_module(name="det3d-lidar")
@INFERENCERS.register_module()
class LidarDet3DInferencer(Base3DInferencer):
    """The inferencer of LiDAR-based detection.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "pointpillars_kitti-3class" or
            "configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py". # noqa: E501
            If model is not specified, user must provide the
            `weights` saved by MMEngine which contains the config string.
            Defaults to None.
        weights (str, optional): Path to the checkpoint. If it is not specified
            and model is a model name of metafile, the weights will be loaded
            from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        scope (str): The scope of the model. Defaults to 'mmdet3d'.
        palette (str): Color palette used for visualization. The order of
            priority is palette -> config -> checkpoint. Defaults to 'none'.
    """

    def __init__(
        self,
        model: Union[ModelType, str, None] = None,
        weights: Optional[str] = None,
        device: Optional[str] = None,
        scope: str = "mmdet3d",
        palette: str = "none",
    ) -> None:
        # A global counter tracking the number of frames processed, for
        # naming of the output results
        self.num_visualized_frames = 0
        super(LidarDet3DInferencer, self).__init__(
            model=model, weights=weights, device=device, scope=scope, palette=palette
        )

    def _inputs_to_list(self, inputs: Union[dict, list], **kwargs) -> list:
        """Preprocess the inputs to a list.

        Preprocess inputs to a list according to its type:

        - list or tuple: return inputs
        - dict: the value with key 'points' is
            - Directory path: return all files in the directory
            - other cases: return a list containing the string. The string
              could be a path to file, a url or other types of string according
              to the task.

        Args:
            inputs (Union[dict, list]): Inputs for the inferencer.

        Returns:
            list: List of input for the :meth:`preprocess`.
        """
        if isinstance(inputs, dict) and isinstance(inputs["points"], str):
            pcd = inputs["points"]
            backend = get_file_backend(pcd)
            if hasattr(backend, "isdir") and isdir(pcd):
                # Backends like HttpsBackend do not implement `isdir`, so
                # only those backends that implement `isdir` could accept
                # the inputs as a directory
                filename_list = list_dir_or_file(pcd, list_dir=False)
                inputs = [
                    {"points": join_path(pcd, filename)} for filename in filename_list
                ]

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return list(inputs)

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        load_point_idx = self._get_transform_idx(pipeline_cfg, "LoadPointsFromFile")
        if load_point_idx == -1:
            raise ValueError("LoadPointsFromFile is not found in the test pipeline")

        load_cfg = pipeline_cfg[load_point_idx]
        self.coord_type, self.load_dim = load_cfg["coord_type"], load_cfg["load_dim"]
        self.use_dim = (
            list(range(load_cfg["use_dim"]))
            if isinstance(load_cfg["use_dim"], int)
            else load_cfg["use_dim"]
        )

        pipeline_cfg[load_point_idx]["type"] = "LidarDet3DInferencerLoader"
        return Compose(pipeline_cfg)

    def visualize(
        self,
        inputs: InputsType,
        preds: PredType,
        return_vis: bool = False,
        show: bool = False,
        wait_time: int = -1,
        draw_pred: bool = True,
        pred_score_thr: float = 0.3,
        no_save_vis: bool = False,
        img_out_dir: str = "",
    ) -> Union[List[np.ndarray], None]:
        """Visualize predictions.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            preds (PredType): Predictions of the model.
            return_vis (bool): Whether to return the visualization result.
                Defaults to False.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to -1.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            no_save_vis (bool): Whether to force not to save prediction
                vis results. Defaults to False.
            img_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.

        Returns:
            List[np.ndarray] or None: Returns visualization results only if
            applicable.
        """
        if no_save_vis is True:
            img_out_dir = ""

        if not show and img_out_dir == "" and not return_vis:
            return None

        if getattr(self, "visualizer") is None:
            raise ValueError(
                'Visualization needs the "visualizer" term'
                "defined in the config, but got None."
            )

        results = []
        if len(inputs) != len(preds):
            return results
        for single_input, pred in zip(inputs, preds):
            single_input = single_input["points"]
            if isinstance(single_input, str):
                
                if single_input.endswith('.pcd'):
                    pcd_cloud = pypcd.point_cloud_from_path(single_input)
                    points = np.concatenate(
                        (
                            pcd_cloud.pc_data['x'][:, np.newaxis],
                            pcd_cloud.pc_data['y'][:, np.newaxis],
                            pcd_cloud.pc_data['z'][:, np.newaxis],
                            pcd_cloud.pc_data['intensity'][:, np.newaxis]),
                        axis=1)
                elif single_input.endswith('.npy'):
                    points = np.load(single_input)
                else:
                    # Default format is binary file
                    pts_bytes = mmengine.fileio.get(single_input)
                    points = np.frombuffer(pts_bytes, dtype=np.float32)
                points = points.reshape(-1, self.load_dim)
                points = points[:, self.use_dim]
                pc_name = osp.basename(single_input).rsplit(".", 1)[0]
                pc_name = f"{pc_name}.png"
            elif isinstance(single_input, np.ndarray):
                points = single_input.copy()
                pc_num = str(self.num_visualized_frames).zfill(8)
                pc_name = f"{pc_num}.png"
            else:
                raise ValueError("Unsupported input type: " f"{type(single_input)}")

            data_input = dict(points=points)
            if isinstance(self.visualizer, Det3DLocalVisualizer):
                if img_out_dir != "" and show:
                    save_path = osp.join(img_out_dir, "vis_lidar", pc_name)
                    mmengine.mkdir_or_exist(osp.dirname(save_path))
                else:
                    save_path = None

                self.visualizer.add_datasample(
                    pc_name,
                    data_input,
                    pred,
                    show=show,
                    wait_time=wait_time,
                    draw_gt=False,
                    draw_pred=draw_pred,
                    pred_score_thr=pred_score_thr,
                    o3d_save_path=save_path,
                    vis_task="lidar_det",
                )
            elif isinstance(self.visualizer, BEVLocalVisualizer):
                if not no_save_vis:
                    # for BEVLocalVisualizer, the pred_score_thr is indicated
                    #   in the config file so that we do not pass this
                    #   parameter to the add_datasample function
                    save_path = osp.join(img_out_dir, "vis_lidar", pc_name)
                    self.visualizer.add_datasample(
                        name=pc_name,
                        data_input=data_input,
                        data_sample=pred,
                        show=show,
                        wait_time=wait_time,
                        draw_gt=False,
                        draw_pred=True,
                        pred_score_thr=None,
                        save_file=save_path,
                        vis_task="lidar_det",
                    )
            else:
                raise TypeError(f"Unsupported visualization type: {self.visualizer}")
            results.append(points)
            self.num_visualized_frames += 1

        return results

    def visualize_preds_fromfile(
        self, inputs: InputsType, preds: PredType, **kwargs
    ) -> Union[List[np.ndarray], None]:
        """Visualize predictions from `*.json` files.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            preds (PredType): Predictions of the model.

        Returns:
            List[np.ndarray] or None: Returns visualization results only if
            applicable.
        """
        data_samples = []
        for pred in preds:
            pred = mmengine.load(pred)
            data_sample = Det3DDataSample()
            data_sample.pred_instances_3d = InstanceData()

            data_sample.pred_instances_3d.labels_3d = torch.tensor(pred["labels_3d"])
            data_sample.pred_instances_3d.scores_3d = torch.tensor(pred["scores_3d"])
            if pred["box_type_3d"] == "LiDAR":
                data_sample.pred_instances_3d.bboxes_3d = LiDARInstance3DBoxes(
                    pred["bboxes_3d"]
                )
            elif pred["box_type_3d"] == "Camera":
                data_sample.pred_instances_3d.bboxes_3d = CameraInstance3DBoxes(
                    pred["bboxes_3d"]
                )
            elif pred["box_type_3d"] == "Depth":
                data_sample.pred_instances_3d.bboxes_3d = DepthInstance3DBoxes(
                    pred["bboxes_3d"]
                )
            else:
                raise ValueError("Unsupported box type: " f'{pred["box_type_3d"]}')
            data_samples.append(data_sample)
        return self.visualize(inputs=inputs, preds=data_samples, **kwargs)

    def __call__(
        self,
        inputs: InputsType,
        batch_size: int = 1,
        return_datasamples: bool = False,
        **kwargs,
    ) -> Optional[dict]:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            batch_size (int): Batch size. Defaults to 1.
            return_datasamples (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
            **kwargs: Key words arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            dict: Inference and visualization results.
        """

        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        cam_type = preprocess_kwargs.pop("cam_type", "CAM2")
        ori_inputs = self._inputs_to_list(inputs, cam_type=cam_type)
        inputs = self.preprocess(ori_inputs, batch_size=batch_size, **preprocess_kwargs)

        results_dict = {"lidar_paths": [], "predictions": [], "save_paths": []}
        # inference
        for data in (
            track(inputs, description="LiDAR Model Inference")
            if self.show_progress
            else inputs
        ):
            pred = self.forward(data, **forward_kwargs)
            results = self.postprocess(pred, return_datasamples, **postprocess_kwargs)
            for frame in data["data_samples"]:
                results_dict["lidar_paths"].extend(frame.lidar_path)
            for k, v in results.items():
                if k in results_dict:
                    results_dict[k].extend(v)
        if not visualize_kwargs["no_save_vis"]:
            for data, pred in tqdm(
                zip(ori_inputs, results_dict["save_paths"]),
                total=len(ori_inputs),
                desc="Visualize results",
            ):
                self.visualize_preds_fromfile([data], [pred], **visualize_kwargs)
        return results_dict

    def postprocess(
        self,
        preds: PredType,
        return_datasample: bool = False,
        print_result: bool = False,
        no_save_pred: bool = False,
        pred_out_dir: str = "",
        reserve_results: bool = False,
    ) -> Union[ResType, Tuple[ResType, np.ndarray]]:
        """Process the predictions and visualization results from ``forward``
        and ``visualize``.

        This method should be responsible for the following tasks:

        1. Convert datasamples into a json-serializable dict if needed.
        2. Pack the predictions and visualization results and return them.
        3. Dump or log the predictions.

        Args:
            pred (Dict): Predictions of the model.
            return_datasample (bool): Whether to use Datasample to store
                inference results. If False, dict will be used.
                Defaults to False.
            print_result (bool): Whether to print the inference result w/o
                visualization to the console. Defaults to False.
            pred_out_dir (str): Directory to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Inference and visualization results with key ``predictions``
            and ``visualization``.

            - ``visualization`` (Any): Returned by :meth:`visualize`.
            - ``predictions`` (dict or DataSample): Returned by
            :meth:`forward` and processed in :meth:`postprocess`.
            If ``return_datasample=False``, it usually should be a
            json-serializable dict containing only basic data elements such
            as strings and numbers.
        """
        if no_save_pred is True:
            pred_out_dir = ""

        result_dict = {
            "predictions": [],
            "save_paths": [],
        }
        # Just save the results instead of appending
        for pred in preds:
            result, json_path = self.pred2dict(pred, pred_out_dir)
            result_dict["save_paths"].append(json_path)
        # result_dict["predictions"] = results
        if print_result:
            print(result)
        return result_dict

    def pred2dict(self, data_sample: Det3DDataSample, pred_out_dir: str = "") -> Dict:
        """Extract elements necessary to represent a prediction into a
        dictionary.

        It's better to contain only basic data elements such as strings and
        numbers in order to guarantee it's json-serializable.

        Args:
            data_sample (:obj:`DetDataSample`): Predictions of the model.
            pred_out_dir: Dir to save the inference results w/o
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

        Returns:
            dict: Prediction results.
        """
        result = {}
        if "pred_instances_3d" in data_sample:
            pred_instances_3d = data_sample.pred_instances_3d.numpy()
            result = {
                "labels_3d": pred_instances_3d.labels_3d.tolist(),
                "scores_3d": pred_instances_3d.scores_3d.tolist(),
                "bboxes_3d": pred_instances_3d.bboxes_3d.tensor.cpu().tolist(),
            }

        if "pred_pts_seg" in data_sample:
            pred_pts_seg = data_sample.pred_pts_seg.numpy()
            result["pts_semantic_mask"] = pred_pts_seg.pts_semantic_mask.tolist()

        result["box_type_3d"] = "LiDAR"
        if pred_out_dir != "":
            if "lidar_path" in data_sample:
                lidar_path = osp.basename(data_sample.lidar_path)
                lidar_path = osp.splitext(lidar_path)[0]
                out_json_path = osp.join(pred_out_dir, "preds", lidar_path + ".json")
            elif "img_path" in data_sample:
                img_path = osp.basename(data_sample.img_path)
                img_path = osp.splitext(img_path)[0]
                out_json_path = osp.join(pred_out_dir, "preds", img_path + ".json")
            else:
                out_json_path = osp.join(
                    pred_out_dir,
                    "preds",
                    f"{str(self.num_visualized_imgs).zfill(8)}.json",
                )
            dump(result, out_json_path)
        return result, out_json_path
