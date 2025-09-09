import argparse
import json
import os
import os.path as osp
from pathlib import Path
import pytest
import torch

from mmengine import load
from mmengine.structures import InstanceData
from mmdet3d.evaluation.metrics import Custom3DMetric
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes


def determine_eval_range(args):
    """
    if the xmin and xmax are given, the eval range is determined by
    this two parameters. Otherwise, it is determined by x_range
    """
    if (args.xmin is not None) and (args.xmax is not None):
        if args.xmax < args.xmin:
            raise ValueError("xmax < xmin")
        xmin = args.xmin
        xmax = args.xmax
    else:
        xmax = args.x_range
        xmin = -1 * xmax

    if (args.ymin is not None) and (args.ymax is not None):
        if args.ymax < args.ymin:
            raise ValueError("ymax < ymin")
        ymin = args.ymin
        ymax = args.ymax
    else:
        ymax = args.y_range
        ymin = -1 * ymax
    eval_range = [xmin, ymin, -4, xmax, ymax, 3]
    return eval_range


def _init_evaluate_input(result_dir, sample_dict, select_frame=None):

    result_list = sorted(os.listdir(result_dir))
    data_batch = {}
    predictions_list = []

    for i, result_fn in enumerate(result_list):
        if result_fn.endswith(".json"):
            frame_name = Path(result_fn)
            json_pth = osp.join(result_dir, result_fn)
            frame_name = frame_name.stem
            if frame_name.endswith(".pcd") or frame_name.endswith(".bin"):
                frame_name = frame_name.rsplit(".", 1)[0]
            # if indicated frame, we only extract the gt and pred for such frame
            if select_frame is not None:
                if select_frame not in frame_name:
                    continue
            if frame_name in sample_dict:
                sample_idx = sample_dict[frame_name]
                predictions = Det3DDataSample()
                with open(json_pth, "r") as f:
                    json_results = json.load(f)
                    metainfo = dict(sample_idx=sample_idx, file_name=frame_name)
                    predictions.set_metainfo(metainfo)
                    predictions.pred_instances = InstanceData()
                    pred_instances_3d = InstanceData()
                    # load bboxes
                    if "bboxes_3d" in json_results:
                        pred_instances_3d.bboxes_3d = LiDARInstance3DBoxes(
                            torch.as_tensor(
                                json_results["bboxes_3d"], dtype=torch.float32
                            )
                        )
                    elif "bboxes" in json_results:
                        pred_instances_3d.bboxes_3d = LiDARInstance3DBoxes(
                            torch.as_tensor(json_results["bboxes"], dtype=torch.float32)
                        )
                    else:
                        raise KeyError("No bboxes or bboxes_3d in results")

                    # load scores
                    if "scores_3d" in json_results:
                        pred_instances_3d.scores_3d = torch.as_tensor(
                            json_results["scores_3d"], dtype=torch.float32
                        )
                    elif "scores" in json_results:
                        pred_instances_3d.scores_3d = torch.as_tensor(
                            json_results["scores"], dtype=torch.float32
                        )
                    else:
                        raise KeyError("No scores or scores_3d in results")

                    # load labels
                    if "labels_3d" in json_results:
                        pred_instances_3d.labels_3d = torch.as_tensor(
                            json_results["labels_3d"], dtype=torch.int32
                        )
                    elif "labels" in json_results:
                        pred_instances_3d.labels_3d = torch.as_tensor(
                            json_results["labels"], dtype=torch.int32
                        )
                    else:
                        raise KeyError("No labels or labels_3d in results")

                    predictions.pred_instances_3d = pred_instances_3d
                    predictions = predictions.to_dict()
                    predictions_list.append(predictions)
            else:
                print("Frame {} has no labels".format(frame_name))
                pass
                # raise KeyError(
                #     f"Sample dict has not the frame {frame_name}")
        else:
            print("Only support json format")
    return data_batch, predictions_list


def test_custom_3d_metric_mAP(args):
    if not torch.cuda.is_available():
        pytest.skip("Test requires GPU and torch+cuda")

    eval_range = determine_eval_range(args)
    print("Eval range: ", eval_range)

    custommetric = Custom3DMetric(
        args.gt,
        metric=["bbox"],
        pcd_limit_range=eval_range,
        preload_annos=True,
    )

    # get the annotations and determine the sample index
    sample_dict = {}
    for sample_id, frame_infos in enumerate(custommetric.data_infos):
        # Extracts the frame name as key
        cloud_path = frame_infos["lidar_points"]["lidar_path"]
        frame_name = osp.basename(cloud_path)
        if frame_name.endswith(".pcd") or frame_name.endswith(".bin"):
            frame_name = frame_name.rsplit(".", 1)[0]
        sample_dict[frame_name] = sample_id

    custommetric.dataset_meta = dict(
        classes=["Pedestrian", "Cyclist", "Car", "Truck", "Misc"]
    )
    data_batch, predictions = _init_evaluate_input(args.pred, sample_dict, args.frame)
    custommetric.process(data_batch, predictions)
    ap_dict = custommetric.compute_metrics(custommetric.results)
    return ap_dict


def find_line_number(file_path, search_string):
    with open(file_path, "r") as file:
        for line_number, line in enumerate(file, start=1):
            if search_string in line:
                return line_number
    return -1  # Return -1 if the string is not found


def parse_args():
    # metric, pcd_limit_range and classes are defined in the class
    parser = argparse.ArgumentParser(
        description="Evaluate json results for PointPillars"
    )
    parser.add_argument(
        "--gt",
        type=str,
        default=None,
        required=True,
        help="Pickle file path of ground-truth",
    )
    parser.add_argument(
        "--pred",
        type=str,
        default=None,
        required=True,
        help="Directory path of predictions",
    )
    parser.add_argument(
        "--frame",
        type=str,
        default=None,
        required=False,
        help="Frame name",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=None,
        required=False,
        help="Max x",
    )
    parser.add_argument(
        "--xmin",
        type=float,
        default=None,
        required=False,
        help="Min x",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        required=False,
        help="Max y",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        required=False,
        help="Min y",
    )
    parser.add_argument(
        "-x",
        "--x_range",
        type=float,
        default=90.0,
        required=False,
        help="Range y",
    )
    parser.add_argument(
        "-y",
        "--y_range",
        type=float,
        default=28.8,
        required=False,
        help="Range y",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    test_custom_3d_metric_mAP(args)
