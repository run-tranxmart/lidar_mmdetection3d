import pytest
import torch
import json
import argparse
import os
import os.path as osp

from mmengine import load
from mmengine.structures import InstanceData
from mmdet3d.evaluation.metrics.custom_3d_metric_fp import Custom3DMetric_fp
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes

from pathlib import Path

def _init_evaluate_input(result_dir, sample_dict, start_idx, batch_size):
    result_list = sorted(os.listdir(result_dir))
    end_idx = min(start_idx + batch_size, len(result_list))  # Ensure we don't exceed the list length
    data_batch = {}
    predictions_list = []

    for i, result_fn in enumerate(result_list[start_idx:end_idx]):  # Process only the batch
        if result_fn.endswith('.json'):
            frame_name = Path(result_fn)
            json_pth = osp.join(result_dir, result_fn)
            frame_bin_name = frame_name.stem + '.bin'
            
            if frame_bin_name in sample_dict:
                sample_idx = sample_dict[frame_bin_name]
                predictions = Det3DDataSample()
                with open(json_pth, 'r') as f:
                    json_results = json.load(f)
                    metainfo = dict(
                        sample_idx=sample_idx, file_name=frame_bin_name)
                    predictions.set_metainfo(metainfo)
                    predictions.pred_instances = InstanceData()
                    pred_instances_3d = InstanceData()
                    # load bboxes
                    if 'bboxes_3d' in json_results:
                        pred_instances_3d.bboxes_3d = LiDARInstance3DBoxes(
                            torch.as_tensor(
                                json_results["bboxes_3d"],
                                dtype=torch.float32))
                    elif 'bboxes' in json_results:
                        pred_instances_3d.bboxes_3d = LiDARInstance3DBoxes(
                            torch.as_tensor(
                                json_results["bboxes"], dtype=torch.float32))
                    else:
                        raise KeyError("No bboxes or bboxes_3d in results")

                    # load scores
                    if 'scores_3d' in json_results:
                        pred_instances_3d.scores_3d = torch.as_tensor(
                            json_results["scores_3d"], dtype=torch.float32)
                    elif 'scores' in json_results:
                        pred_instances_3d.scores_3d = torch.as_tensor(
                            json_results["scores"], dtype=torch.float32)
                    else:
                        raise KeyError("No scores or scores_3d in results")

                    # load labels
                    if 'labels_3d' in json_results:
                        pred_instances_3d.labels_3d = torch.as_tensor(
                            json_results["labels_3d"], dtype=torch.int32)
                    elif 'labels' in json_results:
                        pred_instances_3d.labels_3d = torch.as_tensor(
                            json_results["labels"], dtype=torch.int32)
                    else:
                        raise KeyError("No labels or labels_3d in results")

                    predictions.pred_instances_3d = pred_instances_3d
                    predictions = predictions.to_dict()
                    predictions_list.append(predictions)
            else:
                raise KeyError(
                    f"Sample dict has not the frame {frame_bin_name}")
        else:
            print("Only support json format")
    return data_batch, predictions_list

def test_custom_3d_metric_mAP(args):
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    
    custommetric = Custom3DMetric_fp(
        args.val_pkl_path,
        metric=['bbox'],
        pcd_limit_range=[-100, -28.8, -4, 156, 28.8, 3],
        preload_annos=True,
        save_dir=args.save_dir,
        score_threshold=args.score_threshold)

    # Get the annotations and determine the sample index
    sample_dict = {}
    for sample_id, frame_infos in enumerate(custommetric.data_infos):
        frame_name = frame_infos['lidar_points']['lidar_path']
        clean_frame_name = os.path.basename(frame_name)
        sample_dict[clean_frame_name] = sample_id

    custommetric.dataset_meta = dict(classes=['Pedestrian', 'Cyclist', 'Car', 'Truck', 'Misc'])
    
    # New batch processing logic
    result_list = sorted(os.listdir(args.cpp_outputs_path))
    total_files = len([f for f in result_list if f.endswith('.json')])
    batch_size = 200  # Adjustable based on your GPU VRAM capacity
    ap_dict = {}
    
    print(f"Total files to process: {total_files}")
    for start_idx in range(0, total_files, batch_size):
        print(f"Processing batch: {start_idx} to {min(start_idx + batch_size, total_files)}")
        data_batch, predictions = _init_evaluate_input(
            args.cpp_outputs_path, sample_dict, start_idx, batch_size)
        
        # Clear previous results to free memory
        custommetric.results.clear()
        torch.cuda.empty_cache()  # Clear GPU memory between batches
        
        custommetric.process(data_batch, predictions)
        batch_ap_dict = custommetric.compute_metrics(custommetric.results)
        
        # Aggregate metrics (e.g., append or merge as needed)
        for key, value in batch_ap_dict.items():
            if key in ap_dict:
                # For simplicity, we'll just take the last batch's metrics
                # If you need to average or accumulate, modify this logic
                ap_dict[key] = value
            else:
                ap_dict[key] = value
    
    return ap_dict

def find_line_number(file_path, search_string):
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            if search_string in line:
                return line_number
    return -1

def parse_args():
    parser = argparse.ArgumentParser(
        description='evaluate cpp outputs for pointpillar')
    parser.add_argument('--val-pkl-path', help='the dir to get inputs pkl')
    parser.add_argument(
        '--cpp-outputs-path', help='the dir to get cpp outputs')
    parser.add_argument(
        '--save_dir',
        default='fp_miss_labeling_frames',
        help='directory to save false positive info'
    )
    parser.add_argument(
        '--score_threshold',
        type=float,
        default=0.8,
        help='score threshold for false positives'
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    test_custom_3d_metric_mAP(args)