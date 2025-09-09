import numpy as np 
import matplotlib.pyplot as plt
import pytest
import torch
import json
import argparse
import os
import os.path as osp

import mmengine
from mmengine import load
from mmengine.structures import InstanceData
from mmdet3d.evaluation.metrics import Custom3DMetric
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes


from pathlib import Path
def _init_evaluate_input(result_dir, sample_dict):

    result_list = sorted(os.listdir(result_dir))
    data_batch = {}
    predictions_list = []

    for i, result_fn in enumerate(result_list):
        if result_fn.endswith('.json'):
            frame_name = Path(result_fn)
            

            json_pth = osp.join(result_dir, result_fn)
            frame_bin_name = frame_name.stem
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
    custommetric = Custom3DMetric(
        args.val_pkl_path,
        metric=['bbox'],
        pcd_limit_range=[-16, -16, -4, 80, 16, 3],
        preload_annos=True)

    # get the annotations and determine the sample index
    sample_dict = {}
    for sample_id, frame_infos in enumerate(custommetric.data_infos):
        frame_name = frame_infos['lidar_points']['lidar_path']
        sample_dict[frame_name] = sample_id
    
    custommetric.dataset_meta = dict(classes=['Pedestrian', 'Cyclist', 'Car', 'Truck'])
    data_batch, predictions = _init_evaluate_input(args.cpp_outputs_path,
                                                   sample_dict)
    custommetric.process(data_batch, predictions)
    ap_dict = custommetric.compute_metrics(custommetric.results)
  
    # plot pr_curve and get best threshold
    ret_dict_custom = ap_dict.get('ret_dict_custom', {})
    pr_results = ret_dict_custom.get('ret_bev', {}).get('ret_dict_unmax', None)

    if pr_results is None:
        raise ValueError("No valid PR results found in the provided dictionary.")

    # Recall values
    recall_values = pr_results['recall']

    # Number of classes, difficulties, and minoverlaps
    num_class = len(recall_values)
    num_difficulty = len(recall_values[0])
    num_minoverlap = len(recall_values[0][0])


    classes = ['Pedestrian', 'Cyclist', 'Car', 'Truck']
    difficultys = ['easy', 'normal', 'hard']

    max_pred = {}

    for class_idx in range(num_class):
        class_name = classes[class_idx]
        
        for diff_idx in range(num_difficulty):
            diff_name = difficultys[diff_idx]
            
            for overlap_idx in range(num_minoverlap):        
                precision = np.array(pr_results['precision'][class_idx][diff_idx][overlap_idx]).flatten()
                recall = np.array(pr_results['recall'][class_idx][diff_idx][overlap_idx]).flatten()
                thresholds = np.array(pr_results['threshold']).flatten()
                
                # Match length of precision, recall, and thresholds
                min_len = min(len(precision), len(recall), len(thresholds))
                precision = precision[:min_len]
                recall = recall[:min_len]
                thresholds = thresholds[:min_len]

          
                pr_product = precision * recall

                # Find the index of the maximum precision * recall
                max_pr_index = np.argmax(pr_product)
                max_pr_threshold = thresholds[max_pr_index]
                max_pr_value = pr_product[max_pr_index]

            
                max_pred[(class_name, diff_name)] = {
                    'max_pr': max_pr_value,
                    'precision': precision[max_pr_index],
                    'recall': recall[max_pr_index],
                    'threshold': max_pr_threshold
                }

            
                plt.figure(figsize=(14, 8))

                # Plot Precision and Recall against Thresholds
                plt.subplot(1, 2, 1)
                plt.plot( precision, thresholds, label='Precision', color='skyblue', alpha=0.7)
                plt.plot(recall, thresholds,label='Recall', color='orange', alpha=0.7)
                plt.axvline(x=max_pr_value, color='red', linestyle='--', label=f'MAX P*R = {max_pr_value:.8f}\nThreshold = {max_pr_threshold:.8f}', alpha=0.7)
                plt.scatter(max_pr_threshold, max_pr_value, color='red')
                plt.text(max_pr_threshold, max_pr_value, f'P = {precision[max_pr_index]:.8f}\nR = {recall[max_pr_index]:.8f}',
                verticalalignment='bottom', horizontalalignment='right')
                plt.xlabel('Score')
                plt.ylabel('Threshold')
                plt.title(f'Precision and Recall vs Thresholds for {class_name} - {diff_name}')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Plot Precision-Recall Curve
                plt.subplot(1, 2, 2)
                plt.plot(recall, precision, marker='o', color='blue', alpha=0.7)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve for {class_name} - {diff_name}')
                plt.grid(True, alpha=0.3)

        
                save_path = osp.join(args.save_dir, f'pr_curve_{class_name}_{diff_name}.png')
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()

    return max_pred

def find_line_number(file_path, search_string):
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            if search_string in line:
                return line_number
    return -1  # Return -1 if the string is not found

    
def parse_args():
    # metric, pcd_limit_range and classes are defined in the class
    parser = argparse.ArgumentParser(
        description='evaluate cpp outputs for pointpillar')
    parser.add_argument('--val-pkl-path', help='the dir to get inputs pkl')
    parser.add_argument(
        '--cpp-outputs-path', help='the dir to get cpp outputs')
    parser.add_argument(
        '--save_dir', help='the dir to save plots')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    max_pred = test_custom_3d_metric_mAP(args)
    print(max_pred)
    
if __name__ == "__main__":
   main()