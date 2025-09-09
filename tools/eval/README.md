# 3D Object Detection Evaluation Scripts

## 1. Overview

In this folder, we provide two scripts: epochs_eval.sh and evaluate_model_outputs.py. The former one is used to evaluate multiple models in the epoch directory. And the later one is used to evaluate the detection results. 

## 2. evaluate_model_outputs.py

This Python script is designed to evaluate the performance of 3D object detection models, specifically for calculating the Mean Average Precision (mAP) metric. It accepts ground truth data and prediction results as inputs, and computes the mAP within a user - specified evaluation range. The script supports filtering results by a specific frame and customizing the evaluation range.

### 2.1 Input arguments
The script accepts the following ar - line arguments:

| Argument | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `--gt` | str | Yes | None | Path to the pickle file containing ground truth data. |
| `--pred` | str | Yes | None | Directory path where the prediction JSON files are stored. |
| `--frame` | str | No | None | Name of the frame to filter the evaluation. If not provided, all frames will be considered. |
| `--xmax` | float | No | None | Maximum value of the x - axis for the evaluation range. |
| `--xmin` | float | No | None | Minimum value of the x - axis for the evaluation range. |
| `--ymax` | float | No | None | Maximum value of the y - axis for the evaluation range. |
| `--ymin` | float | No | None | Minimum value of the y - axis for the evaluation range. |
| `-x`, `--x_range` | float | No | 90.0 | Range of the x - axis for the evaluation. Used if `xmin` and `xmax` are not provided. |
| `-y`, `--y_range` | float | No | 28.8 | Range of the y - axis for the evaluation. Used if `ymin` and `ymax` are not provided. |

### 2.2 Usage

#### 2.2.1 Evaluate all frames

```
python tools/eval/evaluate_model_outputs.py \
    --gt path_to_pickle_file_of_label \
    --pred path_to_directory_of_predictions \
    --xmin xmin \
    --ymin ymin \ 
    --xmax xmax \
    --ymax ymax
```

For example, when evaluating the results for AICC project, the command can be:
```
python tools/eval/evaluate_model_outputs.py \
    --gt path_to_pickle_file_of_label \
    --pred path_to_directory_of_predictions \
    --xmin -100.0 \
    --xmax 156.0 \
    --y 28.8
```

#### 2.2.2 Evaluate single frame
```
python tools/eval/evaluate_model_outputs.py \
    --gt path_to_pickle_file_of_label \
    --pred path_to_directory_of_predictions \
    --frame frame_name
```

### 2.2.3 Outputs

An example of v3.4 ONNX model evaluation results is as follows:
```
----------- mAP Results ------------
Pedestrian AP@0.50:
bev  AP: 60.57
3d   AP: 50.06
aos  AP: 35.61
Cyclist AP@0.50:
bev  AP: 78.11
3d   AP: 71.74
aos  AP: 72.75
Car AP@0.70:
bev  AP: 91.27
3d   AP: 84.74
aos  AP: 90.24
Truck AP@0.70:
bev  AP: 81.83
3d   AP: 74.95
aos  AP: 81.01
Misc AP@0.50:
bev  AP: 18.25
3d   AP: 10.04
aos  AP: 9.12

Overall AP@all:
bev  AP: 66.01
3d   AP: 58.30
aos  AP: 57.75

Overall_without_Misc  AP@all:
bev  AP: 77.95
3d   AP: 70.37
aos  AP: 69.90
------------------------------------

```

### 2.3 Reference
For more details, please refer to this link: https://softwaremotion.feishu.cn/wiki/R6mTwhbXuiSBTbklETQcV7OQnif?fromScene=spaceOverview


## 3. epochs_eval.sh

### 3.1 Input arguments

The script requires the following command - line arguments:

| Argument       | Required | Description                                                                                                              |
|----------------|----------|--------------------------------------------------------------------------------------------------------------------------|
| `<config_file>` | Yes      | Path to the configuration file used for model evaluation.                                                                 |
| `<epoch_dir>`   | Yes      | Directory that contains model checkpoints. The checkpoint files should start with `epoch_` and end with `.pth`.          |
| `<gpu_num>`     | Yes      | The number of GPUs to be used for distributed testing.                                                                    |
| `<log_dir>`     | No       | Directory where the evaluation logs will be saved. If not provided, the default directory `work_dirs/<exper_name>/eval_epochs` will be used. Here, `<exper_name>` is the base name of the configuration file without the `.py` extension. |


### 3.2 Usage

```
./tools/epochs_eval.sh \
    your_config_path \
    your_epoch_directory \
    gpu_num
```

### 3.3 Results

After evaluation, the results of epoches will be saved in **your_epoch_directory/eval_epochs**. In addition, the script selects best epoch results. 