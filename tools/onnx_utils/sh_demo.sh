#!/bin/bash

# salloc -p xhhgexclu03 -N 1 -n 16 --gres=gpu:1
# salloc -p xhhgnormal -N 1 -n 16 --gres=gpu:1
module load anaconda3/5.2.0
source activate mmdet3d

python ./tools/onnx_utils/demo.py \
    --input_pcd "tools/onnx_utils/data/932125763.bin"\
    --input_pkl "work_dirs/export_dir_topk/light_2.0/1576111364819754.pkl"\
    --onnx1 "work_dirs/export_dir_topk/light_2.0/pfns.onnx"\
    --onnx2 "work_dirs/export_dir_topk/light_2.0/detector.onnx"\
    --save_name "work_dirs/export_dir_topk/light_2.0/newdata_932125763_finalout.json"
