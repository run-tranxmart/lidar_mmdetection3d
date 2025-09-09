_base_ = [
    '../models/aicc_pointpillars_heavy_3class_export.py',
    "../datasets/aicc_3class_export.py",
    '../schedules/schedule_2x_lr_1e-4_final_val.py', 
    '../../../_base_/default_runtime.py'
]

load_from = "./model_zoo/pointpillars/pointpillars_heavy_version2_conv1d_epoch_80.pth"

find_unused_parameters = True
