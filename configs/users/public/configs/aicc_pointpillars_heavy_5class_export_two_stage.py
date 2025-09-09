_base_ = [
    '../models/aicc_pointpillars_heavy_5class_two_stage_export.py',
    '../datasets/aicc_5class_export.py', 
    '../schedules/schedule_2x_lr_1e-4_final_val.py', 
    '../../../_base_/default_runtime.py'
]

load_from = "weights/aicc_models/batch2+3+4+8_finetune_epoch_24.pth"

find_unused_parameters = True
