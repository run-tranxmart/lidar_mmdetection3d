_base_ = [
    "../models/aicc_pointpillars_heavy_5class_eval_40m.py",
    "../datasets/aicc_5class_eval_40m.py",
    "../schedules/schedule_2x_lr_1e-4_final_val.py",
    "../../../_base_/default_runtime.py",
]

find_unused_parameters = True
