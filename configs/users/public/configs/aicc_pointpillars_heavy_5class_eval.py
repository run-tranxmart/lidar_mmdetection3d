_base_ = [
    "../models/aicc_pointpillars_heavy_5class_eval_28.8m.py",
    "../datasets/aicc_5class_eval.py",
    "../schedules/schedule_2x_lr_1e-4_final_val.py",
    "../../../_base_/default_runtime.py",
]

find_unused_parameters = True
