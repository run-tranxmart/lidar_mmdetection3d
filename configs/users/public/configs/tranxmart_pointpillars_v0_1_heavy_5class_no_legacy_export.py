_base_ = [
    "../models/pointpillars_hv_secfpn_heavy_5class_export_no_legacy.py",
    "../datasets/tranxmart_5class_export.py",
    "../schedules/schedule_2x_lr_1e-4_final_val.py",
    "../../../_base_/default_runtime.py",
]

find_unused_parameters = True
