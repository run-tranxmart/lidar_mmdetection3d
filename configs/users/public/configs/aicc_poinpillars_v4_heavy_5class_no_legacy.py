_base_ = [
    "../models/pointpillars_hv_secfpn_heavy_5class_no_legacy.py",
    "../datasets/aicc_5class.py",
    "../schedules/cyclic-80e.py",
    "../../../_base_/default_runtime.py",
]

# load_from = "weights/aicc_models/batch2+3+4+8_finetune_epoch_24.pth"

find_unused_parameters = True
