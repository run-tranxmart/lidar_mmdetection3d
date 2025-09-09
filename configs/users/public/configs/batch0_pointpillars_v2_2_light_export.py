_base_ = [
    '../models/pointpillars_hv_secfpn_v2_2_light_onnx_export.py',
    '../datasets/batch0-v2_2-3class.py',
    '../schedules/schedule_linear_5e-4.py',
    '../../../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(times=1, )
)

find_unused_parameters = True
