_base_ = [
    '../models/pointpillars_hv_secfpn_pretrain_light.py',
    '../datasets/batch0-pretrain-3class.py',
    '../schedules/cyclic-80e.py',
    '../../../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=10,
    workers_per_gpu=10,
    train=dict(times=1, )
)

find_unused_parameters = True