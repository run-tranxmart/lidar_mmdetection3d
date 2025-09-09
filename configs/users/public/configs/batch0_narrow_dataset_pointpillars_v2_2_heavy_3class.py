_base_ = [
    '../models/pointpillars_hv_secfpn_v2_2_heavy.py',
    '../datasets/batch0-v2_2-3class.py',
    '../schedules/schedule_linear_4e-3.py',
    '../../../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(times=1, )
)

# train_dataloader = dict(
#     batch_size=1,
#     # num_workers=4
# )

find_unused_parameters = True
