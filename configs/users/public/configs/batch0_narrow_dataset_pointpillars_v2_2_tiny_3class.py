_base_ = [
    '../models/pointpillars_hv_secfpn_v2_2_tiny.py',
    '../datasets/batch0-v2_2-3class.py',
    '../schedules/schedule_linear_5e-4.py',
    '../../../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(times=1, )
)

# train_dataloader = dict(
#     batch_size=2,
#     num_workers=2
# )

find_unused_parameters = True
