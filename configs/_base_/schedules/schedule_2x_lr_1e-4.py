# optimizer
# This schedule is mainly used by models on nuScenes dataset
lr = 0.0001
epoch_num = 24

# max_norm=10 is better for SECOND
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[20, 23]
)
momentum_config = None
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=epoch_num)

train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=4)
val_cfg = dict()
test_cfg = dict()