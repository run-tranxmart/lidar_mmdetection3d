# custom_imports = dict(
#     imports=['mmdet3d.datasets.Kiitti4class'],
#     allow_failed_imports=False)

# dataset settings
point_cloud_range = [-100, -40, -4, 156, 40, 3]
voxel_size_range = [0.2, 0.2, 7]

dataset_type = 'Kitti_5Class_Dataset'
data_root = "data/"

class_names = ['Pedestrian', 'Cyclist', 'Car', 'Truck', 'Misc']

input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,  # x, y, z, intensity
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.087266, 0.087266],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=[
        'batch_2_aicc_data/pkls/5class/val_data_infos.update_prefix.x_offset.pkl',
    ],
    data_prefix=dict(pts=''),
    pipeline=train_pipeline,
    modality=input_modality,
    test_mode=False,
    metainfo=metainfo,
    # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
    # and box_type_3d='Depth' in sunrgbd and scannet dataset.
    box_type_3d='LiDAR',
    convert_cam_to_lidar=False,
    backend_args=backend_args
) 

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=train_dataset
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts=''),
        ann_file=[
            'batch_2_aicc_data/pkls/5class/val_data_infos.update_prefix.x_offset.pkl',
        ],
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args
    )
)
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts=''),
        ann_file=[
            'batch_2_aicc_data/pkls/5class/val_data_infos.update_prefix.x_offset.pkl',
        ],
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args
    )
)

val_evaluator = dict(
    type='Custom3DMetric',
    ann_file=data_root + 'batch_2_aicc_data/pkls/5class/val_data_infos.update_prefix.x_offset.pkl',
    metric='bbox',
    pcd_limit_range=point_cloud_range,
    backend_args=backend_args)
test_evaluator = val_evaluator


vis_range = [[-72, 156], [-72, 72], [-5, 5]]
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="BEVLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
    det_scope=point_cloud_range,
    area_scope=vis_range,
    class_names=class_names,
    score_thresh=[0.4, 0.4, 0.4, 0.4, 0.4],
)
