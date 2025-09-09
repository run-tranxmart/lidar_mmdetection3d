# _base_ = ['../_base_/models/base_model.py']

# point_cloud = [-71.68, -71.68, -4, 92.16, 71.68, 3]
# voxel_size = [0.16, 0.16, point_cloud[5] - point_cloud[2]]

point_cloud = [-16, -16, -4, 80, 16, 3]
voxel_size = [0.2, 0.2, point_cloud[5] - point_cloud[2]]
dx = point_cloud[3] - point_cloud[0]
dy = point_cloud[4] - point_cloud[1]
feat_size = [round(dy / voxel_size[1]), round(dx / voxel_size[0]), 1]
# Maximum numbers of voxels used for training and testing
max_voxels = (32000, 12000)

# BN1D = dict(type='SyncBN', eps=1e-3, momentum=0.01)
# BN2D = dict(type='SyncBN', eps=1e-3, momentum=0.01)
BN1D = dict(type='BN1d', eps=1e-3, momentum=0.01)
BN2D = dict(type='BN2d', eps=1e-3, momentum=0.01)

model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,  # max_points_per_voxel
            point_cloud_range=point_cloud,
            voxel_size=voxel_size,
            max_voxels=max_voxels)),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        norm_cfg=BN1D,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud,
        use_conv1d=True),
    middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[feat_size[0], feat_size[1]]),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 3, 3],
        layer_strides=[2, 2, 2],
        out_channels=[64, 96, 128]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 96, 128],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        long_branch_num=1,
        norm_cfg=BN2D,
        conv_bias=False,
        anchor_generator=dict(
            type='Anchor3DSingleYawRangePerClassGenerator',
            ranges=[
                # Ped: 14=9+4+1
                # 9
                [
                    point_cloud[0] + 0. / 6, point_cloud[1] + 0. / 6, -0.6,
                    point_cloud[3] - voxel_size[0] + 0. / 6,
                    point_cloud[4] - voxel_size[1] + 0. / 6, -0.6
                ],
                [
                    point_cloud[0] + 1. / 6, point_cloud[1] + 0. / 6, -0.6,
                    point_cloud[3] - voxel_size[0] + 1. / 6,
                    point_cloud[4] - voxel_size[1] + 0. / 6, -0.6
                ],
                [
                    point_cloud[0] + 2. / 6, point_cloud[1] + 0. / 6, -0.6,
                    point_cloud[3] - voxel_size[0] + 2. / 6,
                    point_cloud[4] - voxel_size[1] + 0. / 6, -0.6
                ],
                [
                    point_cloud[0] + 0. / 6, point_cloud[1] + 1. / 6, -0.6,
                    point_cloud[3] - voxel_size[0] + 0. / 6,
                    point_cloud[4] - voxel_size[1] + 1. / 6, -0.6
                ],
                [
                    point_cloud[0] + 1. / 6, point_cloud[1] + 1. / 6, -0.6,
                    point_cloud[3] - voxel_size[0] + 1. / 6,
                    point_cloud[4] - voxel_size[1] + 1. / 6, -0.6
                ],
                [
                    point_cloud[0] + 2. / 6, point_cloud[1] + 1. / 6, -0.6,
                    point_cloud[3] - voxel_size[0] + 2. / 6,
                    point_cloud[4] - voxel_size[1] + 1. / 6, -0.6
                ],
                [
                    point_cloud[0] + 0. / 6, point_cloud[1] + 2. / 6, -0.6,
                    point_cloud[3] - voxel_size[0] + 0. / 6,
                    point_cloud[4] - voxel_size[1] + 2. / 6, -0.6
                ],
                [
                    point_cloud[0] + 1. / 6, point_cloud[1] + 2. / 6, -0.6,
                    point_cloud[3] - voxel_size[0] + 1. / 6,
                    point_cloud[4] - voxel_size[1] + 2. / 6, -0.6
                ],
                [
                    point_cloud[0] + 2. / 6, point_cloud[1] + 2. / 6, -0.6,
                    point_cloud[3] - voxel_size[0] + 2. / 6,
                    point_cloud[4] - voxel_size[1] + 2. / 6, -0.6
                ],
                # 4
                [
                    point_cloud[0], point_cloud[1], -0.6,
                    point_cloud[3] - voxel_size[0],
                    point_cloud[4] - voxel_size[1], -0.6
                ],
                [
                    point_cloud[0] + voxel_size[0], point_cloud[1], -0.6,
                    point_cloud[3], point_cloud[4] - voxel_size[1], -0.6
                ],
                [
                    point_cloud[0], point_cloud[1] + voxel_size[1], -0.6,
                    point_cloud[3] - voxel_size[0], point_cloud[4], -0.6
                ],
                [
                    point_cloud[0] + voxel_size[0],
                    point_cloud[1] + voxel_size[1], -0.6, point_cloud[3],
                    point_cloud[4], -0.6
                ],
                # 1
                [
                    point_cloud[0], point_cloud[1], -0.6,
                    point_cloud[3] - voxel_size[0],
                    point_cloud[4] - voxel_size[1], -0.6
                ],

                # Cyc: 16=8+8
                [
                    point_cloud[0], point_cloud[1], -0.6,
                    point_cloud[3] - voxel_size[0],
                    point_cloud[4] - voxel_size[1], -0.6
                ],
                [
                    point_cloud[0] + voxel_size[0], point_cloud[1], -0.6,
                    point_cloud[3], point_cloud[4] - voxel_size[1], -0.6
                ],
                [
                    point_cloud[0], point_cloud[1] + voxel_size[1], -0.6,
                    point_cloud[3] - voxel_size[0], point_cloud[4], -0.6
                ],
                [
                    point_cloud[0] + voxel_size[0],
                    point_cloud[1] + voxel_size[1], -0.6, point_cloud[3],
                    point_cloud[4], -0.6
                ],
                [
                    point_cloud[0], point_cloud[1], -0.6,
                    point_cloud[3] - voxel_size[0],
                    point_cloud[4] - voxel_size[1], -0.6
                ],
                [
                    point_cloud[0] + voxel_size[0], point_cloud[1], -0.6,
                    point_cloud[3], point_cloud[4] - voxel_size[1], -0.6
                ],
                [
                    point_cloud[0], point_cloud[1] + voxel_size[1], -0.6,
                    point_cloud[3] - voxel_size[0], point_cloud[4], -0.6
                ],
                [
                    point_cloud[0] + voxel_size[0],
                    point_cloud[1] + voxel_size[1], -0.6, point_cloud[3],
                    point_cloud[4], -0.6
                ],
                [
                    point_cloud[0], point_cloud[1], -0.6,
                    point_cloud[3] - voxel_size[0],
                    point_cloud[4] - voxel_size[1], -0.6
                ],
                [
                    point_cloud[0] + voxel_size[0], point_cloud[1], -0.6,
                    point_cloud[3], point_cloud[4] - voxel_size[1], -0.6
                ],
                [
                    point_cloud[0], point_cloud[1] + voxel_size[1], -0.6,
                    point_cloud[3] - voxel_size[0], point_cloud[4], -0.6
                ],
                [
                    point_cloud[0] + voxel_size[0],
                    point_cloud[1] + voxel_size[1], -0.6, point_cloud[3],
                    point_cloud[4], -0.6
                ],
                [
                    point_cloud[0], point_cloud[1], -0.6,
                    point_cloud[3] - voxel_size[0],
                    point_cloud[4] - voxel_size[1], -0.6
                ],
                [
                    point_cloud[0] + voxel_size[0], point_cloud[1], -0.6,
                    point_cloud[3], point_cloud[4] - voxel_size[1], -0.6
                ],
                [
                    point_cloud[0], point_cloud[1] + voxel_size[1], -0.6,
                    point_cloud[3] - voxel_size[0], point_cloud[4], -0.6
                ],
                [
                    point_cloud[0] + voxel_size[0],
                    point_cloud[1] + voxel_size[1], -0.6, point_cloud[3],
                    point_cloud[4], -0.6
                ],

                # Car: 6
                [
                    point_cloud[0], point_cloud[1], -1.78,
                    point_cloud[3] - voxel_size[0],
                    point_cloud[4] - voxel_size[1], -1.78
                ],
                [
                    point_cloud[0], point_cloud[1], -1.78,
                    point_cloud[3] - voxel_size[0],
                    point_cloud[4] - voxel_size[1], -1.78
                ],
                [
                    point_cloud[0], point_cloud[1], -1.78,
                    point_cloud[3] - voxel_size[0],
                    point_cloud[4] - voxel_size[1], -1.78
                ],
                [
                    point_cloud[0], point_cloud[1], -1.78,
                    point_cloud[3] - voxel_size[0],
                    point_cloud[4] - voxel_size[1], -1.78
                ],
                [
                    point_cloud[0], point_cloud[1], -1.78,
                    point_cloud[3] - voxel_size[0],
                    point_cloud[4] - voxel_size[1], -1.78
                ],
                [
                    point_cloud[0], point_cloud[1], -1.78,
                    point_cloud[3] - voxel_size[0],
                    point_cloud[4] - voxel_size[1], -1.78
                ],
            ],
            sizes=[
                # [length, width, height, rotation, cls]
                # Pedestrian: 13=9 + 4 + 1
                [0.4, 0.4, 1.58567389, 0, 0],
                [0.4, 0.4, 1.58567389, 0, 0],
                [0.4, 0.4, 1.58567389, 0, 0],
                [0.4, 0.4, 1.58567389, 0, 0],
                [0.4, 0.4, 1.58567389, 0, 0],
                [0.4, 0.4, 1.58567389, 0, 0],
                [0.4, 0.4, 1.58567389, 0, 0],
                [0.4, 0.4, 1.58567389, 0, 0],
                [0.4, 0.4, 1.58567389, 0, 0],
                [0.6, 0.6, 1.58567389, 0, 0],
                [0.6, 0.6, 1.58567389, 0, 0],
                [0.6, 0.6, 1.58567389, 0, 0],
                [0.6, 0.6, 1.58567389, 0, 0],
                [0.8, 0.8, 1.58567389, 0, 0],

                # Cyclist: 8 + 8
                [1.62955035, 0.54944008, 1.19224497, 0, 1],
                [1.62955035, 0.54944008, 1.19224497, 0, 1],
                [1.62955035, 0.54944008, 1.19224497, 0, 1],
                [1.62955035, 0.54944008, 1.19224497, 0, 1],
                [1.97689939, 0.87673796, 1.57529756, 0, 1],
                [1.97689939, 0.87673796, 1.57529756, 0, 1],
                [1.97689939, 0.87673796, 1.57529756, 0, 1],
                [1.97689939, 0.87673796, 1.57529756, 0, 1],
                [1.62955035, 0.54944008, 1.19224497, 1.57, 1],
                [1.62955035, 0.54944008, 1.19224497, 1.57, 1],
                [1.62955035, 0.54944008, 1.19224497, 1.57, 1],
                [1.62955035, 0.54944008, 1.19224497, 1.57, 1],
                [1.97689939, 0.87673796, 1.57529756, 1.57, 1],
                [1.97689939, 0.87673796, 1.57529756, 1.57, 1],
                [1.97689939, 0.87673796, 1.57529756, 1.57, 1],
                [1.97689939, 0.87673796, 1.57529756, 1.57, 1],

                # Car: 6
                [4.36032103, 1.85728288, 1.62119677, 0, 2],
                [4.36032103, 1.85728288, 1.62119677, 1.57, 2],
                [6.98833714, 2.55313427, 2.92700137, 0, 2],
                [6.98833714, 2.55313427, 2.92700137, 1.57, 2],
                [12.78522061, 3.02172211, 3.85993345, 0, 2],
                [12.78522061, 3.02172211, 3.85993345, 1.57, 2],
            ],
            reshape_out=False),
        assigner_per_size=False,
        assign_per_class=True,
        diff_rad_by_sin=True,
        dir_offset=0.7854,
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.5,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=0.01, loss_weight=1.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(
                type='ATSS3DAssigner',  # ped
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                topk=9 * 14,
                ignore_iof_thr=-1),
            dict(
                type='ATSS3DAssigner',  # cyclist
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                topk=9 * 16,
                ignore_iof_thr=-1),
            dict(
                type='ATSS3DAssigner',  # car
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                topk=9 * 6,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        # debug=False
    ),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=4096,
        nms_thr=0.5,
        score_thr=0.1,
        min_bbox_size=0,
        max_num=500),
)
