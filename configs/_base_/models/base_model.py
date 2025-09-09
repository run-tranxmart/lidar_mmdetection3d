model = dict(
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=4096,
        nms_thr=0.5,
        score_thr=0.1,
        min_bbox_size=0,
        max_num=500
    )
)
