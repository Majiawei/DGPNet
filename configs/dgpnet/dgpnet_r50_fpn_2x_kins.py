_base_ = [
    '../_base_/datasets/kins_depth_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
class_names = ['cyclist', 'pedestrian', 'car', 'van', 'misc']
# model settings
model = dict(
    type='MultiTaskRefineATSS',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        conv1_out=True,
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    depth_decoder=dict(
        type='DepthDecoder',
        feat_out_channels=[64, 256, 512, 1024, 2048],  # [64, 256, 512, 1024, 2048]
        num_features=256,
        max_depth=80.0,
        depth_channels=256,
        base_channels=64),
    bbox_head=dict(
        type='RefineATSSIOUHead',
        num_classes=len(class_names),
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        ##$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        fuse_method='mul', # cat,pecat
        #########################  stage 1
        loc_filter_thr=0.005,
        stage1_anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[8],
            strides=[8, 16, 32, 64, 128]),
        stage1_bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        stage1_loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True,
            loss_weight=1.0),
        stage1_loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        # stage1_loss_centerness=dict(
        #     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # stage1_loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        #########################  stage 2
        # anchor_generator=dict(
        #     type='AnchorGenerator',
        #     ratios=[1.0],
        #     octave_base_scale=8,
        #     scales_per_octave=1,
        #     strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.05, 0.05, 0.1, 0.1]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        # loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        stage1_reg_decoded_bbox=False,
        reg_decoded_bbox=True,
        # loss_bbox=dict(type='IoULoss', linear=True, loss_weight=10.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=10.0),
        # loss_bbox=dict(type='SmoothL1Loss', beta=0.04, loss_weight=1.0),
        # loss_centerness=dict(
        #     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    ),
    # training and testing settings
    train_cfg=dict(
        # stage1_assigner=dict(
        #     type='RegionAssigner', center_ratio=0.2, ignore_ratio=0.5),
        stage1_assigner=dict(type='ATSSAssigner', topk=9),
        stage1_sampler=dict(  # stage1
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        ####################################################
        # assigner=dict(type='ATSSAssigner', topk=9),
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.7,
            min_pos_iou=0.3,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='BboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# resume_from = "work_dirs/test_refinemtatss_r50c_fpn_2x_kins/epoch_5.pth"