_base_ = [
    '../../_base_/models/cascade_rcnn_r50_fpn.py', '../../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        type='CBRes2Net', 
        depth=101,
        scales=4,
        base_width=26,
        cb_del_stages=1,
        cb_inplanes=[64, 256, 512, 1024, 2048], 
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://res2net101_v1d_26w_4s')),
    neck=dict(
        type='CBFPN',
    ),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=4,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=4,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=4,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]
        ),
    train_cfg=dict(
        
        rpn=dict(sampler=dict(num=128)),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
        test_cfg = dict(
            rcnn=dict(
                score_thr=0.001,
                nms=dict(type='soft_nms'),
            )
        )
     )

# dataset settings
# train_class = ['引流线松股','导线本体异物','普通地线锈蚀']
train_class = ['线松股','线异物','线断股','线损伤']
sub_dataset_name = '导地线'

val_classes = train_class
dataset_type = 'SG4'
data_root = '/data/wulianjun/datasets/SG5/{}'.format(sub_dataset_name)
val_data_root = '/data/wulianjun/datasets/SG5/{}'.format(sub_dataset_name)
img_norm_cfg = dict(
    mean=[0.485*255, 0.456*255, 0.406*255],
    std=[0.229*255, 0.224*255, 0.225*255],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(1333,800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333,800),
        # img_scale=[(1333, 800) ,(3999, 2400)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='Resize', keep_ratio=True,multiscale_mode='range'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        # type='RepeatDataset',
        # times=3,
        # dataset=dict(
            type=dataset_type,
            ann_file='/data/wulianjun/datasets/SG5/{}/ImageSets/Main/train_split.txt'.format(sub_dataset_name),
            img_prefix=data_root,
            pipeline=train_pipeline,
            classes=train_class,
            val_classes=val_classes
            # )
            ),
    val=dict(
        type=dataset_type,
        ann_file='/data/wulianjun/datasets/SG5/{}/ImageSets/Main/val_split.txt'.format(sub_dataset_name),
        img_prefix=val_data_root,
        pipeline=val_pipeline,
        classes=train_class,
        val_classes=val_classes),
    test=dict(
        type=dataset_type,
        ann_file='/data/wulianjun/datasets/SG5/{}/ImageSets/Main/val_split.txt'.format(sub_dataset_name),
        img_prefix=val_data_root,
        pipeline=val_pipeline,
        classes=train_class,
        val_classes=val_classes))
evaluation = dict(interval=1, metric='mAP')
optimizer = dict(type='SGD', lr=0.01/16*4, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[4,7])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=9)
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
# runner = dict(type='EpochBasedRunner', max_epochs=9)
load_from = None
checkpoint_config = dict(interval=3)