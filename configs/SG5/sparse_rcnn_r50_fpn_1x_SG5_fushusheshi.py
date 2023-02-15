
_base_=['./sparse_rcnn_r50_fpn_1x_base.py']

# model = dict(
#     roi_head=dict(
#         bbox_head=dict(
#             num_classes=4
#             ),
    # train_cfg=dict(rpn=dict(sampler=dict(num=128)),rcnn=dict(sampler=dict(num=256)))
    #  ))

# dataset settings
train_class = ['色标牌退色', '警告牌图文不清', '防鸟刺未打开', '驱鸟器损坏']

sub_dataset_name = '附属设施'

val_classes = train_class
dataset_type = 'SG4'
data_root = '/data/wulianjun/StateGridv5/{}'.format(sub_dataset_name)
val_data_root = '/data/wulianjun/StateGridv5/验证集/{}'.format(sub_dataset_name)
img_norm_cfg = dict(
    mean=[131.94992295, 134.26683355, 125.48345759],
    std=[47.53764514, 45.37312592, 48.50691576],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333,800), keep_ratio=True),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
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
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=12,
    train=dict(
        # type='RepeatDataset',
        # times=3,
        # dataset=dict(
            type=dataset_type,
            ann_file='/data/wulianjun/StateGridv5/{}/ImageSets/Main/train.txt'.format(sub_dataset_name),
            img_prefix=data_root,
            pipeline=train_pipeline,
            classes=train_class,
            val_classes=val_classes
            # )
            ),
    val=dict(
        type=dataset_type,
        ann_file='/data/wulianjun/StateGridv5/验证集/{}/ImageSets/Main/val.txt'.format(sub_dataset_name),
        img_prefix=val_data_root,
        pipeline=val_pipeline,
        classes=train_class,
        val_classes=val_classes),
    test=dict(
        type=dataset_type,
        ann_file='/data/wulianjun/StateGridv5/验证集/{}/ImageSets/Main/val.txt'.format(sub_dataset_name),
        img_prefix=val_data_root,
        pipeline=val_pipeline,
        classes=train_class,
        val_classes=val_classes))
evaluation = dict(interval=1, metric='mAP')

optimizer = dict(_delete_=True, type='AdamW', lr=0.000025/16*24, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2))