
_base_=['./faster_rcnn_r50_fpn_1x_base.py']

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=8
            ),
    train_cfg=dict(rpn=dict(sampler=dict(num=128)),rcnn=dict(sampler=dict(num=256)))
     ))

# dataset settings
train_class = ['螺母锈蚀', '缺销', '销钉安装不到位', '螺母安装不规范', 
'缺螺母', '缺垫片', '缺螺栓', '垫片锈蚀']
val_classes = train_class
dataset_type = 'SG4'
data_root = '/data/wulianjun/StateGridv5/xiaojinju_patch'
val_data_root = '/data/wulianjun/StateGridv5/验证集/xiaojinju_patch_val'
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
        # img_scale=(1333,800),
        img_scale=[(1333, 800) ,(3999, 2400)],
        flip=False,
        transforms=[
            # dict(type='Resize', keep_ratio=True),
            dict(type='Resize', keep_ratio=True,multiscale_mode='range'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=6,
    train=dict(
        # type='RepeatDataset',
        # times=3,
        # dataset=dict(
            type=dataset_type,
            ann_file='/data/wulianjun/StateGridv5/xiaojinju_patch/ImageSets/Main/train.txt',
            img_prefix=data_root,
            pipeline=train_pipeline,
            classes=train_class,
            val_classes=val_classes
            # )
            ),
    val=dict(
        type=dataset_type,
        ann_file='/data/wulianjun/StateGridv5/验证集/xiaojinju_patch_val/ImageSets/Main/val.txt',
        img_prefix=val_data_root,
        pipeline=val_pipeline,
        classes=train_class,
        val_classes=val_classes),
    test=dict(
        type=dataset_type,
        ann_file='/data/wulianjun/StateGridv5/验证集/xiaojinju_patch_val/ImageSets/Main/val.txt',
        img_prefix=val_data_root,
        pipeline=val_pipeline,
        classes=train_class,
        val_classes=val_classes))
evaluation = dict(interval=1, metric='mAP')
optimizer = dict(type='SGD', lr=0.04/16*6, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[4,7])
runner = dict(type='EpochBasedRunner', max_epochs=9)
load_from = None
checkpoint_config = dict(interval=3)