_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn.py', '../../_base_/default_runtime.py'
]


# optimizer
optimizer = dict(type='SGD', lr=0.04/16*12, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[6,8])
runner = dict(type='EpochBasedRunner', max_epochs=9)
load_from = None
checkpoint_config = dict(interval=3)