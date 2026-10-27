import os
from pathlib import Path


_base_ = ['./_base_/gfl_r50_fpn_1x_coco.py']

# The directory must contain VOC2007/, unless VOC_DATA_ROOT points elsewhere.
# Keeping the default relative to this config makes the experiment portable
# between the local checkout and a remote training machine.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
data_root = os.environ.get('VOC_DATA_ROOT', str(PROJECT_ROOT / 'data'))
dataset_type = 'VOCDataset'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

model = dict(
    backbone=dict(
        depth=18,
        init_cfg=None),
    neck=dict(in_channels=[64, 128, 256, 512]),
    bbox_head=dict(num_classes=20))

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Main/trainval.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Main/test.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
param_scheduler = [
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))
