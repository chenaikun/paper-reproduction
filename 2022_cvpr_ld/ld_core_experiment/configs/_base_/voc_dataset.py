import os
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = EXPERIMENT_ROOT.parent
VOC_DATA_ROOT = Path(os.environ.get(
    'VOC_DATA_ROOT', PROJECT_ROOT / 'data' / 'VOCdevkit')).expanduser()
DATA_ROOT = str(VOC_DATA_ROOT.resolve()) + os.sep
SAMPLES_PER_GPU = int(os.environ.get('SAMPLES_PER_GPU', '1'))
WORKERS_PER_GPU = int(os.environ.get('WORKERS_PER_GPU', '2'))
GPU_COUNT = int(os.environ.get('GPU_COUNT', '1'))

dataset_type = 'VOCDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]

data_root = DATA_ROOT
optimizer = dict(lr=0.01 * SAMPLES_PER_GPU * GPU_COUNT / 16)
VOC2012_TRAIN_ANN = Path(DATA_ROOT) / 'VOC2012' / 'ImageSets' / 'Main' / 'trainval.txt'
if VOC2012_TRAIN_ANN.is_file():
    TRAIN_ANN_FILE = [
        DATA_ROOT + 'VOC2007/ImageSets/Main/trainval.txt',
        str(VOC2012_TRAIN_ANN),
    ]
    TRAIN_IMG_PREFIX = [DATA_ROOT + 'VOC2007/', DATA_ROOT + 'VOC2012/']
else:
    TRAIN_ANN_FILE = DATA_ROOT + 'VOC2007/ImageSets/Main/trainval.txt'
    TRAIN_IMG_PREFIX = DATA_ROOT + 'VOC2007/'

data = dict(
    samples_per_gpu=SAMPLES_PER_GPU,
    workers_per_gpu=WORKERS_PER_GPU,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=TRAIN_ANN_FILE,
            img_prefix=TRAIN_IMG_PREFIX,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=DATA_ROOT + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=DATA_ROOT + 'VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=DATA_ROOT + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=DATA_ROOT + 'VOC2007/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
