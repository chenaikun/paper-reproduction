import os
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = EXPERIMENT_ROOT.parent
VOC_DATA_ROOT = Path(os.environ.get(
    'VOC_DATA_ROOT', PROJECT_ROOT / 'data' / 'VOCdevkit')).expanduser()
DATA_ROOT = str(VOC_DATA_ROOT.resolve()) + os.sep

RUNTIME_PROFILES = {
    'tiny': dict(width=640, height=384, samples_per_gpu=1, workers=0),
    'low': dict(width=800, height=480, samples_per_gpu=1, workers=0),
    'default': dict(width=1000, height=600, samples_per_gpu=1, workers=2),
    'high': dict(width=1333, height=800, samples_per_gpu=1, workers=4),
}


def _auto_profile():
    try:
        import torch
        if torch.cuda.is_available():
            memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if memory_gib < 12:
                return 'tiny'
            if memory_gib < 20:
                return 'low'
    except Exception:
        pass
    return 'default'


PROFILE_NAME = os.environ.get('LD_PROFILE', 'auto').lower()
if PROFILE_NAME == 'auto':
    PROFILE_NAME = _auto_profile()
if PROFILE_NAME not in RUNTIME_PROFILES:
    raise ValueError(
        'Unknown LD_PROFILE {!r}; choose auto, tiny, low, default, or high'.format(
            PROFILE_NAME))
PROFILE = RUNTIME_PROFILES[PROFILE_NAME]


def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value is not None else default


IMAGE_WIDTH = _env_int('LD_IMAGE_WIDTH', PROFILE['width'])
IMAGE_HEIGHT = _env_int('LD_IMAGE_HEIGHT', PROFILE['height'])
SAMPLES_PER_GPU = _env_int('SAMPLES_PER_GPU', PROFILE['samples_per_gpu'])
WORKERS_PER_GPU = _env_int('WORKERS_PER_GPU', PROFILE['workers'])
GPU_COUNT = _env_int('GPU_COUNT', 1)

dataset_type = 'VOCDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(IMAGE_WIDTH, IMAGE_HEIGHT), keep_ratio=True),
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
        img_scale=(IMAGE_WIDTH, IMAGE_HEIGHT),
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

if os.environ.get('USE_FP16', '0').lower() in ('1', 'true', 'yes', 'on'):
    fp16 = dict(loss_scale='dynamic')
