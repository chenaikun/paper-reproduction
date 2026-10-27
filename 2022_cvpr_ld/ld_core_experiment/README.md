# LD Core Experiment

This experiment uses the bundled official LD/MMDetection 2.10 code in
`2022_cvpr_ld/LD`. All experiment configs inherit the official VOC/GFL/LD
configs; training and evaluation delegate to that checkout's `tools/` entry
points. Do not install the MMDetection 3 stack in this environment.

## Environment

The original LD guide targets Python 3.7, PyTorch 1.7, CUDA 11.0, and
`mmcv-full==1.2.7`. The default experiment environment is now the modern
compatibility path: PyTorch 2.0/2.1 with `mmcv-full==1.7.2`, while retaining the
official MMDetection 2.10 LD code and APIs. Do not install MMCV 2.x or the
MMDetection 3 stack for this checkout.

If the project was cloned from Git, initialize the official source first:

```bash
git submodule update --init --recursive
```

```bash
# Run these commands inside a platform PyTorch 2.0 or 2.1 environment.
# Verify the preinstalled versions before changing anything:
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -m pip install -U openmim
python -m pip uninstall -y mmcv mmcv-lite mmcv-full
mim install "mmcv-full==1.7.2"
python -m pip install -v -e ./2022_cvpr_ld/LD
```

`mmcv-full==1.7.2` is preferred for the PyTorch 2.x path because its published
packages cover PyTorch 2.0/2.1. If no matching package exists for the exact
platform CUDA build, install it from source and check that the CUDA toolkit is
available:

```bash
nvcc --version
```

If `nvcc` is absent, select an NVIDIA CUDA development image rather than a
runtime-only image. The PyTorch CUDA runtime alone is not enough to compile
MMCV's operators.

Check the installed stack and GPU before training:

```bash
python 2022_cvpr_ld/ld_core_experiment/scripts/check_environment.py --require-cuda
```

The check rejects MMDetection 3, MMCV 2, unsupported PyTorch versions, or a
different imported `mmdet` checkout so the scripts do not silently run against
the wrong API. The original environment is still available with
`requirements-legacy.txt` if exact paper-era reproduction is required.

## Hardware Profiles

The experiment uses the `auto` profile by default and selects a conservative
resize from the visible GPU memory. This lets the same code run on T4, V100,
RTX3090, and smaller cards without editing model configs:

```bash
# 640x384, one sample, no data-loader workers
LD_PROFILE=tiny python scripts/train_baseline.py --data-root /path/to/VOCdevkit

# 1000x600, one sample, two workers
LD_PROFILE=default python scripts/train_baseline.py --data-root /path/to/VOCdevkit

# 1333x800, one sample, four workers
LD_PROFILE=high python scripts/train_baseline.py --data-root /path/to/VOCdevkit
```

The available profiles are `tiny`, `low`, `default`, and `high`. Individual
settings can also be overridden with `LD_IMAGE_WIDTH`, `LD_IMAGE_HEIGHT`,
`SAMPLES_PER_GPU`, and `WORKERS_PER_GPU`. Dynamic-loss-scale FP16 is optional:

```bash
LD_PROFILE=low WORKERS_PER_GPU=0 \
  python scripts/train_ld.py --variant full --fp16 \
  --teacher-checkpoint runs/teacher_r101/epoch_16.pth
```

Multiple visible GPUs can be selected without changing the configs:

```bash
python scripts/train_baseline.py --gpu-ids 0,1 --profile default
```

## Dataset

Set `VOC_DATA_ROOT` to the `VOCdevkit` directory. The configs support both
separate-year data and the merged OpenI `VOC07_12` dataset.

```text
VOCdevkit/
├── VOC2007/
│   ├── Annotations/
│   ├── ImageSets/Main/trainval.txt
│   ├── ImageSets/Main/test.txt
│   └── JPEGImages/
└── VOC2012/
    ├── Annotations/
    ├── ImageSets/Main/trainval.txt
    └── JPEGImages/
```

The OpenI merged dataset may instead place the combined 2007+2012 training
split and VOC2007 test split under `VOCdevkit/VOC2007/`, with no separate
`VOC2012/` directory. The config detects the merged layout when the VOC2012
trainval split file is absent. Its merged `VOC2007/ImageSets/Main/trainval.txt`
should contain about 16,551 training images, as stated in that dataset's
description.

The default is `2022_cvpr_ld/data/VOCdevkit`. Validate it with:

```bash
python 2022_cvpr_ld/ld_core_experiment/scripts/check_dataset.py \
  --data-root /path/to/VOCdevkit
```

The low-resource configs use VOC2007+VOC2012 trainval and VOC2007 test, one
sample per GPU by default, and an automatic resize profile. These are
low-resource settings, not the paper's full COCO setup.

## Smoke Test

This performs a real forward/backward/optimizer step using the official
MMDetection 2 data/model APIs. It does not start a full epoch:

```bash
cd 2022_cvpr_ld/ld_core_experiment
VOC_DATA_ROOT=/path/to/VOCdevkit ./run_smoke.sh
```

After training the teacher, include its checkpoint to also smoke-test the LD
student and frozen-teacher forward:

```bash
VOC_DATA_ROOT=/path/to/VOCdevkit \
TEACHER_CHECKPOINT=/path/to/teacher/epoch_16.pth \
./run_smoke.sh
```

## Training Order

Train the R101 teacher first. The config uses ImageNet initialization and the
official 16-epoch VOC schedule by default:

```bash
python scripts/train_teacher.py \
  --data-root /path/to/VOCdevkit \
  --work-dir runs/teacher_r101 \
  --epochs 16 \
  --samples-per-gpu 1
```

Then train the GFL-R18 student baseline:

```bash
python scripts/train_baseline.py \
  --data-root /path/to/VOCdevkit \
  --work-dir runs/student_r18_baseline \
  --epochs 4 \
  --samples-per-gpu 1
```

Run the three teacher-student ablations. The default teacher checkpoint is
`runs/teacher_r101/epoch_16.pth`:

```bash
python scripts/train_ld.py --variant kd-only \
  --teacher-checkpoint runs/teacher_r101/epoch_16.pth
python scripts/train_ld.py --variant loc-only \
  --teacher-checkpoint runs/teacher_r101/epoch_16.pth
python scripts/train_ld.py --variant full \
  --teacher-checkpoint runs/teacher_r101/epoch_16.pth
```

The LD head uses the official `loss_kd`, `loss_ld`, and `loss_ld_vlr`
implementations. `kd-only` disables both localization losses; `loc-only`
disables classification KD; `full` retains all three. All variants retain the
student's GT detection losses.

Evaluate a checkpoint with the same config used for that run:

```bash
python scripts/evaluate_baseline.py \
  --config configs/gfl_r18_voc_baseline.py \
  --checkpoint runs/student_r18_baseline/epoch_4.pth \
  --data-root /path/to/VOCdevkit
```

For an LD checkpoint, pass its matching `configs/ld_r18_voc_*.py` config and
`--teacher-checkpoint` as well. The T4-oriented default is batch size 1; reduce
the resize further if the teacher-student step runs out of memory.
