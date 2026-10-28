# LD Core Experiment

This experiment uses the bundled official LD/MMDetection 2.10 code in
`2022_cvpr_ld/LD`. All experiment configs inherit the official VOC/GFL/LD
configs; training and evaluation delegate to that checkout's `tools/` entry
points. Do not install the MMDetection 3 stack in this environment.

## Environment

Use Linux x86_64, Python 3.8/3.9 (3.10 also accepted), PyTorch **2.0.1**,
torchvision **0.15.2**, CUDA **11.8**, and **mmcv-full 1.7.2**. PyTorch 2.1.x
is an alternate compatibility target, not a promise that every CUDA/Python
combination has a prebuilt MMCV wheel. Do not install MMCV 2.x or MMDetection
3.x. One V100S 32GB or RTX 3090/4090 24GB is a reasonable starting point;
actual memory use must be checked with the smoke test.

From the repository root, after choosing a CUDA-enabled PyTorch image:

```bash
git submodule update --init --recursive
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python 2022_cvpr_ld/ld_core_experiment/scripts/install_environment.py
```

The installer:

1. Applies `patches/ld-mmcv-1.7.2.patch` to the bundled LD source, idempotently.
   This patch is tracked **in the parent repository** so a fresh cloud clone
   gets the same fixes, without needing to push commits to the official LD repo.
2. Pins NumPy, YAPF (MMCV's `verify=True` formatting API), OpenCV and COCO
   dependencies; replaces the abandoned `mmpycocotools` dependency/check with
   maintained `pycocotools`; keeps the original LD loss/model implementations.
3. Removes conflicting MMCV distributions, then requires a matching binary
   `mmcv-full` wheel from the PyTorch/CUDA-specific OpenMMLab index. It does
   **not** silently start compiling when no wheel exists.
4. Installs the patched local LD checkout in editable mode and executes CPU
   and CUDA NMS plus ROIAlign forward/backward, model registration and config
   formatting checks.

A compiler or wheel-index/network failure is an installation failure, not a
successful environment check. If a matching wheel is unavailable, use a CUDA
**development** image (not runtime-only), check that `nvcc --version` matches
`torch.version.cuda`, then explicitly opt into compilation:

```bash
python 2022_cvpr_ld/ld_core_experiment/scripts/install_environment.py --build-mmcv
```

To reapply/check only the source patch (no Python dependencies required):

```bash
python 2022_cvpr_ld/ld_core_experiment/scripts/apply_ld_compat.py
python 2022_cvpr_ld/ld_core_experiment/scripts/apply_ld_compat.py --check
```

Patch conflicts stop without changing source files; do not discard your own
LD edits to force installation. A modified/dirty LD submodule after patching
is expected. Commit the parent experiment's patch/script files, **not** a new
LD gitlink pointing to an unpublished local commit.

Run environment checks again before training:

```bash
python 2022_cvpr_ld/ld_core_experiment/scripts/check_environment.py --require-cuda
```

Operator checks do not replace a real teacher/student training step. The
single-step VOC checks below still require a GPU, data, and (for LD) a trained
teacher checkpoint. The `requirements-legacy.txt` file records the old MMCV
reference only; the installer/checker above target the modern stack.

Dependency-free patch regression tests can run on a development laptop:

```bash
python -m unittest discover -s 2022_cvpr_ld/ld_core_experiment/tests -v
```

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

Select one GPU for this non-distributed MMCV runner:

```bash
python scripts/train_baseline.py --gpu-id 0 --profile default
```

Multiple `--gpu-ids` are rejected: MMCV 1.7.2's `MMDataParallel.train_step`
requires a single GPU. Distributed multi-GPU launching is not implemented by
these wrappers.

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
