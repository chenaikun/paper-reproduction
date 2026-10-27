# LD Core Experiment

This experiment uses the bundled official LD/MMDetection 2.10 code in
`2022_cvpr_ld/LD`. All experiment configs inherit the official VOC/GFL/LD
configs; training and evaluation delegate to that checkout's `tools/` entry
points. Do not install the MMDetection 3 stack in this environment.

## Environment

The official LD guide targets Python 3.7, PyTorch 1.7, CUDA 11.0, and
`mmcv-full==1.2.7`. The Qizhi image currently starts at PyTorch 1.11, so use the
compatibility path below: keep the official LD/MMDetection 2.10 code and build
the pinned MMCV version locally against the installed PyTorch. Do not install
MMCV 2.x or the MMDetection 3 stack.

If the project was cloned from Git, initialize the official source first:

```bash
git submodule update --init --recursive
```

```bash
# Run these commands inside the platform-provided PyTorch 1.11 environment.
# Verify the preinstalled versions before changing anything:
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -m pip install pip==24.0
python -m pip install setuptools==59.5.0 wheel cython ninja
python -m pip uninstall -y mmcv mmcv-full
python -m pip install --no-binary mmcv-full \
  -r 2022_cvpr_ld/ld_core_experiment/requirements.txt
python -m pip install -v -e ./2022_cvpr_ld/LD
```

The MMCV source build can take several minutes and needs a CUDA toolkit with
`nvcc` in the image. Check it before starting the build:

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

The check rejects MMDetection 3, MMCV 2, or a different imported `mmdet`
checkout so the scripts do not silently run against the wrong API. It accepts
both the official PyTorch 1.7 stack and the PyTorch 1.11 compatibility stack.

## Dataset

Set `VOC_DATA_ROOT` to the directory containing both VOC years:

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

The default is `2022_cvpr_ld/data/VOCdevkit`. Validate it with:

```bash
python 2022_cvpr_ld/ld_core_experiment/scripts/check_dataset.py \
  --data-root /path/to/VOCdevkit
```

The low-resource configs use VOC2007+VOC2012 trainval and VOC2007 test, one
sample per GPU by default, and a 1000x600 resize. These are low-resource
settings, not the paper's full COCO setup.

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
