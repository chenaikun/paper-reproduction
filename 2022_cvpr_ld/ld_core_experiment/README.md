# LD Core Experiment

This experiment uses the bundled official LD/MMDetection 2.10 code in
`2022_cvpr_ld/LD`. All experiment configs inherit the official VOC/GFL/LD
configs; training and evaluation delegate to that checkout's `tools/` entry
points. Do not install the MMDetection 3 stack in this environment.

## Environment

The official install guide targets Python 3.7, PyTorch 1.7, CUDA 11.0, and
`mmcv-full==1.2.7`. Create a separate environment and install the CUDA-matched
MMCV wheel before installing the LD checkout:

```bash
conda create -n ld-core python=3.7 -y
conda activate ld-core
conda install pytorch=1.7 torchvision cudatoolkit=11.0 -c pytorch
python -m pip install pip==24.0
python -m pip install -r 2022_cvpr_ld/ld_core_experiment/requirements.txt \
  -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
python -m pip install -v -e ./2022_cvpr_ld/LD
```

If the project was cloned from Git, initialize the official source first:

```bash
git submodule update --init --recursive
```

Check the installed stack and GPU before training:

```bash
python 2022_cvpr_ld/ld_core_experiment/scripts/check_environment.py --require-cuda
```

The check rejects MMDetection 3, MMCV 2, or a different imported `mmdet`
checkout so the scripts do not silently run against the wrong API.

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
