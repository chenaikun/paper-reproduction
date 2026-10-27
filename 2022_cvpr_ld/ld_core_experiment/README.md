# LD Core Experiment

This directory contains the low-resource GFL baseline and the follow-up
Localization Distillation experiments. The current runnable milestone is the
student baseline smoke test; the teacher-student LD loss is added only after
that baseline is verified.

## 1. Expected data layout

`VOC_DATA_ROOT` must point to the directory containing `VOC2007/`:

```text
2022_cvpr_ld/
├── data/
│   └── VOC2007/
│       ├── Annotations/
│       ├── ImageSets/Main/trainval.txt
│       ├── ImageSets/Main/test.txt
│       └── JPEGImages/
└── ld_core_experiment/
```

The default is `2022_cvpr_ld/data`. On a remote machine, use an absolute path
without changing the config:

```bash
export VOC_DATA_ROOT=/path/to/data
```

## 2. Environment

Use a Python environment with a CUDA-enabled PyTorch installation. Install the
PyTorch and torchvision wheels that match the CUDA image first. Then install
the OpenMMLab packages:

```bash
python -m pip install -U pip openmim
python -m pip install -r 2022_cvpr_ld/ld_core_experiment/requirements.txt
```

If `mmcv` has no matching prebuilt wheel for the selected PyTorch/CUDA pair,
install it with OpenMIM instead of forcing a source build:

```bash
mim install "mmengine>=0.10.0,<1.0" "mmcv>=2.0.0,<2.2.0" "mmdet>=3.0.0,<3.4.0"
```

Verify the runtime before spending GPU time:

```bash
python 2022_cvpr_ld/ld_core_experiment/scripts/check_environment.py --require-cuda
```

## 3. Smoke test

The smoke test checks one forward/backward step and then trains a small subset
for one epoch. It disables validation to keep the first remote run short.

```bash
cd 2022_cvpr_ld/ld_core_experiment
VOC_DATA_ROOT=/path/to/data ./run_smoke.sh
```

Optional overrides:

```bash
SMOKE_SAMPLES=32 SMOKE_EPOCHS=1 WORK_DIR=/tmp/ld-smoke \
  VOC_DATA_ROOT=/path/to/data ./run_smoke.sh
```

The expected checkpoint is:

```text
ld_core_experiment/runs/smoke/epoch_1.pth
```

## 4. Baseline training and evaluation

After the smoke test succeeds, run the configured student baseline:

```bash
python scripts/train_baseline.py \
  --data-root /path/to/data \
  --epochs 12 \
  --work-dir runs/student_r18_baseline
```

Evaluate a checkpoint:

```bash
python scripts/evaluate_baseline.py \
  --config configs/gfl_r18_voc_baseline.py \
  --checkpoint runs/student_r18_baseline/epoch_12.pth \
  --data-root /path/to/data \
  --work-dir runs/student_r18_baseline_eval
```

Inspect the GFL output shapes before implementing distribution distillation:

```bash
python scripts/inspect_gfl_outputs.py --data-root /path/to/data
```

The current config uses VOC2007 trainval and VOC2007 test. The broader
VOC2007+VOC2012 training split described in the research notes still needs a
separate dataset config.

## 5. Current scope

The files currently implement:

- GFL-ResNet-18 student baseline;
- GFL-ResNet-50 teacher architecture config;
- dataset, environment, forward/backward, training, and evaluation checks.

They do not yet implement the four final ablations. In particular, the
teacher config is not a distillation trainer. The official source is available
as the `2022_cvpr_ld/LD` submodule. It bundles MMDetection 2.10.0, while these
baseline scripts/configs use the MMDetection 3 API. Do not install both code
paths into the same Python environment; `requirements.txt` here is for the
MMDetection 3 baseline only. For official LD configs, follow
`2022_cvpr_ld/LD/docs/install.md` in a separate environment. The next step is
to choose one implementation path and make its VOC config/data layout and
smoke test reproducible before starting long training runs.
