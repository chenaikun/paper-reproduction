# LD Core Experiment

This directory contains the low-resource experiments for the LD study.

The official `LD` repository is kept unchanged. Experimental configs, scripts,
logs, checkpoints, and analysis code belong here.

Planned layout:

```text
ld_core_experiment/
├── configs/       # local experiment configs
├── scripts/       # dataset checks and training entry points
├── analysis/      # distribution and gradient analysis
└── README.md
```

Dataset location:

```text
../data/VOC2007/
```

The first experiment is a GFL-ResNet-18 student baseline. Later experiments
will add classification KD, localization distribution distillation, and the
full combination.
