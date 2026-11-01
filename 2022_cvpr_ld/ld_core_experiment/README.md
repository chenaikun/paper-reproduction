# LD 核心机制复现实验

本实验使用仓库内置的官方 LD / MMDetection 2.10 代码（位于 `2022_cvpr_ld/LD`）。所有实验配置都继承官方 VOC / GFL / LD 配置，训练与评估都委托给该 checkout 的 `tools/` 入口。**不要在这个环境里安装 MMDetection 3 那一套。**

相关文档：

- [`docs/zero_ld_core_experiment.md`](docs/zero_ld_core_experiment.md) —— 零基础版全流程讲解（先读这份）
- [`docs/ld_core_experiment_code.md`](docs/ld_core_experiment_code.md) —— 逐文件代码走读 + 已知坑
- [`docs/练习_官方LD.md`](docs/练习_官方LD.md) —— 手写练习与官方代码的逐行对照
- [`one_epoch_mechanism_results.md`](one_epoch_mechanism_results.md) —— 本轮机制实验的原始记录

---

## 一、环境

已完成的核心机制实验所用环境：

| 项 | 记录值 |
|---|---|
| GPU | 单张 Tesla V100S-PCIE-32GB |
| 操作系统 | Linux x86_64 |
| Python | 3.8.18 |
| PyTorch | 2.0.0+cu118 |
| torchvision | 0.15.1+cu118 |
| CUDA / NVCC | 11.8 |
| MMCV | `mmcv-full==1.7.2` |
| 检测框架 | 内置 LD / MMDetection 2.10.0+4287990 |
| 训练数据 | VOC2007 trainval + VOC2012 trainval，16,551 张 |
| 测试数据 | VOC2007 test，4,952 张 |

安装脚本接受 PyTorch 2.0.x 或 2.1.x，并据此选择对应 torchvision 与 CUDA 版本的 `mmcv-full==1.7.2` 预编译 wheel。本次记录使用的是 PyTorch 2.0.0 + torchvision 0.15.1。**不要安装 MMCV 2.x 或 MMDetection 3.x。**

本次记录的云端路径：

```text
数据：  /cloud/cloud-ssd1/ld-data/VOCdevkit
产物：  /cloud/cloud-ssd1/ld-runs
日志：  /cloud/cloud-ssd1/ld-runs/logs
```

一轮机制实验的汇总见 [`one_epoch_mechanism_results.md`](one_epoch_mechanism_results.md)。

在仓库根目录，准备好带 CUDA 的 PyTorch 镜像后：

```bash
git submodule update --init --recursive
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python 2022_cvpr_ld/ld_core_experiment/scripts/install_environment.py
```

安装脚本做的事情：

1. 幂等地把 `patches/ld-mmcv-1.7.2.patch` 打进内置的 LD 源码。该补丁由**父仓库**跟踪，因此云端全新 clone 也能得到相同的修复，无需往官方 LD 仓库推 commit。
2. 固定 NumPy、YAPF（MMCV 的 `verify=True` 格式化接口依赖它）、OpenCV 和 COCO 相关依赖；把已废弃的 `mmpycocotools` 依赖与版本检查替换为维护中的 `pycocotools`；保留官方 LD 的 loss 与模型实现。
3. 卸载冲突的 MMCV 发行版，然后从 OpenMMLab 按 PyTorch/CUDA 区分的索引安装匹配的 `mmcv-full` **二进制 wheel**。**找不到 wheel 时不会悄悄开始编译。**
4. 以 editable 模式安装打好补丁的本地 LD，并执行 CPU / CUDA 的 NMS 与 ROIAlign 前反向、模型注册和配置格式化检查。

编译失败或 wheel 索引/网络失败，都算安装失败，不算环境检查通过。如果确实没有匹配的 wheel：换用 CUDA **开发版**镜像（不要用 runtime-only），确认 `nvcc --version` 与 `torch.version.cuda` 一致，然后显式开启编译：

```bash
python 2022_cvpr_ld/ld_core_experiment/scripts/install_environment.py --build-mmcv
```

只想重新打补丁或检查补丁状态（不需要任何 Python 依赖）：

```bash
python 2022_cvpr_ld/ld_core_experiment/scripts/apply_ld_compat.py
python 2022_cvpr_ld/ld_core_experiment/scripts/apply_ld_compat.py --check
```

补丁冲突时会直接停止且不改动任何源文件；不要为了让安装通过而丢弃自己对 LD 的修改。打完补丁后 LD 子模块处于"已修改"状态是**预期行为**。需要提交的是父仓库里的补丁和脚本文件，**不要**提交一个指向本地未发布 commit 的 LD gitlink。

训练前再跑一次环境检查：

```bash
python 2022_cvpr_ld/ld_core_experiment/scripts/check_environment.py --require-cuda
```

算子检查不能替代真正的教师/学生训练一步。下面的 VOC 单步检查仍然需要 GPU、数据，LD 那一步还需要训练好的教师权重。`requirements-legacy.txt` 只用于记录旧的 MMCV 版本参考；上面的安装与检查脚本都面向新栈。

不依赖第三方库的补丁回归测试可以在开发机上直接跑：

```bash
python -m unittest discover -s 2022_cvpr_ld/ld_core_experiment/tests -v
```

---

## 二、硬件档位

实验默认使用 `auto` 档位，根据可见显存自动选择保守的输入尺寸。本次 V100S 记录使用的是 `tiny`（`640x384`）、batch size 4、2 个 dataloader worker、动态 loss-scale FP16、单卡。各学生变体的显存占用都远低于 32 GB：

```text
Baseline 训练：约 812 MiB
KD-only 训练：约 1,189 MiB
Full LD 冒烟（batch size 1）：约 506 MiB
```

可选档位如下：

```bash
# 640x384，单样本，无 dataloader worker
LD_PROFILE=tiny python scripts/train_baseline.py --data-root /path/to/VOCdevkit

# 1000x600，单样本，2 个 worker
LD_PROFILE=default python scripts/train_baseline.py --data-root /path/to/VOCdevkit

# 1333x800，单样本，4 个 worker
LD_PROFILE=high python scripts/train_baseline.py --data-root /path/to/VOCdevkit
```

可用档位为 `tiny`、`low`、`default`、`high`。单项设置也可以用 `LD_IMAGE_WIDTH`、`LD_IMAGE_HEIGHT`、`SAMPLES_PER_GPU`、`WORKERS_PER_GPU` 覆盖。动态 loss-scale 的 FP16 是可选项：

```bash
LD_PROFILE=low WORKERS_PER_GPU=0 \
  python scripts/train_ld.py --variant full --fp16 \
  --teacher-checkpoint runs/teacher_r101/epoch_16.pth
```

所有封装脚本都提供 `--gpu-id`，因为这套非分布式的 MMCV runner **只支持单卡**：

```bash
python scripts/train_baseline.py --gpu-id 0 --profile default
```

这些封装脚本没有实现分布式多卡启动。

---

## 三、数据集

把 `VOC_DATA_ROOT` 指向 `VOCdevkit` 目录。配置同时支持按年份分开的布局和 OpenI 合并版 `VOC07_12`。

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

OpenI 的合并版可能把 2007+2012 的合并训练划分与 VOC2007 test 划分都放在 `VOCdevkit/VOC2007/` 下，没有单独的 `VOC2012/` 目录。配置在 **VOC2012 的 trainval 划分文件不存在**时判定为合并版。合并版的 `VOC2007/ImageSets/Main/trainval.txt` 应包含约 16,551 张训练图，与该数据集的说明一致。

默认路径会从 `2022_cvpr_ld/data/VOCdevkit` 或 `2022_cvpr_ld/data` 中自动探测（取包含 `VOC2007/` 的那个）。校验命令：

```bash
python 2022_cvpr_ld/ld_core_experiment/scripts/check_dataset.py \
  --data-root /path/to/VOCdevkit --full
```

低资源配置使用 VOC2007+VOC2012 trainval 训练、VOC2007 test 测试，默认每卡 1 个样本，输入尺寸自动选择。**这些是低资源设置，不是论文的完整 COCO 协议。**

---

## 四、冒烟测试

它会用官方 MMDetection 2 的数据与模型 API 真跑一次 forward / backward / optimizer step，不启动完整 epoch：

```bash
cd 2022_cvpr_ld/ld_core_experiment
VOC_DATA_ROOT=/path/to/VOCdevkit ./run_smoke.sh
```

训练完教师后，可以带上教师权重，一并冒烟测试 LD 学生与冻结教师的前向：

```bash
VOC_DATA_ROOT=/path/to/VOCdevkit \
TEACHER_CHECKPOINT=/path/to/teacher/epoch_16.pth \
./run_smoke.sh
```

---

## 五、已完成的一轮机制实验

本次记录的机制实验：在合并后的 VOC07+12 训练集上完整过一遍，`640x384`，单卡 V100S 上 batch size 4，FP16，离线评估。所有学生变体共用这套设置，从而让比较只反映蒸馏 loss 的差异。`LD_REPEAT_TIMES=1` 把"一个 epoch"从官方的三次重复遍历改为完整遍历 16,551 张一次。

```bash
export VOC_DATA_ROOT=/path/to/VOCdevkit
export LD_PROFILE=tiny
export LD_REPEAT_TIMES=1
export WORKERS_PER_GPU=2
TEACHER=/path/to/teacher_r101/epoch_1.pth

TEACHER_CHECKPOINT="${TEACHER}" ./run_smoke.sh

python scripts/train_baseline.py --epochs 1 --repeat-times 1 \
  --samples-per-gpu 4 --fp16 --no-validate --seed 0 --deterministic
python scripts/train_ld.py --variant kd-only \
  --teacher-checkpoint "${TEACHER}" --epochs 1 \
  --repeat-times 1 --samples-per-gpu 4 --fp16 --no-validate \
  --seed 0 --deterministic
python scripts/train_ld.py --variant loc-only \
  --teacher-checkpoint "${TEACHER}" --epochs 1 \
  --repeat-times 1 --samples-per-gpu 4 --fp16 --no-validate \
  --seed 0 --deterministic
python scripts/train_ld.py --variant full \
  --teacher-checkpoint "${TEACHER}" --epochs 1 \
  --repeat-times 1 --samples-per-gpu 4 --fp16 --no-validate \
  --seed 0 --deterministic
python scripts/train_ld.py --variant main-ld-only \
  --teacher-checkpoint "${TEACHER}" --epochs 1 \
  --repeat-times 1 --samples-per-gpu 4 --fp16 --no-validate \
  --seed 0 --deterministic
```

这验证的是蒸馏链路与方向性趋势，**不是**复现论文的 COCO AP 或 12/16 epoch 训练计划。完整协议与实测 VOC 结果记录在 [`one_epoch_mechanism_results.md`](one_epoch_mechanism_results.md)。

> 注：那些已记录的运行早于 `--seed` 与 `--repeat-times` 命令行选项的加入；后续运行请使用上面显式的 seed 0。

---

## 六、训练顺序

**第一步：训练 R101 教师。** 配置默认使用 ImageNet 初始化与官方 16 epoch VOC 计划：

```bash
python scripts/train_teacher.py \
  --data-root /path/to/VOCdevkit \
  --work-dir runs/teacher_r101 \
  --epochs 16 \
  --samples-per-gpu 1
```

如果希望先用 GFL COCO 检测器权重初始化、再做 VOC 微调，可以传 `--load-from /path/to/gfl_r101_coco.pth`。80 类的分类头与 20 类的 VOC 头对不上，但骨干、neck 和能对上的 head 权重仍然可以用来初始化教师。

**第二步：训练 GFL-R18 学生基线。**

```bash
python scripts/train_baseline.py \
  --data-root /path/to/VOCdevkit \
  --work-dir runs/student_r18_baseline \
  --epochs 4 \
  --samples-per-gpu 1
```

**第三步：跑教师-学生的各组消融。** 教师权重的默认路径是 `runs/teacher_r101/epoch_16.pth`：

```bash
python scripts/train_ld.py --variant kd-only \
  --teacher-checkpoint runs/teacher_r101/epoch_16.pth
python scripts/train_ld.py --variant loc-only \
  --teacher-checkpoint runs/teacher_r101/epoch_16.pth
python scripts/train_ld.py --variant full \
  --teacher-checkpoint runs/teacher_r101/epoch_16.pth
```

LD head 使用官方的 `loss_kd`、`loss_ld`、`loss_ld_vlr` 实现：

| 变体 | `loss_kd` | `loss_ld` | `loss_ld_vlr` |
|---|---:|---:|---:|
| `kd-only` | 10 | 0 | 0 |
| `main-ld-only` | 0 | 0.25 | 0 |
| `loc-only` | 0 | 0.25 | 0.25 |
| `full` | 10 | 0.25 | 0.25 |

所有变体都保留学生自己的 GT 检测损失。

**评估**：用与该次训练相同的 config 评估 checkpoint：

```bash
python scripts/evaluate_baseline.py \
  --config configs/gfl_r18_voc_baseline.py \
  --checkpoint runs/student_r18_baseline/epoch_4.pth \
  --data-root /path/to/VOCdevkit \
  --profile tiny
```

如果是 LD 的 checkpoint，除了对应的 `configs/ld_r18_voc_*.py`，还要同时传 `--teacher-checkpoint`。封装脚本默认仍是 batch size 1 以便移植；V100S 上的机制实验用的是 `640x384`、batch size 4。

评估的 stdout 会自动保存到 checkpoint 同目录下的 `<config>_eval.log`，也可以用 `--log-file` 指定其它路径。

**想做学生与教师输出的直接对比**，可以跑确定性的分布诊断脚本：

```bash
python analysis/analyze_output_distances.py \
  --config configs/ld_r18_voc_loc_only.py \
  --checkpoint /path/to/mech_loc_only_run/epoch_1.pth \
  --teacher-checkpoint /path/to/teacher_r101/epoch_1.pth \
  --profile tiny
```

该脚本会在学生 checkpoint 旁边写出 `output_distances.json`，并在每个 FPN 层上报告"分类软标签 L1 误差"以及"四条边定位分布 L1 误差"。
