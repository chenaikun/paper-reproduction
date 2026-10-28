# `ld_core_experiment` 代码导读

> 目标：读完这份文档，你能不看代码就讲清楚"一条命令是怎么变成一个 loss 的"，并且知道每个文件改坏了会出什么事。

---

## 0. 一句话定位

`ld_core_experiment` 不是模型代码，是**一层壳**：

- 模型、loss、数据管道全部来自 `2022_cvpr_ld/LD`（官方 MMDetection 2.10 分支）；
- 这层壳只做三件事：**改配置**（数据集/分辨率/学习率）、**拼环境变量**、**用 `subprocess` 调官方 `tools/train.py` 和 `tools/test.py`**；
- 唯一会碰官方代码的地方是 `patches/ld-mmcv-1.7.2.patch`，而且是"打补丁"而不是"改源码"。

所以阅读这套代码的正确姿势是：**先搞懂配置文件怎么被合并出来，再搞懂官方那两个入口被喂了什么**。剩下的脚本都是围绕这两件事的护栏。

---

## 1. 三层心智模型

```text
┌──────────────────────────────────────────────────────────────┐
│ 第 3 层  驱动脚本  scripts/*.py                               │
│          argparse → build_env() → subprocess 调官方 tools/    │
└───────────────────────────┬──────────────────────────────────┘
                            │ 环境变量（唯一配置通道）+ 命令行参数
┌───────────────────────────▼──────────────────────────────────┐
│ 第 2 层  薄壳配置  configs/*.py                                │
│          _base_ 指官方配置 → globals().update(运行时覆盖)      │
└───────────────────────────┬──────────────────────────────────┘
                            │ mmcv.Config.fromfile() 递归合并
┌───────────────────────────▼──────────────────────────────────┐
│ 第 1 层  官方 LD 代码  LD/                                     │
│          tools/train.py → train_detector → KD 检测器 → LDHead   │
└──────────────────────────────────────────────────────────────┘
```

关键约束：**第 2、3 层不允许 import 官方 Python 模块**。除了 `check_baseline.py` 和 `inspect_gfl_outputs.py` 这两个"自检/调试"脚本，所有训练都走 `subprocess`。这样做的好处是官方代码出问题时，`git stash` 一下就能定位是不是壳的锅；坏处是错误信息要跨进程看。

---

## 2. 目录速览：每个文件一句话

| 文件 | 一句话 | 需要 GPU？ |
|---|---|---|
| `configs/_base_/voc_dataset.py` | 环境变量 → config 的翻译器，整个实验的"超参中枢" | 否 |
| `configs/gfl_r101_voc_teacher.py` | 官方 `gfl_r101_fpn_voc.py` + 运行时覆盖 | 否 |
| `configs/gfl_r18_voc_baseline.py` | 官方 `gfl_r18_fpn1x_voc.py` + 运行时覆盖 | 否 |
| `configs/ld_r18_voc.py` | 官方 `ld_r18_gflv1_r101_fpn_voc_1x.py` + 教师路径 + 三个 KD 权重 | 否 |
| `configs/ld_r18_voc_full.py` | 只写一行 `_base_`，等价于基配置 | 否 |
| `configs/ld_r18_voc_kd_only.py` | 把 `loss_ld`/`loss_ld_vlr` 权重置 0 | 否 |
| `configs/ld_r18_voc_loc_only.py` | 把 `loss_kd` 权重置 0 | 否 |
| `scripts/_common.py` | 路径常量、profile 解析、`build_env`、`run_train`/`run_test` | 否 |
| `scripts/train_teacher.py` | 训 R101 教师，默认 16 epoch | 是 |
| `scripts/train_baseline.py` | 训 GFL-R18 无蒸馏基线，默认 4 epoch | 是 |
| `scripts/train_ld.py` | 训三个蒸馏变体，**强制检查教师权重存在** | 是 |
| `scripts/evaluate_baseline.py` | 调官方 `tools/test.py --eval mAP` | 是 |
| `scripts/apply_ld_compat.py` | 幂等打补丁 / `--check` 只检查 | 否 |
| `scripts/install_environment.py` | 装 mmcv-full 1.7.2 wheel，拒绝静默编译，**只允许 Linux** | 是 |
| `scripts/check_environment.py` | 真跑 NMS + ROIAlign 前后向，校验注册表与版本 | 是 |
| `scripts/check_dataset.py` | 纯 stdlib 校验 VOC 划分文件与图片/标注存在性 | 否 |
| `scripts/check_baseline.py` | 真跑一步 forward+backward+step，打印各 loss 与显存 | 是 |
| `scripts/inspect_gfl_outputs.py` | 打印五层特征、cls logits、bbox 分布 logits 的形状 | 是 |
| `patches/ld-mmcv-1.7.2.patch` | 10 个文件的兼容性修复（详见第 8 节） | 否 |
| `tests/test_compat.py` | 不依赖 mmcv 的补丁回归测试 | 否 |
| `run_smoke.sh` | 把上面几个检查串成一条冒烟命令 | 部分 |

---

## 3. 完整调用链：从命令行到 loss

以最有代表性的一条命令为例：

```bash
python scripts/train_ld.py --variant loc-only \
  --teacher-checkpoint runs/teacher_r101/epoch_16.pth --epochs 4
```

### 3.1 `train_ld.py` 自己做的事

`train_ld.py:30-41`：

1. `teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()`，然后 `if not is_file(): raise SystemExit`。
   **这是硬性门禁**——没有教师权重连 config 都不会加载。设计意图是对的（否则训练到第 1 个 iter 才报错），但意味着"只想验证配置能不能解析"这件事它不给你做。
2. `config = args.config or LD_CONFIGS[args.variant]`，`work_dir` 默认 `runs/ld_<variant>`。
3. `gpu_count = len(gpu_ids.replace(',', ' ').split())` —— 只是数个数，传给 `build_env` 用来缩放学习率，**不代表真支持多卡**（见 5.4）。
4. `env = build_env(...)`。
5. `run_train(config, work_dir, env, gpu_ids, epochs, resume_from, no_validate)`。

### 3.2 `_common.build_env()` 做的事

`_common.py:69-93`：

```python
env = os.environ.copy()                      # 继承当前 shell 的所有变量
resolved_profile = resolve_profile(profile)  # auto → tiny/low/default
env['VOC_DATA_ROOT'] = 绝对路径
env['LD_PROFILE']    = 解析后的档位名
env['LD_IMAGE_WIDTH' / 'LD_IMAGE_HEIGHT']    # 显式值优先，否则取档位
env['SAMPLES_PER_GPU' / 'WORKERS_PER_GPU']   # 命令行参数 > 已有环境变量 > 档位
env['GPU_COUNT']     # 仅当显式传入
env['USE_FP16']      # 仅当显式传入
env['TEACHER_CHECKPOINT'] = 绝对路径
```

两个细节值得记住：

- **优先级是"命令行 > 已存在的环境变量 > profile 默认值"**，而不是反过来。所以 `LD_IMAGE_WIDTH=800 python train_ld.py` 不会被 profile 覆盖（`voc_dataset.py` 里也是同样逻辑，两处对称）。
- `build_env` 只往 `env` 字典里写，**不修改 `os.environ`**。真正 export 是在 `subprocess.run(..., env=env)` 那一刻。`check_baseline.py:48-52` 是唯一的例外——它必须在同进程里生效，所以手动 copy 回 `os.environ`。

### 3.3 `run_train()` 做的事

`_common.py:109-130`：

```python
train_tool = require_ld_checkout()        # ① 子模块在不在 + 补丁打没打
normalized_gpu_ids = _normalize_gpu_ids(gpu_ids)
if len(normalized_gpu_ids) != 1: raise SystemExit(...)   # ② 单卡门禁
command = [sys.executable, str(train_tool), str(config),
           '--work-dir', work_dir, '--gpu-ids', ...]
if resume_from: command += ['--resume-from', ...]
if no_validate: command.append('--no-validate')
if epochs is not None: command += ['--cfg-options', 'total_epochs=N']   # ③
subprocess.run(command, cwd=str(LD_ROOT), env=env, check=True)          # ④
```

① `require_ld_checkout()`（`_common.py:96-106`）做两件事：找 `LD/tools/train.py`；调 `apply_patch(check_only=True)` 确认返回 `'applied'`。返回的不是 `'applied'`（也就是 `'needed'`，说明补丁没打）就直接退出并提示跑 `apply_ld_compat.py`。这避免了"环境没配好训了三小时才崩"。

② `sys.executable` 而不是 `'python'`：保证子进程和你当前解释器是同一个，不会因为 PATH 里另一个 python 把环境搞混。

③ `--cfg-options total_epochs=N lr_config.step=[M]`：
`run_train` 会同步修改 epoch 数和学习率衰减点。衰减点统一取
`ceil(0.75 * epochs)`，因此 4 epoch 对应 `[3]`、16 epoch
对应 `[12]`。这样用 8 epoch 时得到 `[6]`，不会继续沿用 base config
中为其他 epoch 数设置的旧衰减点。

④ `cwd=LD_ROOT`：官方 `tools/train.py` 里有相对路径逻辑（`work_dirs/`、`teacher_config`），必须在 LD 根目录下跑。

### 3.4 config 被加载时发生的事（重点）

`mmcv.Config.fromfile()` 的内部顺序大致是：

1. 读文件文本，用 `_substitute_predefined_vars()` 把 `{{ fileDirname }}` 等替换成真实路径，写到同目录的临时文件；
2. `import_module()` 把临时文件当 Python 模块执行——**所以 config 里可以写任意 Python 代码**，这是第 4 节所有技巧的基础；
3. 从模块 `__dict__` 里收集所有不以 `__` 开头的名字，过滤掉 module 对象，得到 `cfg_dict`；
4. 如果 `cfg_dict` 里有 `_base_`，递归加载每个 base，**深度合并**（dict 递归 merge，list 整体替换），最后把子配置的键覆盖上去。

第 4 步的"深度合并"很重要，举个真实例子：子配置里写 `optimizer = dict(lr=0.000625)`，合并后 `optimizer` 仍然是 `dict(type='SGD', momentum=0.9, weight_decay=0.0001, lr=0.000625)`——不是只剩 `lr`。

---

## 4. 配置系统：这套代码最烧脑的部分

### 4.1 `{{ fileDirname }}` 是什么

`configs/ld_r18_voc.py:9,12,24,25` 里出现的 `'{{ fileDirname }}/...'`，**不是 Python 语法，也不是某个 IDE 的模板变量，是 mmcv 内置的预定义变量**。

mmcv 1.7.2 的 `Config._substitute_predefined_vars` 支持四个：

| 变量 | 替换成 |
|---|---|
| `{{ fileDirname }}` | 当前 config 文件所在目录（反斜杠已转成 `/`） |
| `{{ fileBasename }}` | 文件名带扩展名 |
| `{{ fileBasenameNoExtension }}` | 文件名不带扩展名 |
| `{{ fileExtname }}` | 扩展名 |

它解决的是一个真实痛点：`mmcv.Config.fromfile()` 会把 config 拷到临时目录再 import，如果 config 里写相对路径，一到运行时就被解释成"相对临时目录"。用 `{{ fileDirname }}` 锚定到源码目录，就能做到**不依赖 CWD**。

> 代价：这些 config 只能用 `mmcv.Config` 读，用 `json.dump`、`yaml.safe_load`、或者自己 `exec` 都拿不到替换结果。这是把它们锁死在 mmcv 1.7.2 上的一个隐性成本。

### 4.2 为什么用 `_base_` + `globals().update()` 两段式

`configs/gfl_r18_voc_baseline.py` 全文只有 9 行：

```python
_base_ = '../../LD/configs/gfl/gfl_r18_fpn1x_voc.py'

import runpy as _runpy

globals().update(_runpy.run_path(
    '{{ fileDirname }}/_base_/voc_dataset.py')['build_voc_overrides'](
        '{{ fileDirname }}/../..'))
del _runpy
```

拆开看：

- `_base_` 负责"继承官方模型的全部结构"——backbone、neck、head、train_cfg、test_cfg、lr_config 全都来自官方，一行都不抄；
- `import runpy` + `run_path` 负责"把 `voc_dataset.py` 当脚本跑一遍，取它返回的 dict"；
- `globals().update(...)` 负责"把这个 dict 里的每个键塞进当前模块的命名空间"——于是 `data`、`optimizer`、`evaluation` 这些名字就成了 config 的顶层键，等价于手写；
- `del _runpy` 是防止 `runpy` 模块对象混进 `cfg_dict`（mmcv 会过滤 module 对象，但显式删掉更干净）。

**为什么不用 `_base_` 声明 `voc_dataset.py`？** 因为它不是一个配置，是一个函数——`_base_` 只能合并 dict，不能调用函数。这是 mmcv 1.x 配置系统的一个硬限制，绕过去的标准做法就是这个 `runpy` + `globals().update`。

**顺序有讲究**：`ld_r18_voc.py` 里先写 `model = dict(...)`，后写 `globals().update(...)`。因为 overrides 里不含 `model` 键，所以互不干扰。如果哪天有人往 `build_voc_overrides` 的返回值里加了 `model`，就会静默覆盖掉教师的三个 KD 权重——这是这个设计最脆弱的地方。

### 4.3 `build_voc_overrides()` 逐段

`configs/_base_/voc_dataset.py`，函数体 130 行，可以切成六段：

**(1) 路径与档位（第 9-43 行）**

```python
PROJECT_ROOT = Path(project_root).resolve()          # 传进来的是 '{{ fileDirname }}/../..' → ld_core_experiment 的父目录，即 2022_cvpr_ld
VOC_DATA_ROOT = Path(os.environ.get('VOC_DATA_ROOT', PROJECT_ROOT/'data'/'VOCdevkit'))
DATA_ROOT = str(VOC_DATA_ROOT.resolve()) + os.sep    # 注意补了结尾分隔符
```

`os.sep` 那一下不能省：后面全是 `DATA_ROOT + 'VOC2007/ImageSets/...'` 的字符串拼接，Windows 上 `os.sep` 是 `\`，拼出来是 `...\VOC2007/ImageSets/...`——混用分隔符在 Windows 上能工作，但 `Path` 处理会更干净。这是跨平台折中。

档位表 `RUNTIME_PROFILES`（第 14-19 行）：

| 档位 | 宽×高 | batch | workers |
|---|---|---|---|
| `tiny` | 640×384 | 1 | 0 |
| `low` | 800×480 | 1 | 0 |
| `default` | 1000×600 | 1 | 2 |
| `high` | 1333×800 | 1 | 4 |

`_auto_profile()`（第 22-33 行）读 `torch.cuda.get_device_properties(0).total_memory`：`<12GiB → tiny`、`<20GiB → low`、否则 `default`。**注意 `high` 永远不会被自动选到**——它是手动档。

这里有个隐藏假设：只按**第一块卡**的显存判断，多卡场景下若各卡型号不同会误判。

**(2) 环境变量读取（第 46-58 行）**

```python
def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value is not None else default
```

写法朴素，但注意 `int('')` 会抛 `ValueError`——`LD_PROFILE=` 这种空值不会被当成 unset。`REPEAT_TIMES < 1` 有显式校验，其他几个没有。

**(3) 数据管道（第 66-91 行）**

训练管道：`LoadImageFromFile → LoadAnnotations → Resize(keep_ratio) → RandomFlip(0.5) → Normalize → Pad(size_divisor=32) → DefaultFormatBundle → Collect`。

测试管道用 `MultiScaleFlipAug` 包一层但 `flip=False`、只有一个尺度——等价于单尺度测试，写法沿用官方。

**`Pad(size_divisor=32)` 和五层 FPN 的 `strides=[8,16,32,64,128]` 是绑定的**：输入高宽必须是 32 的倍数，否则最后一层特征图对不齐。四个档位的尺寸都是 32 的倍数，`LD_IMAGE_WIDTH` 手动改的时候要自己保证。

**(4) 学习率（第 94 行）**

```python
optimizer = dict(lr=0.01 * SAMPLES_PER_GPU * GPU_COUNT / 16)
```

mmdet 线性缩放惯例是 `0.02 * bs * n_gpu / 16`（8 卡 × 2 图 = 16 → 基准 lr 0.02）。这里系数是 **0.01**，等于官方缩放的一半。

对照一下：官方 LD VOC 配置 `samples_per_gpu=3, lr=0.00375`，正好是 `0.02*3/16`。而本套代码在 `samples_per_gpu=1` 下给出 `0.000625`，官方等价值是 `0.00125`。

**这不影响四组变体的相对比较**（大家用同一个公式），但影响"和论文/官方脚本对齐"这件事。README 没提这个系数选择，属于隐含偏差，值得记一笔。

**(5) 两种 VOC 布局的自动识别（第 95-104 行）**

```python
VOC2012_TRAIN_ANN = Path(DATA_ROOT)/'VOC2012'/'ImageSets'/'Main'/'trainval.txt'
if VOC2012_TRAIN_ANN.is_file():
    TRAIN_ANN_FILE  = [DATA_ROOT+'VOC2007/.../trainval.txt', str(VOC2012_TRAIN_ANN)]
    TRAIN_IMG_PREFIX= [DATA_ROOT+'VOC2007/', DATA_ROOT+'VOC2012/']
else:
    TRAIN_ANN_FILE  = DATA_ROOT+'VOC2007/ImageSets/Main/trainval.txt'
    TRAIN_IMG_PREFIX= DATA_ROOT+'VOC2007/'
```

判据是"**VOC2012 的 trainval.txt 在不在**"，而不是"VOC2012 目录在不在"。这个选择很稳：OpenI 的合并版可能残留一个空的 `VOC2012/`，但不会有划分文件。

代价：如果某个数据集真的是 VOC07+12 但 `VOC2012/ImageSets/Main/trainval.txt` 恰好缺失，会被静默当成合并版，训练集从 16551 张缩到 5011 张，而日志里只有 dataset 长度变化这一个线索。`check_dataset.py` 会把布局打印出来，跑之前看一眼。

**(6) 其余覆盖项（第 106-137 行）**

```python
data = dict(samples_per_gpu=..., workers_per_gpu=...,
            train=dict(type='RepeatDataset', times=REPEAT_TIMES, dataset=dict(...)),
            val=..., test=...)
evaluation = dict(interval=1, metric='mAP')
if os.environ.get('USE_FP16','0').lower() in ('1','true','yes','on'):
    fp16 = dict(loss_scale='dynamic')
overrides = dict(data=..., optimizer=..., evaluation=..., img_norm_cfg=...,
                 train_pipeline=..., test_pipeline=..., data_root=DATA_ROOT)
```

注意 `overrides` 里还回传了 `img_norm_cfg` / `train_pipeline` / `test_pipeline` 三个**顶层键**。它们其实不被 mmdet 读取（真正生效的是 `data.train.dataset.pipeline`），只是为了和官方 config 的顶层键保持视觉一致。属于"无害的死键"，但会让读 config dump 的人误以为它们起作用。

`fp16` 只在开的时候注入，用的是 `if 'fp16' in locals()` 判断——这种写法依赖函数作用域的 locals，可读性一般但能工作。

### 4.4 环境变量契约表

这是第 2 层和第 3 层之间**唯一的接口**，值得单独记：

| 变量 | 谁写 | 谁读 | 默认值 | 说明 |
|---|---|---|---|---|
| `VOC_DATA_ROOT` | `build_env` | `voc_dataset.py` | `<2022_cvpr_ld>/data/VOCdevkit` | 必须指向含 `VOC2007/` 的目录 |
| `LD_PROFILE` | `build_env` | `voc_dataset.py`、`_common.py` | `auto` | 解析后写回具体档位名 |
| `LD_IMAGE_WIDTH` | `build_env`（或用户） | `voc_dataset.py` | 档位值 | 必须是 32 的倍数 |
| `LD_IMAGE_HEIGHT` | 同上 | 同上 | 档位值 | 同上 |
| `SAMPLES_PER_GPU` | `build_env` | `voc_dataset.py` | 档位值 | 同时决定 lr |
| `WORKERS_PER_GPU` | `build_env` | `voc_dataset.py` | 档位值 | Windows 上建议 0 |
| `GPU_COUNT` | `build_env` | `voc_dataset.py` | 1 | **只用于 lr 缩放** |
| `LD_REPEAT_TIMES` | 用户 | `voc_dataset.py` | 3 | 冒烟测试设 1 |
| `USE_FP16` | `build_env` | `voc_dataset.py` | 未设置 | `1` 时注入 `fp16=dict(loss_scale='dynamic')` |
| `TEACHER_CHECKPOINT` | `build_env` | `ld_r18_voc.py` | `runs/teacher_r101/epoch_16.pth` | 教师权重的绝对路径 |

**只有这 10 个变量**。想加新的超参，必须同时改 `build_env`、`voc_dataset.py`、README 三处——这是"环境变量是唯一通道"这个设计的维护成本。

---

## 5. 驱动脚本层

### 5.1 三个 train 脚本的差异

`train_teacher.py` / `train_baseline.py` / `train_ld.py` 的 argparse
定义大部分相同（`--config`、`--data-root`、`--work-dir`、`--epochs`、
`--samples-per-gpu`、`--workers`、`--profile`、`--gpu-id`、
`--repeat-times`、`--fp16`、`--seed`、`--deterministic`、`--load-from`、
`--resume-from`、`--no-validate`），差异主要在三处：

| | teacher | baseline | ld |
|---|---|---|---|
| 默认 config | `gfl_r101_voc_teacher.py` | `gfl_r18_voc_baseline.py` | 按 `--variant` 选 |
| 默认 work_dir | `runs/teacher_r101` | `runs/student_r18_baseline` | `runs/ld_<variant>` |
| 默认 epochs | 16 | 4 | 4 |
| 教师权重 | 无 | 无 | **强制存在** |

这种"三份几乎一样的 argparse"是刻意复制的：让每个脚本单独看都自解释，不逼读者去翻共享函数。
自 2026-10-28 起三个脚本都默认 `--seed 0`，并可显式启用
`--deterministic`。`--load-from` 用于加载完整检测器 checkpoint，
`--resume-from` 只用于恢复同一训练任务；两者不能同时使用。

### 5.2 `require_ld_checkout()`：补丁即门禁

```python
def require_ld_checkout():
    train_tool = LD_ROOT / 'tools' / 'train.py'
    if not train_tool.is_file():
        raise SystemExit('Official LD submodule is missing. Run `git submodule update --init --recursive`')
    from apply_ld_compat import apply_patch
    if apply_patch(check_only=True) != 'applied':
        raise SystemExit('LD compatibility patch missing. Run python scripts/apply_ld_compat.py.')
    return train_tool
```

**"补丁已打"被提升成了运行前置条件**，这是个很有价值的设计：补丁没打时 LD 在 `import mmdet` 阶段就会因版本断言失败，那个报错和真正的"环境装错了"长得一模一样，很难区分。这里提前 3 秒报错，且直指解法。

注意 `from apply_ld_compat import ...` 写在函数内部——因为 `_common.py` 和 `apply_ld_compat.py` 是同级脚本目录，不在 package 里，靠 `sys.path[0]`（脚本所在目录）互相 import。写成顶层 import 也没问题，放函数里是为了让 `--help` 之类的路径不触发。

### 5.3 `resolve_profile()` 与 `_gpu_memory_gib()`

```python
def resolve_profile(profile=None):
    requested = (profile or os.environ.get('LD_PROFILE', 'auto')).lower()
    if requested != 'auto':
        if requested not in RUNTIME_PROFILES: raise ValueError(...)
        return requested
    memory_gib = _gpu_memory_gib()
    if memory_gib is None or memory_gib >= 20: return 'default'
    if memory_gib >= 12: return 'low'
    return 'tiny'
```

这段和 `voc_dataset.py:22-33` 的 `_auto_profile()` **逻辑重复了**（一份在脚本侧、一份在 config 侧）。原因：脚本要在传环境变量前就知道档位（因为要把 `LD_IMAGE_WIDTH` 之类的具体值写进去），而 config 侧在有人绕过脚本直接调 `tools/train.py` 时也要能工作。两份必须同步改，是一个已知的重复点。

`memory_gib is None` 落到 `default`——没装 torch 或没 GPU 时选最贵档位，这个默认值偏乐观，但反正没 GPU 也训不了。

### 5.4 为什么只支持单卡

`run_train` 里：

```python
if len(normalized_gpu_ids) != 1:
    raise SystemExit('This MMCV runner supports one GPU per process. Select one --gpu-id; '
                     'multi-GPU training requires a separate distributed launcher.')
```

原因在官方 `mmdet/apis/train.py`：非分布式分支是

```python
model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
```

模型只被放到 `cfg.gpu_ids[0]` 上，多出来的 device id 交给 `DataParallel` 散射数据，但 `train_step` 路径上各卡的 loss 不聚合回主卡——这不是一个可用的多卡训练配置（这正是 README 里"MMCV 1.7.2's `MMDataParallel.train_step` requires a single GPU"的依据）。真正的多卡只能走 `--launcher pytorch` 的分布式路径（`init_dist` + `MMDistributedDataParallel`），这套脚本没实现。

所以 `--gpu-ids 0,1` 会被**明确拒绝**，而不是"接受但只用卡 0"——这个选择是对的，静默降级会让人误以为在跑多卡。

副作用：`build_env` 里的 `GPU_COUNT` 在多卡被拒之前就已经参与 lr 计算了。也就是说 `--gpu-ids 0,1` 会先算出 2× 的 lr，然后才被拒绝——不影响正确性（进程没起），但读代码时会觉得别扭。

---

## 6. 蒸馏到底发生在哪：官方代码导读

壳到这里就讲完了。真正决定实验成败的是下面这条链，全在 `LD/` 里。

### 6.1 `KnowledgeDistillationSingleStageDetector`（`LD/mmdet/models/detectors/kd_one_stage.py`）

继承 `SingleStageDetector`，加了四样东西：

**(1) 构造教师（`__init__`，第 39-44 行）**

```python
if isinstance(teacher_config, str):
    teacher_config = mmcv.Config.fromfile(teacher_config)
self.teacher_model = build_detector(teacher_config['model'])
if teacher_ckpt is not None:
    load_checkpoint(self.teacher_model, teacher_ckpt, map_location='cpu')
```

`teacher_config['model']` —— **只取 `model` 键**，教师的数据管道、优化器全部丢弃。这也是为什么教师 config 里那堆 `data`/`optimizer` 覆盖不影响学生训练。但反过来，教师 config 里 `pretrained='torchvision://resnet101'` 会让 `build_detector` 先下载/加载 ImageNet 权重，然后才被 `load_checkpoint` 覆盖——**离线环境下这一步会卡住或报错**，即使你根本不缺教师权重。

**(2) 前向（`forward_train`，第 69-80 行）**

```python
x = self.extract_feat(img)
with torch.no_grad():
    teacher_x  = self.teacher_model.extract_feat(img)
    out_teacher = self.teacher_model.bbox_head(teacher_x)
if not self.output_feature:
    losses = self.bbox_head.forward_train(x, out_teacher, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
else:
    losses = self.bbox_head.forward_train(x, out_teacher, teacher_x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
```

三个要点：

- 教师整条前向在 `no_grad` 里，**显存只吃激活不吃梯度**；
- `out_teacher = teacher.bbox_head(teacher_x)` 返回的是 `tuple(cls_scores), tuple(bbox_preds)`（每层一个张量），不是单个张量；
- `output_feature=True`（我们的 config 里就是 True）时，额外把 `teacher_x`（五层 FPN 特征）也传下去——**这个参数只为 `loss_im` 服务**。

**(3) `__setattr__` 技巧（第 97-107 行）**

```python
def __setattr__(self, name, value):
    if name == 'teacher_model':
        object.__setattr__(self, name, value)   # 绕过 nn.Module.__setattr__
    else:
        super().__setattr__(name, value)
```

这一行让 `teacher_model` **不注册为子模块**。后果：

- `model.parameters()` 不含教师 → 优化器不更新教师（正确）；
- `model.state_dict()` 不含教师 → 学生 checkpoint 只有学生权重（所以 `evaluate_baseline.py` 必须能重新加载教师）；
- `.cuda()` 不会自动搬教师 → 所以重写了 `cuda()`（第 83-87 行）手动搬。

**(4) `train()` 重写（第 89-95 行）**

```python
if self.eval_teacher:
    self.teacher_model.train(False)
super().train(mode)
```

教师永远 eval，BN 用 running stats。这是标准做法，否则教师 BN 统计会跟着学生 batch 抖动。

### 6.2 `LDHead.loss_single()`：蒸馏的核心 30 行

`LD/mmdet/models/dense_heads/ld_head.py:116-282`。先把形状摊平（第 143-160 行），然后：

```python
pos_inds    = ((labels >= 0) & (labels < num_classes)).nonzero().squeeze(1)   # 正样本锚点
remain_inds = (vlr_region > 0).nonzero().squeeze(1)                            # VLR 区域锚点
```

注意 `pos_inds` 和 `remain_inds` **可以重叠**（一个锚点既可以是正样本又在 VLR 里），这是官方实现的行为，不是 bug——两个 loss 作用在不同维度上（cls vs bbox 分布）。

然后三段：

**(a) 正样本段（第 192-245 行）**

```python
loss_bbox = self.loss_bbox(pos_decode_bbox_pred, pos_decode_bbox_targets,
                           weight=weight_targets, avg_factor=1.0)
loss_dfl  = self.loss_dfl(pred_corners, target_corners,
                          weight=..., avg_factor=4.0)
loss_ld   = self.loss_ld(pred_corners, soft_corners,
                          weight=..., avg_factor=4.0)          # ← 定位蒸馏
loss_kd   = self.loss_kd(cls_score[pos_inds], soft_label[pos_inds],
                          weight=label_weights[pos_inds],
                          avg_factor=pos_inds.shape[0])         # ← 分类蒸馏
```

- `pred_corners` 形状 `(N_pos*4, reg_max+1)` = `(N*4, 17)`：每个正样本的 4 条边，每条边一个 17 维离散分布；
- `soft_corners` 是教师对应的 17 维分布 logits；
- `weight_targets` 是**学生自己**的质量分（`cls_score.sigmoid().max(dim=1)`），不是教师的——所以低质量正样本贡献小。

**(b) VLR 段（第 254-274 行）**

```python
loss_ld_vlr = self.loss_ld_vlr(neg_pred_corners, neg_soft_corners,
                               weight=remain_targets[:, None].expand(-1,4).reshape(-1),
                               avg_factor=16.0)
loss_kd_neg = 0 * self.loss_kd(cls_score[remain_inds], ...)
```

`remain_targets` 是 `vlr_region` 的值，也就是该锚点和 GT 的 IoU——**用 IoU 当权重**，IoU 越高的 VLR 锚点学得越多。

`loss_kd_neg` 被 `0 *` 乘掉了，是**死代码**：作者留了个占位，实际不产生梯度。读到这里不用怀疑自己看漏了。

**(c) 无正样本时（第 246-252 行）**

全部 `bbox_pred.sum() * 0`——保住计算图连通的常见写法（保证 `backward` 不会因为某个 loss 不存在而报错）。

**(d) 末尾（第 276-281 行）**

```python
loss_cls = self.loss_cls(cls_score, (labels, score), weight=label_weights, avg_factor=num_total_samples)
```

`loss_cls` 用**学生自己的** GT 监督，和蒸馏无关。所有变体都保留它——这是"四个变体只差蒸馏 loss"的准确含义。

### 6.3 三个 KD loss 的数学

`LD/mmdet/models/losses/kd_loss.py:12-36`：

```python
target  = F.softmax(soft_label / T, dim=1).detach()
kd_loss = F.kl_div(F.log_softmax(pred / T, dim=1), target, reduction='none').mean(1) * (T*T)
```

即标准 Hinton 蒸馏：`KL(σ(s/T) ‖ σ(t/T)) · T²`，`detach()` 保证教师端不回传梯度。

| loss | 作用对象 | 维度 | T | weight | avg_factor | 权重来源 |
|---|---|---|---|---|---|---|
| `loss_kd` | 正样本的分类 logits | 20（VOC 类数） | 2 | 10 | `num_pos` | `label_weights`（全 1） |
| `loss_ld` | 正样本的 bbox 分布 | 4×17 = 68 | 10 | 0.25 | 4.0 | 学生质量分 |
| `loss_ld_vlr` | VLR 锚点的 bbox 分布 | 4×17 = 68 | 10 | 0.25 | 16.0 | VLR 的 IoU 值 |

`T` 的差异是有讲究的：分类分布本身较尖（20 类），`T=2` 就够了；定位分布是 17 个离散 bin，需要 `T=10` 才能把"教师的不确定性"摊开传递。

`avg_factor` 的差异（4.0 vs 16.0）对应"每条边"和"四个角点×4"两种归一化口径，属于历史实现细节。

### 6.4 VLR 区域从哪来：`ATSSAssigner.get_vlr_region()`（论文 Algorithm 1）

`LD/mmdet/core/bbox/assigners/atss_assigner.py:183-298`。这是整套代码里最值得单独读的函数：

```python
overlaps = iou_calculator(bboxes, gt_bboxes)
diou     = iou_calculator(bboxes, gt_bboxes, mode='diou')

# ① 每层取全部锚点作为候选，另外取 topk 个最近邻
candidate_idxs   = 全量锚点（逐层）
candidate_idxs_t = topk 锚点（逐层，topk=9）

# ② 用 topk 的 IoU 算自适应的 μ+σ 阈值
thr = overlaps[candidate_idxs_t].mean(0) + overlaps[candidate_idxs_t].std(0)

# ③ VLR = DIoU 落在 [0.25·thr, thr) 区间的锚点
is_pos = (t_diou < thr) & (t_diou >= 0.25 * thr)

# ④ 只保留"中心在 GT 内"的（这里的实现是把候选索引按 gt_idx 偏移后统一处理）

# ⑤ 返回值是这些锚点与 GT 的 IoU（其余为 0）
vlr_region_iou[max_overlaps != -INF] = max_overlaps[...]
return vlr_region_iou
```

一句话解释：**正样本是"IoU 足够高的锚点"，VLR 是"IoU 不够高、但位置已经很接近"的那一圈锚点**。这圈锚点的定位其实还有救，而且它们数量大（远多于正样本），所以定位蒸馏的信息主要从这里来。

为什么要"有价值"限定？因为全图所有背景锚点做定位蒸馏，等于让模型在完全无关的位置上模仿教师的随机分布，纯噪声。

两个实现细节值得注意：

- 阈值用的是 topk 的 μ+σ，`topk=9` 来自 `train_cfg.assigner.topk`；
- 判据用 **DIoU**、权重用 **IoU**，两套口径混用——读的时候容易看错。

**一个仍需注意的边界问题**：第 203-216 行在 `num_gt == 0` 时返回
`AssignResult` 对象而不是 tensor。当前 `CustomDataset` 默认
`filter_empty_gt=True`，训练前会过滤没有目标类别的图片，所以正常 VOC
配置通常不会触发；但如果以后显式关闭该过滤，调用方后续执行
`unmap(vlr_region, ...)` 时仍会出错。

### 6.5 为什么 `loss_im` 权重是 0 却仍然在算

官方 VOC 配置 `loss_im=dict(type='IMLoss', loss_weight=0)`（`LD/configs/ld/ld_r18_gflv1_r101_fpn_voc_1x.py`）。但 `loss_single` 里第 170-176 行是**无条件执行**的：

```python
if self.imitation_method == 'gibox':
    gi_idx = self.get_gi_region(...)
    loss_im = self.loss_im(gi_student, gi_teacher)
```

`loss_weight=0` 只让结果乘 0，不阻止计算。而 `get_gi_region()`（第 613-638 行）里有一行：

```python
idx_out = torch.ops.torchvision.nms(gibox, giscore, 0.3)[:10]
```

**每个特征层、每个 batch 都会跑一次 NMS**，5 层就是 5 次。在 4 GiB 显存的卡上这是实打实的浪费。

所以本实验实际是"**纯 logit 蒸馏，无特征模仿**"，但代码路径上仍保留了特征模仿的开销。想省这点开销，可以把 `imitation_method` 换掉或直接改 `loss_single`——但那就要动官方代码了，违背这套实验层的设计约束。

---

## 7. 实验变体的差异（精确到行）

继承关系：

```text
官方 ld_r18_gflv1_r101_fpn_voc_1x.py
        └── configs/ld_r18_voc.py            （加教师路径 + 三个 KD 权重）
                ├── ld_r18_voc_full.py       （一行 _base_，原样继承）
                ├── ld_r18_voc_kd_only.py    （loss_ld=0, loss_ld_vlr=0）
                ├── ld_r18_voc_main_ld_only.py（只开 Main LD）
                └── ld_r18_voc_loc_only.py   （loss_kd=0，Main LD + VLR LD）
```

| 变体 | `loss_kd` | `loss_ld` | `loss_ld_vlr` | 回答的问题 |
|---|---|---|---|---|
| baseline（`gfl_r18_voc_baseline.py`） | — | — | — | 学生不用蒸馏能到多少 |
| `kd-only` | 10 (T=2) | 0 | 0 | 只给语义知识，涨不涨 |
| `main-ld-only` | 0 | 0.25 (T=10) | 0 | 对应论文 VOC Table A4 的 Main-region-only |
| `loc-only` | 0 | 0.25 (T=10) | 0.25 (T=10) | 只给定位知识，涨不涨 |
| `full` | 10 (T=2) | 0.25 (T=10) | 0.25 (T=10) | 论文完整方法 |

`full` 和基配置数值完全相同，单独存在是为了让"四个变体"这条命令序列在写法上对称（`--variant full` 而不是"省略参数"）。

**共同前提**：所有变体都保留学生的 GT 检测 loss
（`loss_cls`/`loss_bbox`/`loss_dfl`），并共享分辨率、学习率公式、
随机种子和数据重复设置。`RepeatDataset` 默认仍为 3 次，单轮机制实验
通过 `--repeat-times 1` 改为完整数据一遍。任何一项不一致，消融就失效。

---

## 8. 兼容补丁：10 个文件逐条

`patches/ld-mmcv-1.7.2.patch`，改动分三类：

### A. 版本门禁（1 个文件）

| 文件 | 改动 | 不改会怎样 |
|---|---|---|
| `mmdet/__init__.py` | `assert 1.2.4 <= mmcv <= 1.3` → `legacy or mmcv == '1.7.2'` | mmcv 1.7.2 下 `import mmdet` 直接 AssertionError |

注意它**没有**放宽成"≤1.7.2"，而是 `legacy_mmcv or compat_mmcv`：保留旧区间，额外放行恰好 `1.7.2`，并明确拒绝 2.x。这个写法比 `<=1.7.2` 严格得多——`1.4.0`~`1.7.1` 仍然被拒，因为没人验证过。

### B. NumPy 别名与 API 漂移（6 个文件）

| 文件 | 改动 | 原因 |
|---|---|---|
| `mmdet/apis/train.py` | `get_root_logger(cfg.log_level)` → `get_root_logger(log_level=...)` | mmcv 1.7 把它改成了 keyword-only |
| `mmdet/core/bbox/samplers/iou_balanced_neg_sampler.py` | 3 处 `np.int` → `np.int64` | NumPy ≥1.24 删除 `np.int` |
| `mmdet/core/evaluation/mean_ap.py` | 2 处 `np.bool` → `np.bool_` | 同上 |
| `mmdet/core/mask/structures.py` | 3 处 `np.int`/`np.bool` → `np.int64`/`np.bool_` | 同上（含两处 docstring 里的 doctest） |
| `mmdet/datasets/custom.py` | `.astype(np.int)` → `.astype(np.int64)` | 同上 |
| `mmdet/models/roi_heads/bbox_heads/bbox_head.py` | doctest 里 `np.int` → `np.int64` | 同上 |

`np.int`/`np.bool` 那一类是**同一个根因的六处散布**：官方代码写于 NumPy 1.x 时代，NumPy 2.0 删了这些别名。注意 `requirements.txt` 里 pin 了 `numpy==1.23.5`，理论上不会触发——补丁是**防御性**的，防止有人跳过 requirements 装到 1.24+。

### C. 运行时行为（2 个文件 + 1 个依赖）

| 文件 | 改动 | 原因 |
|---|---|---|
| `mmdet/core/post_processing/bbox_nms.py` | `torch.arange(num_classes, dtype=long)` 加 `device=multi_scores.device` | 老版 torch 允许 CPU/CUDA 索引混用，新版 strict 模式报错 |
| `mmdet/datasets/coco.py` | 删掉 `pycocotools.__version__ >= '12.0.2'` 断言 | `mmpycocotools` 已废弃，维护中的 `pycocotools` 不暴露 `__version__` |
| `requirements/runtime.txt` | `mmpycocotools` → `pycocotools>=2.0.7` | 同上 |

### 补丁的幂等机制（`apply_ld_compat.py`）

这是这个文件最值得学的部分：

```python
def patch_state(ld_root=...):
    def check(reverse=False):
        command = ['git', 'apply', '--check', '--unsafe-paths', '--directory='+root]
        if reverse: command.append('--reverse')
        command.append(str(PATCH_PATH))
        return subprocess.run(command, cwd=str(root), capture_output=True, text=True)

    if check(reverse=True).returncode == 0:   # 能反向应用 → 已打过
        return 'applied'
    if check().returncode == 0:               # 能正向应用 → 没打过
        return 'needed'
    raise RuntimeError('conflicts ... no files changed.\n' + result.stderr)
```

三个状态机：`applied` / `needed` / **冲突则抛异常且一个文件都不改**。

- 用 `--check --reverse` 判"已打"，比在文件里 grep 某个字符串可靠（不会被注释或相似代码骗到）；
- `--directory=<root>` 让 patch 里 `a/mmdet/...` 的路径前缀对上任意 checkout；
- 冲突时先 `--check` 再决定，**绝不先 apply 再回滚**——这是"部分修改"最有效的防御；
- 不需要任何 Python 第三方依赖，所以能在裸环境里跑。

`tests/test_compat.py` 专门验证了这三点：幂等（`test_patch_applies_and_is_idempotent`）、冲突不动文件（`test_conflicting_patch_does_not_modify_other_files`）、版本门禁的边界值（`1.2.3` 拒 / `1.2.4` 收 / `1.3` 收 / `1.7.0` 拒 / `1.7.2` 收 / `1.7.2rc1` 拒 / `2.0.0` 拒）。

它还顺带验证了 `mmcv_wheel_index()` 的 URL 生成：

```python
mmcv_wheel_index('2.0.1+cu118', '11.8')
# → https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
```

注意 `torch2.0.0` 这个细节：OpenMMLab 的 wheel 索引对 PyTorch 的**小版本**不区分，`2.0.1` 和 `2.0.0` 共用 `torch2.0.0` 目录。`2.1.2` 同理落到 `torch2.1.0`。

---

## 9. 自检脚本的分层

这些脚本不是随便加的，它们覆盖了**四种不同的失败模式**：

| 层次 | 脚本 | 验证什么 | 失败时说明 |
|---|---|---|---|
| 源码 | `apply_ld_compat.py --check` | 补丁状态 | 子模块没初始化 / 补丁冲突 |
| 单元 | `tests/test_compat.py` | 补丁逻辑本身 | 补丁脚本有 bug（不需要 GPU/mmcv） |
| 环境 | `check_environment.py` | import + 版本 + **真跑 op** | mmcv/mmdet/torch 版本不对 |
| 数据 | `check_dataset.py` | 划分文件、图片、标注存在 | 数据路径或布局错 |
| 模型 | `check_baseline.py` | 一步 forward+backward+step | 模型/数据/loss 组合跑不通 |
| 形状 | `inspect_gfl_outputs.py` | 五层输出形状 | 用来肉眼确认 bbox 通道 = 4×(reg_max+1) |

`check_environment.py` 有三个设计值得学：

1. **真跑而不是 import**（`check_ops()`，第 12-30 行）：NMS 传三个框、两个完全重叠，断言返回 `[0, 2]`；ROIAlign 做前后向，断言梯度有限且非零。import 成功但 `.so` 和 torch ABI 不匹配的情况太常见了，只 import 测不出来。
2. **校验 `mmdet` 是不是从 bundled checkout 来的**：
   ```python
   if LD_ROOT.resolve() not in Path(mmdet.__file__).resolve().parents:
       errors.append('mmdet is not imported from the bundled LD checkout')
   ```
   防止 pip 装了一个同名 mmdet 把本地 checkout 遮住。
3. **校验注册表**：`HEADS.get('GFLHead')`、`HEADS.get('LDHead')`、`DETECTORS.get('KnowledgeDistillationSingleStageDetector')`。前两个证明官方 head 在，第三个证明 KD 检测器被 import 了（`mmdet/models/detectors/__init__.py` 里漏注册是很常见的坑）。

`check_baseline.py` 最有价值的是它打印的东西：

```python
print('losses:', {key: float(reduce_loss(value).detach()) for key, value in losses.items()})
print('total_loss: ...')
print('peak_memory_mib: ...')
```

一个 health 的 LD 学生，`losses` 里应该同时出现 `loss_cls / loss_bbox / loss_dfl / loss_ld / loss_ld_vlr / loss_kd / loss_kd_neg / loss_im` 八个键。**如果 `loss_ld` 是 0 而 config 里权重不是 0，说明 VLR 区域或正样本没取到**——这是最快的诊断信号。

`run_smoke.sh` 就是把这些串起来：`apply --check → check_environment → check_dataset → check_baseline → （有 TEACHER_CHECKPOINT 时）check_baseline 跑 LD 配置`。没设教师变量时明确打印 "LD step skipped"，不假装成功。

---

## 10. 现状与坑

### 10.1 `runs/` 里的产物不是这条路径跑出来的

`runs/baseline_full_1ep/20261027_163259/` 的日志里有：

```text
mmengine - INFO - ...
MMEngine: 0.10.7
default_hooks = dict(checkpoint=...)
optim_wrapper = dict(type='OptimWrapper')
train_cfg = dict(max_epochs=1, type='EpochBasedTrainLoop')
data_preprocessor = dict(type='DetDataPreprocessor')
```

这些键只存在于 **MMDetection 3.x / MMEngine**。dump 出来的 `runs/*/gfl_r18_voc_baseline.py` 也是 3.x 风格（`data_root` 在 `train_dataloader.dataset` 下面、`param_scheduler` 用 `MultiStepLR`）。

而本套 config 继承的是 MMDetection **2.10** 的 `_base_`，用的是 `optimizer_config`/`lr_config`/`total_epochs`/`pretrained` 这套 2.x 键。

结论：**`runs/` 里的 MMEngine / MMDetection 3 目录是另一条技术栈的遗留物**，`LDHead` 的蒸馏 loss 在其中从未被调用。里面的低 mAP 是早期 debug 结果，不能与 MMDetection 2 的单轮消融混用。

### 10.2 本机环境与云端实验环境

| 项 | 实验层要求 | 本机 `DL` 环境实测 |
|---|---|---|
| OS | Linux x86_64（`install_environment.py:44-45` 硬性拒绝其他系统） | Windows |
| Python | 3.8–3.10 | 3.10.22 ✅ |
| torch | 2.0.x / 2.1.x | 2.3.1+cu121 ❌ |
| torchvision | 0.15.x / 0.16.x | 0.18.1 ❌ |
| mmcv-full | 1.7.2 | **mmcv 2.2.0** ❌ |
| mmdet | 2.10.x（bundled） | **mmdet 3.3.0** ❌ |
| GPU | ≥12 GiB 建议 | 本机 RTX 3050 Laptop **4 GiB** ❌；云端 V100S 32 GiB ✅ |

已完成机制实验的云端实际环境为 Python 3.8.18、
PyTorch 2.0.0+cu118、torchvision 0.15.1+cu118、
`mmcv-full==1.7.2`、MMDetection 2.10.0+4287990 和
Tesla V100S-PCIE-32GB。实测 student `640x384`、batch 4、FP16
可以正常运行。

`check_environment.py` 会在第一次比较版本时就把这些全列出来退出（`errors` 列表一次性收集，不会只报第一条）。

所以要在本机跑通，只有三条路：
1. 租 Linux + 大显存 GPU（原设计路径）；
2. WSL2 里装 Python 3.9 + torch 2.0.1+cu118 + mmcv-full 1.7.2（4 GiB 显存仍只够 `tiny` 档，教师 R101 前向会非常吃紧）；
3. 放弃这套壳，用现有的 MMDet 3 栈重写 LD（等于放弃"不改官方代码"这条约束）。

### 10.3 已知的小坑清单

| 坑 | 现象 | 位置 |
|---|---|---|
| 教师权重必须存在 | `train_ld.py` 立刻 `SystemExit` | `train_ld.py:31-32` |
| 改 epochs 现在会同步改 lr 调度 | 衰减点取 `ceil(0.75 * epochs)` | `_common.py` 的 `schedule_decay_epoch` |
| lr 系数是官方一半 | 绝对值偏低（不影响四组相对比较） | `voc_dataset.py:94` |
| 空标注图片会崩 | `get_vlr_region` 返回 `AssignResult` 而非 tensor | `atss_assigner.py:203-216` |
| `loss_im` 权重 0 但仍在算 | 每层一次 NMS 的开销 | `ld_head.py:170-176`、`:637` |
| 教师 config 的 `pretrained` 仍会加载 | 离线环境卡在下载 torchvision 权重 | `LD/configs/gfl/gfl_r101_fpn_voc.py` |
| 顶层 `img_norm_cfg`/`train_pipeline` 是死键 | 改了不生效 | `voc_dataset.py:132-134` |
| 多卡被拒但 lr 已按多卡算 | 无害但读起来别扭 | `train_ld.py:36-39` → `_common.py:113-116` |
| 合并版数据集识别是"猜"的 | 布局误判时训练集从 16551 掉到 5011 且不报错 | `voc_dataset.py:95-104` |
| `{{ fileDirname }}` 只认 mmcv | 换任何非 mmcv 工具读 config 都失效 | 全部 config |

---

## 11. 学习路径

### 11.1 建议的阅读顺序

1. `README.md`——先建立"这是什么、在什么机器上跑"的印象；
2. `configs/_base_/voc_dataset.py`——整套实验的超参全在这里，读懂它等于读懂实验设计；
3. `scripts/_common.py`——四条路径常量和 `build_env`/`run_train`，比单个 train 脚本信息密度高；
4. `scripts/train_ld.py`——最短的完整例子（45 行）；
5. `configs/ld_r18_voc.py` → `ld_r18_voc_kd_only.py`——看三个 loss 权重怎么被覆盖；
6. `LD/mmdet/models/detectors/kd_one_stage.py`——教师怎么被挂上去的；
7. `LD/mmdet/models/dense_heads/ld_head.py` 的 `loss_single`（第 116-282 行）——蒸馏核心；
8. `LD/mmdet/core/bbox/assigners/atss_assigner.py` 的 `get_vlr_region`（第 183-298 行）——论文 Algorithm 1；
9. `patches/ld-mmcv-1.7.2.patch` + `scripts/apply_ld_compat.py`——最后看，它是工程适配不是算法。

### 11.2 三个动手验证（不需要 GPU）

```bash
# ① 补丁状态机：先看状态，再跑测试
python 2022_cvpr_ld/ld_core_experiment/scripts/apply_ld_compat.py --check
python -m unittest discover -s 2022_cvpr_ld/ld_core_experiment/tests -v

# ② 数据集布局识别：打印的是 split VOC07+12 还是 merged
python 2022_cvpr_ld/ld_core_experiment/scripts/check_dataset.py \
  --data-root 2022_cvpr_ld/data

# ③ 只解析 config，不建模型：把环境变量伪造好，用 mmcv 打印 pretty_text
```

③ 的写法（需要 mmcv 1.7.2 环境）：

```python
import os
os.environ['VOC_DATA_ROOT'] = r'D:\...\2022_cvpr_ld\data'
os.environ['LD_PROFILE'] = 'tiny'
from mmcv import Config
cfg = Config.fromfile(r'...\configs\ld_r18_voc_full.py')
print(cfg.pretty_text)   # 检查 data_root / img_scale / lr / 三个 loss weight
```

这一步能一次性验证 `{{ fileDirname }}`、`runpy` 覆盖、`_base_` 递归合并三件事，而且**不需要 GPU**。是排查"配置到底合成了什么"最快的办法。

### 11.3 自测题（能答上来就算真的懂了）

1. `build_env()` 里的 `VOC_DATA_ROOT` 和 `voc_dataset.py` 里读的 `VOC_DATA_ROOT`，哪个优先？如果两个都设了会怎样？
2. 为什么 `ld_r18_voc.py` 里 `model = dict(...)` 必须写在 `globals().update(...)` 之前？反过来会出什么事？
3. 把 `--epochs` 从 4 改成 8，学习率调度会变成什么样？为什么？
4. `loss_ld` 的 `weight` 用的是教师的质量分还是学生的？为什么这样设计？
5. `get_vlr_region` 里为什么"选候选"用 DIoU、"算权重"用 IoU？
6. `teacher_model` 不注册为子模块，会带来哪四个后果？
7. 补丁的 `--check --reverse` 为什么比 grep 某个字符串更可靠？
8. 如果 `loss_ld` 在 `check_baseline.py` 的输出里恒为 0，最可能的原因是什么？
9. 补丁一共改 10 个文件，其中 NumPy 别名那类改动有 10 处却只分布在 5 个文件里——剩下 5 个文件各在解决什么问题？为什么其中一个改了还会"不影响运行"？
10. 四个变体里，`full` 和基配置数值完全一样，为什么还要单独存在一个文件？

---

## 附：一句话记住每个设计决策

| 决策 | 一句话理由 |
|---|---|
| 不 fork 官方代码，用 patch | 官方仓库能继续 `git pull`，补丁由父仓库跟踪，云端 clone 可复现 |
| 锁死 mmcv 1.7.2 + mmdet 2.10 | LD 的 `LDHead`、`mmcv.runner`、`mmcv.parallel` 在 3.x 里不存在了 |
| 环境变量而非命令行传超参 | 官方 `tools/train.py` 只认 `--cfg-options`，用它改 10 个嵌套键会不可读 |
| `{{ fileDirname }}` 而非相对路径 | config 被 mmcv 拷到临时目录执行，相对路径会错 |
| `runpy` + `globals().update` | mmcv 的 `_base_` 只能合并 dict，不能调用函数 |
| 补丁状态做成"已打/没打/冲突"三态 | 冲突时一个文件都不改，比 apply 后回滚安全 |
| 拒绝静默编译 mmcv | 编译失败会留下半装的包，比"装不上"更难查 |
| 脚本强制单卡 | 非分布式 `MMDataParallel` 不聚合多卡 loss，静默降级会骗人 |
| 教师前向包在 `no_grad` 里 | 显存只吃激活，不吃梯度 |
| `teacher_model` 不注册为子模块 | 不更新、不进 checkpoint、不进 `parameters()`，但要手动 `.cuda()` |
| VOC 上做消融而非 COCO | 单卡可跑，类别少收敛快，足以看方向性 |
| 四个变体共享一切非蒸馏超参 | 否则消融不成立 |
