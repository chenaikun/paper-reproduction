# LD 单轮 VOC 机制验证实验

## 1. 实验目标

本次实验不是为了复现论文在 COCO 上的绝对 AP，而是用较小的训练成本验证
Localization Distillation（LD）的核心机制：

1. teacher 能否被正确加载，并向 student 提供知识；
2. `loss_kd`、`loss_ld` 和 `loss_ld_vlr` 能否正确参与前向与反向传播；
3. classification KD、localization distillation 及其组合分别产生什么影响；
4. 结果趋势是否与论文“LD 主要改善高 IoU 定位质量”的结论一致。

## 2. 实验环境

| 项目 | 设置 |
|---|---|
| GPU | Tesla V100S-PCIE-32GB |
| Python | 3.8.18 |
| PyTorch | 2.0.0+cu118 |
| torchvision | 0.15.1+cu118 |
| CUDA / NVCC | 11.8 |
| MMCV | `mmcv-full==1.7.2` |
| 检测框架 | 仓库内的 LD / MMDetection 2.10.0+4287990 |
| Teacher | GFL + ResNet-101 |
| Student | GFL + ResNet-18 |
| 训练集 | VOC2007 trainval + VOC2012 trainval，共 16,551 张 |
| 测试集 | VOC2007 test，共 4,952 张 |

Teacher 使用默认 profile（1000 x 600、batch 1、数据重复 3 次）训练
1 个 epoch，共 49,653 iterations，checkpoint 为：

```text
/cloud/cloud-ssd1/ld-runs/teacher_r101/epoch_1.pth
```

本次 teacher 训练日志中的 `load_from = None`，backbone 使用
`torchvision://resnet101` 初始化；当时下载的 GFL COCO 权重没有传入这次
teacher 训练。后续版本已增加 `--load-from` 参数，可在重新训练时显式加载
完整检测器 checkpoint。

该 checkpoint 在第一次训练后验证时，因为 NMS 中类别标签与预测张量的设备
不一致而中断；训练检查点本身已经正常保存。兼容补丁将类别标签放到与原预测
相同的设备后，checkpoint 可以正常评估和用于蒸馏。

## 3. 学生训练协议

四组 student 实验保持相同设置：

| 项目 | 设置 |
|---|---|
| 输入尺寸 | 640 x 384（`tiny`） |
| Batch size | 4 |
| Epoch | 1 |
| 数据重复 | `LD_REPEAT_TIMES=1`，每轮遍历全部 16,551 张训练图片 |
| 混合精度 | 动态 loss-scale FP16 |
| 训练中验证 | 关闭，训练完成后统一下载 checkpoint 评估 |
| 学生初始化 | `torchvision://resnet18` |
| Teacher 推理 | `torch.no_grad()`，参数冻结 |

Teacher 与 student 的训练分辨率和数据曝光并不完全相同：teacher 是
1000 x 600、batch 1、每轮重复数据 3 次；student 是 640 x 384、batch 4、
每轮完整数据一遍。四组 student 之间是受控比较，teacher 与 student 的
绝对指标只用于确认 teacher 更强，不能视为严格匹配的容量消融。

学生损失组合如下：

| 实验 | GFL 检测损失 | `loss_kd` | `loss_ld` | `loss_ld_vlr` |
|---|---:|---:|---:|---:|
| Student Baseline | 是 | 0 | 0 | 0 |
| KD-only | 是 | 是 | 0 | 0 |
| Loc-only | 是 | 0 | 是 | 是 |
| Full LD | 是 | 是 | 是 | 是 |

其中：

- Main KD 和 Main LD 在标签分配得到的主要正样本区域计算；
- VLR LD 在 Valuable Localization Region 计算；
- 当前配置关闭 VLR KD，因此 `loss_kd_neg` 为 0；
- 当前实验不使用 feature imitation，因此 `loss_im` 为 0。

## 4. Teacher 与学生结果

### 4.1 总体指标

| 模型 | AP | AP50 | AP70 | AP75 |
|---|---:|---:|---:|---:|
| ResNet-101 Teacher | 0.3375 | 0.5361 | 0.4088 | 0.3505 |
| R18 Student Baseline | 0.2404 | 0.4153 | 0.2917 | 0.2354 |
| R18 + KD-only | 0.2302 | 0.3933 | 0.2774 | 0.2194 |
| R18 + Loc-only | **0.2828** | **0.4636** | **0.3469** | **0.2921** |
| R18 + Full LD | 0.2666 | 0.4384 | 0.3252 | 0.2648 |

这里的 `AP` 是 IoU 阈值 0.50 到 0.95、步长 0.05 的平均 AP。VOC 评估接口
最后返回的 `mAP` 数值实际对应 AP50，因此报告和比较统一使用 AP50 这一列。

### 4.2 相对 Student Baseline 的变化

| 实验 | ΔAP | ΔAP50 | ΔAP70 | ΔAP75 |
|---|---:|---:|---:|---:|
| KD-only | -0.0103 | -0.0219 | -0.0143 | -0.0159 |
| Loc-only | **+0.0423** | **+0.0484** | **+0.0552** | **+0.0567** |
| Full LD | +0.0261 | +0.0232 | +0.0335 | +0.0294 |

以百分点表示，Loc-only 相对 baseline 的 AP、AP50、AP70 和 AP75 分别提升
约 4.23、4.84、5.52 和 5.67 个百分点。

### 4.3 高 IoU 指标

| 实验 | AP75 | AP90 | AP95 |
|---|---:|---:|---:|
| Student Baseline | 0.2354 | 0.0573 | 0.0227 |
| KD-only | 0.2194 | 0.0615 | 0.0234 |
| Loc-only | **0.2921** | 0.0756 | 0.0217 |
| Full LD | 0.2648 | **0.0811** | **0.0416** |

Loc-only 在 AP75 上表现最好。Full LD 的总体 AP75 低于 Loc-only，但在 AP90
和 AP95 上更高。由于本轮只有一个随机种子，这一极严格阈值上的差异只能作为
观察，不能形成稳健结论。

## 5. 蒸馏损失检查

训练前执行了带 teacher 的完整单步检查。

Student Baseline 单步损失：

```text
loss_cls  = 0.3744
loss_bbox = 1.1341
loss_dfl  = 0.6985
total     = 2.2071
```

Full LD 单步损失：

```text
loss_cls    = 0.0299
loss_bbox   = 1.6753
loss_dfl    = 0.7499
loss_ld     = 0.0147
loss_ld_vlr = 0.2163
loss_kd     = 0.4540
total       = 3.1402
```

这些结果说明：

- teacher checkpoint 能正常加载；
- 三项蒸馏损失均为有限值且非零；
- 完整 LD 可以完成前向、反向和优化器更新；
- Student Baseline 不包含任何蒸馏损失，符合消融定义。

## 6. 实验结论

### 6.1 支持论文核心机制的部分

1. Teacher 在 AP、AP50、AP70 和 AP75 上均高于 student，满足知识蒸馏需要
   更强 teacher 的前提。
2. Loc-only 明显优于 Student Baseline，AP75 的提升最大，说明当前 teacher
   的定位分布包含学生能够利用的信息。
3. Full LD 仍然优于 Student Baseline，说明同时启用分类和定位蒸馏不会破坏
   学生的基本检测能力。
4. 结果方向与论文“LD 更直接改善定位质量，并主要在高 IoU 阈值上获益”的
   结论一致。

### 6.2 与论文趋势不同的部分

KD-only 在本轮设置下使 AP、AP50、AP70 和 AP75 全部下降。Full LD 也不如
Loc-only。一个合理解释是：

- teacher 只训练了 1 个 epoch，和充分收敛的论文 teacher 不同；
- 当前沿用了论文的 `loss_kd` 权重 10，对短训练 student 形成了过强约束；
- teacher 与 student 能力差距较大，分类软标签不一定适合当前欠训练阶段。

因此，本实验支持定位蒸馏是更直接有效的监督，但不能据此否定论文中充分训练
后 Full LD 的效果。

## 7. 与论文补充材料 Table A4 的关系

论文 VOC 补充实验使用：

- VOC07+12 训练集和 VOC2007 test；
- ResNet-18 student；
- 4 个 epoch；
- 1333 x 800 输入；
- 只在 Main distillation region 上使用 LD。

论文 Table A4 中 ResNet-101 teacher 的结果为：

| 设置 | AP | AP50 | AP70 | AP90 |
|---|---:|---:|---:|---:|
| R18 baseline | 51.8 | 75.8 | 62.9 | 20.6 |
| R18 + LD | 53.0 | 75.9 | 64.0 | 22.7 |

论文中 AP50 基本不变，而 AP70 和 AP90 提升，进一步说明 LD 主要改善严格
定位阈值下的表现。

本实验没有直接复现 Table A4：

| 项目 | 论文 Table A4 | 本次实验 |
|---|---|---|
| Epoch | 4 | 1 |
| 输入 | 1333 x 800 | 640 x 384 |
| 数据重复 | 官方 VOC 协议 | 每轮完整数据一遍 |
| Teacher 收敛程度 | 充分训练 | 1 epoch |
| LD 区域 | Main only | Main + VLR |
| 硬件与并行 | 论文训练协议 | 单卡 V100S |

因此两边的绝对数值不能直接比较，只能比较变化方向。仓库已经增加
`configs/ld_r18_voc_main_ld_only.py`，但本次实验尚未运行该配置；
现有 `loc-only` 同时启用了 `loss_ld` 和 `loss_ld_vlr`。

## 8. 局限性

- 只训练 1 个 epoch，student 仍处于明显欠训练状态；
- 原四组训练发生在 seed 参数加入之前，日志中没有显式随机种子；
- 只完成一次训练，不能估计训练方差；
- 使用较小输入和单卡 batch size 4，与论文 VOC 协议不同；
- teacher 只训练 1 个 epoch，不是论文中充分训练的 teacher；
- teacher 与 student 使用了不同分辨率和数据重复设置；
- 没有运行 Main-LD-only，因此尚未单独量化 VLR 的贡献；
- 当前数据规模较小，AP90、AP95 等严格阈值结果容易受个别样本影响。

## 9. 建议的后续实验

如果继续研究，优先顺序如下：

1. 运行已经添加的 Main-LD-only 配置：

   ```text
   loss_kd     = 0
   loss_ld     = 0.25
   loss_ld_vlr = 0
   ```

   该设置可直接对应论文 Table A4，并与本次 Loc-only 比较 VLR 的增量效果。

2. 对 Baseline、Main-LD-only、Loc-only 和 Full LD 使用多个随机种子，
   报告均值与标准差。

3. 如果计算预算允许，按论文 VOC 协议训练 4 个 epoch，输入提高到
   1333 x 800，并恢复每轮三遍数据重复。

4. 在不使用测试集调参的前提下，对 `loss_kd` 权重做预定义的小范围敏感性
   实验，判断本轮 KD-only 下降是否来自权重过大。

## 10. 实验产物

云端结果目录：

```text
/cloud/cloud-ssd1/ld-runs/
```

已下载到本地的工作区根目录：

```text
ld-weights.tar.gz
ld-logs.tar.gz
```

后续版本会默认使用 `--seed 0`，并通过 `--deterministic` 启用
确定性选项。新评估会自动把 stdout 保存到 checkpoint 同目录下的
`<config>_eval.log`；本次早期评估的终端输出没有单独保存为原始日志。

仓库还加入了
`analysis/analyze_output_distances.py`，可在同一输入上统计学生和教师
每层分类软标签 L1 以及四条边的定位分布 L1，并输出 JSON。该分析可在
后续复评时直接用于补强机制证据。

本次实验的核心结论是：

> 在单 epoch、小分辨率、单随机种子的低资源设置下，定位蒸馏显著改善了
> student 的 AP70 和 AP75；分类蒸馏单独使用会带来负迁移，并与定位蒸馏
> 组合时抵消部分收益。结果支持 LD 传递定位知识的核心机制，但不构成对
> 论文正式 COCO 或 VOC 数值的复现。
