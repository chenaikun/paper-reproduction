# GFL 复现项目

## 论文
- 标题：Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection
- 会议：NeurIPS 2020
- 论文链接：https://arxiv.org/abs/2006.04388

## 复现目标
1. 理解 GFL 的 QFL、DFL 和分布式边界框表示；
2. 找到论文公式对应的 MMDetection 源码；
3. 跑通原版 GFL 训练、验证和推理；
4. 独立实现并注册一个自定义模块。

## 当前进度
- [ ] 环境安装完成
- [ ] 版本和 commit 已记录
- [ ] GFL 配置解析成功
- [ ] 模型构建成功
- [ ] 单图推理成功
- [ ] 小规模训练成功
- [ ] checkpoint 恢复成功
- [ ] QFL/DFL 源码理解完成
- [ ] 自定义模块注册成功
- [ ] 原版与改进版对照完成

## 目录说明
- `source.md`：论文、仓库和源码版本
- `environment.md`：运行环境和依赖版本
- `notes.md`：论文理解和论文—代码对应
- `configs/`：本项目使用的配置
- `experiments/`：每次实验记录
- `custom_modules/`：自己的模块

## 当前结论
填写已经确认的事实。

## 当前问题
填写还没有解决的问题。

## 下一步
填写接下来准备做的 1—3 件事。
