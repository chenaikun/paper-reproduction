# GFL 源码与论文来源

## 论文信息
- 标题：Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection
- 作者：Xiang Li, Wenhai Wang, Lijun Wu, Shuo Chen, Xiaolin Hu, Jun Li, Jinhui Tang, Jian Yang
- 会议：NeurIPS 2020
- arXiv：https://arxiv.org/abs/2006.04388

## 上游代码
- MMDetection：https://github.com/open-mmlab/mmdetection
- 本地路径：./mmdetection
- 使用分支或标签：
- Git commit：cfd5d3a985b0249de009b67d04f37263e11cdf3d
- 获取日期：

## 使用配置
- 配置文件：
  `configs/gfl/gfl_r50_fpn_1x_coco.py`
- 自己是否修改配置：是/否
- 修改后的配置位置：
- 修改内容：

## 相关源码位置
- GFL 检测头：
  `mmdet/models/dense_heads/gfl_head.py`
- GFL 损失：
  `mmdet/models/losses/gfocal_loss.py`
- 配置目录：
  `mmdet/configs/gfl/`

## 自定义代码
- 位置：`custom_modules/`
- 是否修改 MMDetection 原始源码：否/是
- 如果修改，写明文件和原因：

## 代码状态
- 上游代码是否原样使用：
- 是否使用官方 checkpoint：
- 是否有自己的 patch：