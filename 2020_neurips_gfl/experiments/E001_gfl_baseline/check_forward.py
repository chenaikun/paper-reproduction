from pathlib import Path
import sys

import torch
from mmengine.config import Config

#resolve()转换为绝对路径    parnets[2] 即是向上回溯两级
PROJECT_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT_ROOT/"mmdetection"))

# 从注册器中引入构建模型的工厂类 MODELS
from mmdet.registry import MODELS
# 引入全模块注册函数（激活 MMDetection 所有的 Backbone、Neck、Head 类）
from mmdet.utils import register_all_modules


def main():
    register_all_modules(init_default_scope=True)

    config_path=(
        PROJECT_ROOT
        / "mmdetection"
        / "configs"
        / "gfl"
        / "gfl_r50_fpn_1x_coco.py"
    )

    print(f"{config_path}\n")
    print(f"{str(config_path)}\n")
    cfg = Config.fromfile(str(config_path))
    model = MODELS.build(cfg.model)
    model.eval()

    #模拟FPN的五个输出层
    features= (
        torch.randn(1,256,80,80),
        torch.randn(1,256,40,40),
        torch.randn(1,256,20,20),
        torch.randn(1,256,10,10),
        torch.randn(1,256,5,5),
    )

    with torch.no_grad():
        cls_scores,bbox_preds=model.bbox_head(features)

    print(f"模型类型：{type(model).__name__}")
    print(f"检测头类型：{type(model.bbox_head).__name__}")
    print(f"reg_max:{model.bbox_head.reg_max}")

    for level, (cls_score,bbox_pred) in enumerate(
        zip(cls_scores,bbox_preds)
    ):
        print(
            f"level{level}:"
            f"cls={tuple(cls_score.shape)},"
            f"bbox={tuple(bbox_pred.shape)}"
        )



if __name__ == "__main__":
    main()
