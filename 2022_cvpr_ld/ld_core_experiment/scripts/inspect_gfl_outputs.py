"""Print the tensor shapes used by GFL and localization distribution distillation."""

from pathlib import Path

import torch
from mmengine.config import Config

from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs' / 'gfl_r18_voc_baseline.py'


def main():
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(CONFIG))
    model = MODELS.build(cfg.model).cuda().eval()
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    sample = dataset[0]
    batch = model.data_preprocessor(
        {'inputs': [sample['inputs']], 'data_samples': [sample['data_samples']]},
        training=False,
    )

    with torch.no_grad():
        feats = model.extract_feat(batch['inputs'])
        cls_scores, bbox_preds = model.bbox_head(feats)

    print(f"input: {tuple(batch['inputs'].shape)}")
    print(f"feature levels: {len(feats)}")
    for level, (feat, cls, bbox) in enumerate(zip(feats, cls_scores, bbox_preds)):
        print(
            f"level {level}: feature={tuple(feat.shape)}, "
            f"cls_logits={tuple(cls.shape)}, bbox_logits={tuple(bbox.shape)}"
        )
    print('bbox channels = 4 * (reg_max + 1); each edge has a discrete distribution.')


if __name__ == '__main__':
    main()
