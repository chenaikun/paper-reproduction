import argparse
from pathlib import Path

import torch
from mmcv import Config
from mmdet.models import build_detector

from _common import BASELINE_CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=BASELINE_CONFIG)
    parser.add_argument('--height', type=int, default=600)
    parser.add_argument('--width', type=int, default=1000)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    cfg = Config.fromfile(str(args.config.resolve()))
    model = build_detector(cfg.model).to(args.device).eval()
    image = torch.zeros(1, 3, args.height, args.width, device=args.device)
    with torch.no_grad():
        features = model.extract_feat(image)
        cls_scores, bbox_preds = model.bbox_head(features)

    print('input: {}'.format(tuple(image.shape)))
    for level, (feature, cls_score, bbox_pred) in enumerate(
            zip(features, cls_scores, bbox_preds)):
        print('level {}: feature={}, cls_logits={}, bbox_logits={}'.format(
            level, tuple(feature.shape), tuple(cls_score.shape),
            tuple(bbox_pred.shape)))
    print('bbox channels = 4 * (reg_max + 1); each edge is a discrete distribution.')


if __name__ == '__main__':
    main()
