"""Measure student-teacher output distances for an LD checkpoint.

This is a mechanism diagnostic, not a replacement for VOC evaluation. It uses
one deterministic synthetic input so it can run quickly on any supported GPU.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from _common import CONFIG_ROOT, build_env  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=CONFIG_ROOT / 'ld_r18_voc_full.py')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--teacher-checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--profile',
                        choices=['auto', 'tiny', 'low', 'default', 'high'],
                        default='tiny')
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, default=None)
    return parser.parse_args()


def distribution_distance(student, teacher, bins=None):
    if bins is not None:
        batch, _, height, width = student.shape
        channels = student.shape[1] // bins
        student = student.reshape(batch, channels, bins, height, width)
        teacher = teacher.reshape(batch, channels, bins, height, width)
    student_prob = torch.softmax(student.float(), dim=2 if bins else 1)
    teacher_prob = torch.softmax(teacher.float(), dim=2 if bins else 1)
    return float((student_prob - teacher_prob).abs().mean().item())


def main():
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit('Student checkpoint not found: {}'.format(checkpoint))
    if not teacher_checkpoint.is_file():
        raise SystemExit('Teacher checkpoint not found: {}'.format(teacher_checkpoint))
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for output-distance analysis.')

    env = build_env(
        args.data_root,
        teacher_checkpoint,
        profile=args.profile,
        gpu_count=1,
    )
    os.environ.update(env)
    cfg = Config.fromfile(str(args.config.resolve()))
    model = build_detector(cfg.model).to(args.device).eval()
    load_checkpoint(model, str(checkpoint), map_location='cpu')
    if not hasattr(model, 'teacher_model'):
        raise SystemExit('The config must build a knowledge-distillation detector.')

    image = torch.zeros(
        1, 3, args.height, args.width, device=args.device)
    with torch.no_grad():
        student_features = model.extract_feat(image)
        student_cls, student_bbox = model.bbox_head(student_features)
        teacher_features = model.teacher_model.extract_feat(image)
        teacher_cls, teacher_bbox = model.teacher_model.bbox_head(teacher_features)

    reg_max = int(model.bbox_head.reg_max)
    levels = []
    for level, (s_cls, t_cls, s_bbox, t_bbox) in enumerate(
            zip(student_cls, teacher_cls, student_bbox, teacher_bbox)):
        levels.append({
            'level': level,
            'classification_l1': distribution_distance(s_cls, t_cls),
            'localization_distribution_l1': distribution_distance(
                s_bbox, t_bbox, bins=reg_max + 1),
        })

    result = {
        'config': str(args.config.resolve()),
        'checkpoint': str(checkpoint),
        'teacher_checkpoint': str(teacher_checkpoint),
        'input': [1, 3, args.height, args.width],
        'reg_max': reg_max,
        'levels': levels,
        'mean_classification_l1': sum(
            item['classification_l1'] for item in levels) / len(levels),
        'mean_localization_distribution_l1': sum(
            item['localization_distribution_l1'] for item in levels) / len(levels),
    }
    output = args.output or checkpoint.parent / 'output_distances.json'
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print('saved: {}'.format(output))


if __name__ == '__main__':
    main()
