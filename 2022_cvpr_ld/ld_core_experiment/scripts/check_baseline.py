"""Run one real MMDetection 2 training step on the configured detector."""

import argparse
import math
import os
from pathlib import Path

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from _common import BASELINE_CONFIG, build_env


def reduce_loss(value):
    if isinstance(value, (list, tuple)):
        return sum(item.mean() for item in value)
    return value.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=BASELINE_CONFIG)
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--teacher-checkpoint', type=Path, default=None)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--samples-per-gpu', type=int, default=None)
    parser.add_argument('--profile', choices=['auto', 'tiny', 'low', 'default', 'high'],
                        default=None)
    parser.add_argument('--fp16', action='store_true',
                        help='enable dynamic-loss-scale mixed precision')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for the GPU training-step check.')

    env = build_env(
        args.data_root,
        args.teacher_checkpoint,
        args.samples_per_gpu,
        args.workers,
        profile=args.profile,
        gpu_count=1,
        fp16=args.fp16)
    for name in ('VOC_DATA_ROOT', 'LD_PROFILE', 'LD_IMAGE_WIDTH',
                 'LD_IMAGE_HEIGHT', 'SAMPLES_PER_GPU', 'WORKERS_PER_GPU',
                 'GPU_COUNT', 'USE_FP16', 'TEACHER_CHECKPOINT'):
        if name in env:
            os.environ[name] = env[name]

    cfg = Config.fromfile(str(args.config.resolve()))
    dataset = build_dataset(cfg.data.train)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=True)
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.train()
    model = MMDataParallel(model.cuda(args.gpu_id), device_ids=[args.gpu_id])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.optimizer.lr,
        momentum=cfg.optimizer.get('momentum', 0.9),
        weight_decay=cfg.optimizer.get('weight_decay', 0.0001))

    torch.cuda.reset_peak_memory_stats(args.gpu_id)
    batch = next(iter(dataloader))
    optimizer.zero_grad()
    losses = model(return_loss=True, **batch)
    total_loss = sum(
        reduce_loss(value) for key, value in losses.items() if 'loss' in key)
    if not math.isfinite(float(total_loss.detach())):
        raise SystemExit('Non-finite training loss: {}'.format(total_loss.item()))
    total_loss.backward()
    optimizer.step()
    torch.cuda.synchronize(args.gpu_id)

    print('dataset_size: {}'.format(len(dataset)))
    print('losses: {}'.format({
        key: float(reduce_loss(value).detach()) for key, value in losses.items()
    }))
    print('total_loss: {:.6f}'.format(float(total_loss.detach())))
    print('peak_memory_mib: {:.1f}'.format(
        torch.cuda.max_memory_allocated(args.gpu_id) / 1024 ** 2))


if __name__ == '__main__':
    main()
