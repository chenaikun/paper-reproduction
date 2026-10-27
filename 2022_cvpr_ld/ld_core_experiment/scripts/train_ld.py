import argparse
from pathlib import Path

from _common import (DEFAULT_TEACHER_CHECKPOINT, LD_CONFIGS, RUNS_ROOT,
                    build_env, run_train)


def main():
    parser = argparse.ArgumentParser(description='Train an LD VOC ablation.')
    parser.add_argument('--variant', choices=sorted(LD_CONFIGS), default='full')
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--teacher-checkpoint', type=Path,
                        default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument('--work-dir', type=Path, default=None)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--samples-per-gpu', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--profile', choices=['auto', 'tiny', 'low', 'default', 'high'],
                        default=None)
    parser.add_argument('--gpu-ids', default=None,
                        help='comma-separated GPU ids, for example 0,1')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--fp16', action='store_true',
                        help='enable dynamic-loss-scale mixed precision')
    parser.add_argument('--resume-from', type=Path, default=None)
    parser.add_argument('--no-validate', action='store_true')
    args = parser.parse_args()

    teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()
    if not teacher_checkpoint.is_file():
        raise SystemExit('Teacher checkpoint not found: {}'.format(teacher_checkpoint))
    config = args.config or LD_CONFIGS[args.variant]
    work_dir = args.work_dir or (RUNS_ROOT / 'ld_{}'.format(args.variant))
    gpu_ids = args.gpu_ids or str(args.gpu_id)
    gpu_count = len(gpu_ids.replace(',', ' ').split())
    env = build_env(args.data_root, teacher_checkpoint,
                    args.samples_per_gpu, args.workers, profile=args.profile,
                    gpu_count=gpu_count, fp16=args.fp16)
    run_train(config, work_dir, env, gpu_ids, args.epochs,
              args.resume_from, args.no_validate)


if __name__ == '__main__':
    main()
