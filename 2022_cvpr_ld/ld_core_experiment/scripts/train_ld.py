import argparse
from pathlib import Path

from _common import (DEFAULT_TEACHER_CHECKPOINT, LD_CONFIGS, RUNS_ROOT,
                    build_env, resolve_single_gpu_id, run_train)


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
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--repeat-times', type=int, default=None)
    parser.add_argument('--fp16', action='store_true',
                        help='enable dynamic-loss-scale mixed precision')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--load-from', type=Path, default=None,
                        help='Initialize student weights from a detector checkpoint.')
    parser.add_argument('--resume-from', type=Path, default=None)
    parser.add_argument('--no-validate', action='store_true')
    args = parser.parse_args()

    teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()
    if not teacher_checkpoint.is_file():
        raise SystemExit('Teacher checkpoint not found: {}'.format(teacher_checkpoint))
    config = args.config or LD_CONFIGS[args.variant]
    work_dir = args.work_dir or (RUNS_ROOT / 'ld_{}'.format(args.variant))
    gpu_id = resolve_single_gpu_id(args.gpu_id)
    env = build_env(args.data_root, teacher_checkpoint,
                    args.samples_per_gpu, args.workers, profile=args.profile,
                    gpu_count=1, fp16=args.fp16,
                    repeat_times=args.repeat_times)
    run_train(
        config,
        work_dir,
        env,
        gpu_ids=gpu_id,
        epochs=args.epochs,
        resume_from=args.resume_from,
        no_validate=args.no_validate,
        load_from=args.load_from,
        seed=args.seed,
        deterministic=args.deterministic,
    )


if __name__ == '__main__':
    main()
