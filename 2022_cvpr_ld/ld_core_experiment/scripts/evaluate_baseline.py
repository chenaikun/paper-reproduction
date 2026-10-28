import argparse
from pathlib import Path

from _common import (BASELINE_CONFIG, DEFAULT_TEACHER_CHECKPOINT, build_env,
                    run_test)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a VOC checkpoint.')
    parser.add_argument('--config', type=Path, default=BASELINE_CONFIG)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--teacher-checkpoint', type=Path, default=None)
    parser.add_argument('--profile',
                        choices=['auto', 'tiny', 'low', 'default', 'high'],
                        default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--log-file', type=Path, default=None)
    args = parser.parse_args()

    teacher = args.teacher_checkpoint
    if 'ld_r18_voc' in args.config.name and teacher is None:
        teacher = DEFAULT_TEACHER_CHECKPOINT
    if teacher is not None and not teacher.expanduser().is_file():
        raise SystemExit('Teacher checkpoint not found: {}'.format(teacher))
    env = build_env(
        args.data_root,
        teacher,
        workers_per_gpu=args.workers,
        profile=args.profile,
        gpu_count=1,
        fp16=args.fp16,
    )
    run_test(
        args.config,
        args.checkpoint,
        env,
        output=args.output,
        log_file=args.log_file,
    )


if __name__ == '__main__':
    main()
