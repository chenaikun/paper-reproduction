import argparse
from pathlib import Path

from _common import (BASELINE_CONFIG, RUNS_ROOT, build_env,
                     resolve_single_gpu_id, run_train)


def main():
    parser = argparse.ArgumentParser(description='Train the GFL-R18 VOC baseline.')
    parser.add_argument('--config', type=Path, default=BASELINE_CONFIG)
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--work-dir', type=Path,
                        default=RUNS_ROOT / 'student_r18_baseline')
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
                        help='Initialize model weights from a detector checkpoint.')
    parser.add_argument('--resume-from', type=Path, default=None)
    parser.add_argument('--no-validate', action='store_true')
    args = parser.parse_args()

    gpu_id = resolve_single_gpu_id(args.gpu_id)
    env = build_env(args.data_root, samples_per_gpu=args.samples_per_gpu,
                    workers_per_gpu=args.workers, profile=args.profile,
                    gpu_count=1, fp16=args.fp16,
                    repeat_times=args.repeat_times)
    run_train(
        args.config,
        args.work_dir,
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
