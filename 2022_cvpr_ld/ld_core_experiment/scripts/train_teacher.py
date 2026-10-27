import argparse
from pathlib import Path

from _common import RUNS_ROOT, TEACHER_CONFIG, build_env, run_train


def main():
    parser = argparse.ArgumentParser(description='Train the GFL-R101 VOC teacher.')
    parser.add_argument('--config', type=Path, default=TEACHER_CONFIG)
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--work-dir', type=Path,
                        default=RUNS_ROOT / 'teacher_r101')
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--samples-per-gpu', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--resume-from', type=Path, default=None)
    parser.add_argument('--no-validate', action='store_true')
    args = parser.parse_args()

    env = build_env(args.data_root, samples_per_gpu=args.samples_per_gpu,
                    workers_per_gpu=args.workers)
    run_train(args.config, args.work_dir, env, args.gpu_id, args.epochs,
              args.resume_from, args.no_validate)


if __name__ == '__main__':
    main()
