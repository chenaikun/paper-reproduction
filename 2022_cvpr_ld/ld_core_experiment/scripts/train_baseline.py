import argparse
from pathlib import Path

from _common import BASELINE_CONFIG, RUNS_ROOT, build_env, run_train


def main():
    parser = argparse.ArgumentParser(description='Train the GFL-R18 VOC baseline.')
    parser.add_argument('--config', type=Path, default=BASELINE_CONFIG)
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--work-dir', type=Path,
                        default=RUNS_ROOT / 'student_r18_baseline')
    parser.add_argument('--epochs', type=int, default=4)
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
