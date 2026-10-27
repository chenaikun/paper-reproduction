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

    gpu_ids = args.gpu_ids or str(args.gpu_id)
    gpu_count = len(gpu_ids.replace(',', ' ').split())
    env = build_env(args.data_root, samples_per_gpu=args.samples_per_gpu,
                    workers_per_gpu=args.workers, profile=args.profile,
                    gpu_count=gpu_count, fp16=args.fp16)
    run_train(args.config, args.work_dir, env, gpu_ids, args.epochs,
              args.resume_from, args.no_validate)


if __name__ == '__main__':
    main()
