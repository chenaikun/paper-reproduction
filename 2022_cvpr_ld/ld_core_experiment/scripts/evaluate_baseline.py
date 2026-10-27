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
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    teacher = args.teacher_checkpoint
    if 'ld_r18_voc' in args.config.name and teacher is None:
        teacher = DEFAULT_TEACHER_CHECKPOINT
    env = build_env(args.data_root, teacher)
    run_test(args.config, args.checkpoint, env, args.output)


if __name__ == '__main__':
    main()
