from pathlib import Path
import argparse

from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.utils import register_all_modules


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs' / 'gfl_r18_voc_baseline.py'
WORK_DIR = ROOT / 'runs' / 'baseline_debug'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--work-dir', type=Path, default=None)
    args = parser.parse_args()

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(args.config))
    cfg.default_scope = 'mmdet'

    if args.num_samples is not None:
        cfg.train_dataloader.dataset.indices = list(range(args.num_samples))
    cfg.train_cfg.max_epochs = args.epochs
    cfg.train_cfg.val_interval = 999
    cfg.val_dataloader = None
    cfg.val_evaluator = None
    cfg.test_dataloader = None
    cfg.test_evaluator = None
    cfg.work_dir = str(args.work_dir or WORK_DIR)
    cfg.log_level = 'INFO'

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
