from pathlib import Path
import argparse

from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.utils import register_all_modules


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs' / 'gfl_r18_voc_baseline.py'
WORK_DIR = ROOT / 'runs' / 'baseline_debug'


def set_data_root(cfg, data_root):
    if not data_root:
        return
    for loader_name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        loader = getattr(cfg, loader_name, None)
        if loader is not None and loader.dataset is not None:
            loader.dataset.data_root = str(data_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override the epoch count from the config.')
    parser.add_argument('--data-root', type=Path, default=None,
                        help='Directory containing VOC2007/.')
    parser.add_argument('--device', default=None,
                        help='Runner device, for example cuda or cuda:0.')
    parser.add_argument('--skip-val', action='store_true',
                        help='Disable validation and test dataloaders for a smoke test.')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from the latest checkpoint in work_dir.')
    parser.add_argument('--val-interval', type=int, default=None,
                        help='Override validation interval in epochs.')
    parser.add_argument('--work-dir', type=Path, default=None)
    args = parser.parse_args()

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(args.config))
    cfg.default_scope = 'mmdet'
    set_data_root(cfg, args.data_root)

    if args.num_samples is not None:
        cfg.train_dataloader.dataset.indices = list(range(args.num_samples))
    if args.epochs is not None:
        cfg.train_cfg.max_epochs = args.epochs
    if args.val_interval is not None:
        cfg.train_cfg.val_interval = args.val_interval
    if args.skip_val:
        cfg.val_dataloader = None
        cfg.val_evaluator = None
        cfg.test_dataloader = None
        cfg.test_evaluator = None
    if args.device:
        cfg.device = args.device
    cfg.resume = args.resume
    cfg.work_dir = str(args.work_dir or WORK_DIR)
    cfg.log_level = 'INFO'

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
