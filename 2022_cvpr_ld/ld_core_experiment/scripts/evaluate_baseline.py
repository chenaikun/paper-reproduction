from pathlib import Path
import argparse

from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.utils import register_all_modules


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs' / 'gfl_r18_voc_baseline.py'
CHECKPOINT = ROOT / 'runs' / 'baseline_debug' / 'epoch_1.pth'
WORK_DIR = ROOT / 'runs' / 'baseline_eval_debug'


def set_data_root(cfg, data_root):
    if data_root:
        cfg.test_dataloader.dataset.data_root = str(data_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--work-dir', type=Path, default=WORK_DIR)
    parser.add_argument('--data-root', type=Path, default=None,
                        help='Directory containing VOC2007/.')
    parser.add_argument('--device', default=None,
                        help='Runner device, for example cuda or cuda:0.')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Evaluate only the first N test images.')
    args = parser.parse_args()

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(args.config))
    cfg.default_scope = 'mmdet'
    set_data_root(cfg, args.data_root)
    cfg.load_from = str(args.checkpoint)
    cfg.work_dir = str(args.work_dir)
    if args.device:
        cfg.device = args.device

    if args.num_samples is not None:
        cfg.test_dataloader.dataset.indices = list(range(args.num_samples))
    cfg.test_cfg = dict(type='TestLoop')
    cfg.train_dataloader = None
    cfg.train_cfg = None
    cfg.optim_wrapper = None
    cfg.param_scheduler = None
    cfg.val_dataloader = None
    cfg.val_evaluator = None
    cfg.val_cfg = None

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()
