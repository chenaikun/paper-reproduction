"""Regression tests for environment and command construction helpers."""

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import _common  # noqa: E402


class CommonHelpersTest(unittest.TestCase):
    def test_single_gpu_validation(self):
        self.assertEqual(_common.resolve_single_gpu_id(0), '0')
        self.assertEqual(_common.resolve_single_gpu_id('1'), '1')
        with self.assertRaisesRegex(ValueError, 'Exactly one'):
            _common.resolve_single_gpu_id('0,1')

    def test_schedule_decay_matches_existing_voc_schedules(self):
        self.assertIsNone(_common.schedule_decay_epoch(1))
        self.assertEqual(_common.schedule_decay_epoch(4), 3)
        self.assertEqual(_common.schedule_decay_epoch(8), 6)
        self.assertEqual(_common.schedule_decay_epoch(16), 12)
        with self.assertRaisesRegex(ValueError, 'at least 1'):
            _common.schedule_decay_epoch(0)

    def test_build_env_has_explicit_repeat_and_profile(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
                os.environ, {}, clear=True):
            env = _common.build_env(
                data_root=temp,
                profile='tiny',
                gpu_count=1,
                repeat_times=1,
            )
        self.assertEqual(env['LD_PROFILE'], 'tiny')
        self.assertEqual(env['LD_REPEAT_TIMES'], '1')
        self.assertEqual(env['GPU_COUNT'], '1')
        self.assertEqual(env['LD_IMAGE_WIDTH'], '640')
        self.assertEqual(env['LD_IMAGE_HEIGHT'], '384')

    def test_run_train_passes_load_seed_and_schedule_together(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / 'config.py'
            config.write_text('model = dict()\n', encoding='utf-8')
            load_from = root / 'pretrained.pth'
            load_from.write_bytes(b'checkpoint')
            work_dir = root / 'work'
            with patch.object(
                    _common, 'require_ld_checkout',
                    return_value=root / 'train.py'), patch.object(
                        _common.subprocess, 'run') as run:
                _common.run_train(
                    config,
                    work_dir,
                    {},
                    gpu_ids='0',
                    epochs=8,
                    load_from=load_from,
                    seed=3,
                    deterministic=True,
                )
        command = run.call_args.args[0]
        self.assertIn('total_epochs=8', command)
        self.assertIn('lr_config.step=[6]', command)
        self.assertIn('load_from={}'.format(load_from.resolve()), command)
        self.assertIn('--seed', command)
        self.assertIn('3', command)
        self.assertIn('--deterministic', command)

    def test_main_ld_only_variant_is_registered(self):
        self.assertIn('main-ld-only', _common.LD_CONFIGS)
        self.assertTrue(_common.LD_CONFIGS['main-ld-only'].is_file())

    def test_run_test_captures_stdout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / 'config.py'
            config.write_text('model = dict()\n', encoding='utf-8')
            checkpoint = root / 'epoch_1.pth'
            checkpoint.write_bytes(b'checkpoint')
            log_file = root / 'eval.log'
            process = MagicMock()
            process.stdout = iter(['metric line\n'])
            process.wait.return_value = 0
            with patch.object(
                    _common, 'require_ld_checkout',
                    return_value=root / 'test.py'), patch.object(
                        _common.subprocess, 'Popen',
                        return_value=process), redirect_stdout(io.StringIO()):
                _common.run_test(config, checkpoint, {}, log_file=log_file)
            self.assertEqual(log_file.read_text(encoding='utf-8'),
                             'metric line\n')


if __name__ == '__main__':
    unittest.main()
