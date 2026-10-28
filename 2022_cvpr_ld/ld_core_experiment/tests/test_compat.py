"""Dependency-free regression tests for the shipped compatibility patch."""

import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
LD_ROOT = ROOT.parent / 'LD'
sys.path.insert(0, str(ROOT / 'scripts'))
from apply_ld_compat import PATCH_PATH, apply_patch, patch_state  # noqa: E402
from install_environment import mmcv_wheel_index  # noqa: E402


class CompatibilityPatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkout = Path(self.temp.name) / 'LD'
        self.checkout.mkdir()
        self.paths = re.findall(
            r'^diff --git a/(.*?) b/', PATCH_PATH.read_text(encoding='utf-8'), re.M)
        for relative in self.paths:
            target = self.checkout / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(
                ['git', '-C', str(LD_ROOT), 'show', 'HEAD:' + relative]))

    def test_patch_applies_and_is_idempotent(self):
        self.assertEqual(patch_state(self.checkout), 'needed')
        self.assertEqual(apply_patch(self.checkout, check_only=True), 'needed')
        self.assertEqual(apply_patch(self.checkout), 'applied')
        before = {p: (self.checkout / p).read_bytes() for p in self.paths}
        self.assertEqual(apply_patch(self.checkout), 'applied')
        self.assertEqual(before, {p: (self.checkout / p).read_bytes() for p in self.paths})

    def test_conflicting_patch_does_not_modify_other_files(self):
        target = self.checkout / 'mmdet/__init__.py'
        target.write_text('a conflicting local edit\n')
        before = {p: (self.checkout / p).read_bytes() for p in self.paths}
        with self.assertRaisesRegex(RuntimeError, 'conflicts'):
            apply_patch(self.checkout)
        self.assertEqual(before, {p: (self.checkout / p).read_bytes() for p in self.paths})

    def test_version_guard_accepts_only_legacy_or_exact_compat_release(self):
        apply_patch(self.checkout)
        package = '_ld_compat_version_test'
        for version, accepted in [('1.2.3', False), ('1.2.4', True),
                                  ('1.2.7', True), ('1.3', True),
                                  ('1.7.0', False), ('1.7.2', True),
                                  ('1.7.2rc1', False), ('2.0.0', False)]:
            with self.subTest(mmcv=version):
                mmcv = types.ModuleType('mmcv')
                mmcv.__version__ = version
                # The guard's only relative import is version.py.
                version_module = types.ModuleType(package + '.version')
                version_module.__version__ = '2.10.0'
                version_module.short_version = '2.10.0'
                spec = importlib.util.spec_from_file_location(
                    package, self.checkout / 'mmdet/__init__.py',
                    submodule_search_locations=[str(self.checkout / 'mmdet')])
                module = importlib.util.module_from_spec(spec)
                with patch.dict(sys.modules, {'mmcv': mmcv, package: module,
                                             package + '.version': version_module}):
                    if accepted:
                        spec.loader.exec_module(module)
                    else:
                        with self.assertRaises(AssertionError):
                            spec.loader.exec_module(module)

    def test_patched_sources_compile_and_remove_deprecated_aliases(self):
        apply_patch(self.checkout)
        for relative in self.paths:
            if relative.endswith('.py'):
                source = (self.checkout / relative).read_text(encoding='utf-8')
                compile(source, relative, 'exec')
                self.assertIsNone(re.search(r'\bnp\.(bool|int)\b', source), relative)
        self.assertIn('get_root_logger(log_level=cfg.log_level)',
                      (self.checkout / 'mmdet/apis/train.py').read_text(
                          encoding='utf-8'))
        self.assertNotIn('mmpycocotools',
                         (self.checkout / 'requirements/runtime.txt').read_text(
                             encoding='utf-8'))
        self.assertIn(
            'device=multi_scores.device',
            (self.checkout / 'mmdet/core/post_processing/bbox_nms.py').read_text(
                encoding='utf-8'))

    def test_missing_checkout_has_actionable_error(self):
        with self.assertRaisesRegex(RuntimeError, 'submodule update'):
            patch_state(self.checkout / 'missing')


class WheelIndexTests(unittest.TestCase):
    def test_patch_versions_use_minor_release_index(self):
        self.assertEqual(mmcv_wheel_index('2.0.1+cu118', '11.8'),
                         'https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html')
        self.assertIn('/cu121/torch2.1.0/', mmcv_wheel_index('2.1.2', '12.1'))

    def test_reject_cpu_and_unsupported_torch(self):
        for torch_version, cuda in [('2.5.1', '12.4'), ('1.11.0', '11.3'),
                                    ('2.0.1', None)]:
            with self.subTest(torch=torch_version, cuda=cuda):
                with self.assertRaises(ValueError):
                    mmcv_wheel_index(torch_version, cuda)


if __name__ == '__main__':
    unittest.main()
