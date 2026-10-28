"""Regression tests for VOC dataset layout classification."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from check_dataset import classify_data_layout  # noqa: E402


class DatasetLayoutTest(unittest.TestCase):
    def test_voc2007_only_is_not_called_merged(self):
        self.assertEqual(
            classify_data_layout(5011, has_voc2012=False),
            'VOC2007 only',
        )

    def test_merged_layout_requires_combined_count(self):
        self.assertEqual(
            classify_data_layout(16551, has_voc2012=False),
            'merged VOC07+12 under VOC2007',
        )

    def test_split_year_layout(self):
        self.assertEqual(
            classify_data_layout(5011, has_voc2012=True),
            'VOC2007 + VOC2012',
        )


if __name__ == '__main__':
    unittest.main()
