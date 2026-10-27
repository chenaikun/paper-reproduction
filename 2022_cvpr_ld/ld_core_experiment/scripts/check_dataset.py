"""Validate the VOC2007+VOC2012 directory expected by the LD configs."""

import argparse
from pathlib import Path

from _common import DEFAULT_VOC_ROOT


def check_split(root, year, split):
    split_file = root / year / 'ImageSets' / 'Main' / '{}.txt'.format(split)
    if not split_file.is_file():
        raise SystemExit('Missing split file: {}'.format(split_file))
    image_ids = [line.split()[0] for line in split_file.read_text().splitlines()
                 if line.strip()]
    if not image_ids:
        raise SystemExit('Empty split file: {}'.format(split_file))
    for image_id in image_ids[:5]:
        image = root / year / 'JPEGImages' / '{}.jpg'.format(image_id)
        annotation = root / year / 'Annotations' / '{}.xml'.format(image_id)
        if not image.is_file() or not annotation.is_file():
            raise SystemExit(
                'Split references missing image or annotation: {}'.format(image_id))
    print('{} {}: {} samples'.format(year, split, len(image_ids)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=DEFAULT_VOC_ROOT,
                        help='Directory containing VOC2007/ and VOC2012/.')
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()

    check_split(root, 'VOC2007', 'trainval')
    check_split(root, 'VOC2012', 'trainval')
    check_split(root, 'VOC2007', 'test')
    print('dataset_root: {}'.format(root))
    print('dataset: valid')


if __name__ == '__main__':
    main()
