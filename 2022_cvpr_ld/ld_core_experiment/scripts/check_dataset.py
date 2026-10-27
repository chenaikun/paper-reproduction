"""Validate split-year or merged VOC07+12 data layouts."""

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
                        help='VOCdevkit directory containing VOC2007/.')
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()

    check_split(root, 'VOC2007', 'trainval')
    check_split(root, 'VOC2007', 'test')
    voc2012_train = root / 'VOC2012' / 'ImageSets' / 'Main' / 'trainval.txt'
    if voc2012_train.is_file():
        check_split(root, 'VOC2012', 'trainval')
        print('data_layout: split VOC2007 + VOC2012')
    else:
        print('data_layout: merged VOC07+12 under VOC2007')
    print('dataset_root: {}'.format(root))
    print('dataset: valid')


if __name__ == '__main__':
    main()
