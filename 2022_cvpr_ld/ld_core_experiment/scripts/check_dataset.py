"""Validate split-year or merged VOC07+12 data layouts."""

import argparse
from pathlib import Path

from _common import DEFAULT_VOC_ROOT


def check_split(root, year, split, check_all=False):
    split_file = root / year / 'ImageSets' / 'Main' / '{}.txt'.format(split)
    if not split_file.is_file():
        raise SystemExit('Missing split file: {}'.format(split_file))
    image_ids = [
        line.split()[0]
        for line in split_file.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if not image_ids:
        raise SystemExit('Empty split file: {}'.format(split_file))
    checked_ids = image_ids if check_all else image_ids[:5]
    for image_id in checked_ids:
        image = root / year / 'JPEGImages' / '{}.jpg'.format(image_id)
        annotation = root / year / 'Annotations' / '{}.xml'.format(image_id)
        if not image.is_file() or not annotation.is_file():
            raise SystemExit(
                'Split references missing image or annotation: {}'.format(image_id))
    print('{} {}: {} samples'.format(year, split, len(image_ids)))
    return len(image_ids)


def classify_data_layout(voc2007_train_count, has_voc2012):
    if has_voc2012:
        return 'VOC2007 + VOC2012'
    if voc2007_train_count >= 15000:
        return 'merged VOC07+12 under VOC2007'
    return 'VOC2007 only'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=DEFAULT_VOC_ROOT,
                        help='VOCdevkit directory containing VOC2007/.')
    parser.add_argument('--full', action='store_true',
                        help='Check every image and annotation path, not only five.')
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()

    voc2007_train_count = check_split(
        root, 'VOC2007', 'trainval', check_all=args.full)
    check_split(root, 'VOC2007', 'test', check_all=args.full)
    voc2012_train = root / 'VOC2012' / 'ImageSets' / 'Main' / 'trainval.txt'
    if voc2012_train.is_file():
        check_split(root, 'VOC2012', 'trainval', check_all=args.full)
    layout = classify_data_layout(voc2007_train_count, voc2012_train.is_file())
    print('data_layout: {}'.format(layout))
    if layout == 'VOC2007 only':
        print('warning: this is not the 16,551-image VOC07+12 training set')
    print('dataset_root: {}'.format(root))
    print('dataset: valid')


if __name__ == '__main__':
    main()
