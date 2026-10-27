"""Verify that this checkout is using the official MMDetection 2.10 stack."""

import argparse
from pathlib import Path
import platform
import sys

from _common import LD_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()

    print('python: {}'.format(sys.version.split()[0]))
    print('platform: {}'.format(platform.platform()))

    errors = []
    try:
        import torch
        import mmcv
        import mmdet
        from mmcv.ops import nms  # noqa: F401
    except Exception as exc:
        raise SystemExit('OpenMMLab import failed: {}: {}'.format(
            type(exc).__name__, exc))

    torch_version = torch.__version__.split('+')[0]
    mmcv_version = mmcv.__version__.split('+')[0]
    mmdet_version = mmdet.__version__.split('+')[0]
    print('torch: {}'.format(torch.__version__))
    print('mmcv: {}'.format(mmcv.__version__))
    print('mmdet: {}'.format(mmdet.__version__))
    print('mmdet_path: {}'.format(Path(mmdet.__file__).resolve()))
    print('cuda_available: {}'.format(torch.cuda.is_available()))
    print('torch_cuda: {}'.format(torch.version.cuda or 'none'))

    if not (torch_version.startswith('1.7.') or
            torch_version.startswith('1.11.')):
        errors.append('expected PyTorch 1.7.x or the Qizhi-compatible 1.11.x')
    if not mmcv_version.startswith('1.2.'):
        errors.append('expected mmcv-full 1.2.x (the experiment pins 1.2.7)')
    if not mmdet_version.startswith('2.10.'):
        errors.append('expected MMDetection 2.10.x')
    if not str(Path(mmdet.__file__).resolve()).startswith(str(LD_ROOT.resolve())):
        errors.append('mmdet is not imported from the bundled LD checkout')
    if args.require_cuda and not torch.cuda.is_available():
        errors.append('CUDA is not visible to PyTorch')

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print('gpu_{}: {}'.format(index, torch.cuda.get_device_name(index)))

    if errors:
        raise SystemExit('; '.join(errors))
    print('environment: compatible')


if __name__ == '__main__':
    main()
