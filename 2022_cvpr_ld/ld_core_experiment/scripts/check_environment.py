"""Verify the LD/MMDetection 2.10 stack and its PyTorch compatibility mode."""

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

    supported_torch = (
        '1.7.', '1.8.', '1.9.', '1.10.', '1.11.', '2.0.', '2.1.')
    if not torch_version.startswith(supported_torch):
        errors.append('expected PyTorch 1.7-1.11 or 2.0-2.1')

    if not mmcv_version.startswith(('1.2.', '1.7.')):
        errors.append('expected mmcv-full 1.2.x or 1.7.x')
    elif torch_version.startswith(('2.0.', '2.1.')) and not mmcv_version.startswith('1.7.'):
        errors.append('PyTorch 2.x requires the MMCV 1.7.x compatibility path')
    elif torch_version.startswith(('1.7.', '1.8.', '1.11.')) and mmcv_version.startswith('1.2.'):
        print('environment mode: original LD reference stack')
    elif torch_version.startswith(('2.0.', '2.1.')):
        print('environment mode: PyTorch 2.x compatibility stack')
    else:
        print('warning: this torch/MMCV pair is accepted but not a primary '
              'tested combination')
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
