"""Check patched LD imports, model registries and real MMCV operator execution."""

import argparse
from pathlib import Path
import platform
import sys

from _common import LD_ROOT
from apply_ld_compat import apply_patch


def check_ops(torch, nms, roi_align, device):
    """Execute extension code, not just import it (including backward)."""
    boxes = torch.tensor([[0., 0., 8., 8.], [0., 0., 8., 8.],
                          [20., 20., 28., 28.]], device=device)
    scores = torch.tensor([0.9, 0.8, 0.7], device=device)
    _, keep = nms(boxes, scores, 0.5)
    if keep.cpu().tolist() != [0, 2]:
        raise RuntimeError('NMS returned unexpected indices on ' + device)
    feature = torch.ones(1, 1, 8, 8, device=device, requires_grad=True)
    rois = torch.tensor([[0., 1., 1., 6., 6.]], device=device)
    output = roi_align(feature, rois, (2, 2), spatial_scale=1.0,
                       sampling_ratio=2, pool_mode='avg', aligned=True)
    output.sum().backward()
    if (not torch.isfinite(output).all().item() or feature.grad is None
            or not torch.isfinite(feature.grad).all().item()
            or feature.grad.abs().sum().item() == 0):
        raise RuntimeError('ROIAlign forward/backward failed on ' + device)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    print('operators [{}]: NMS + ROIAlign forward/backward passed'.format(device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()
    try:
        if apply_patch(check_only=True) != 'applied':
            raise RuntimeError('Run python scripts/apply_ld_compat.py first.')
    except RuntimeError as exc:
        raise SystemExit(str(exc))
    print('python: {}'.format(sys.version.split()[0]))
    print('platform: {}'.format(platform.platform()))

    try:
        import torch
        import torchvision
        import numpy as np
        import mmcv
        import mmdet
        from mmcv.ops import nms, roi_align
        from mmdet.apis import train_detector  # noqa: F401
        from mmdet.datasets import build_dataset  # noqa: F401
        from mmdet.models.builder import HEADS, DETECTORS
    except Exception as exc:
        raise SystemExit('OpenMMLab import failed: {}: {}'.format(
            type(exc).__name__, exc))

    print('torch: {}'.format(torch.__version__))
    print('torchvision: {}'.format(torchvision.__version__))
    print('numpy: {}'.format(np.__version__))
    print('mmcv: {}'.format(mmcv.__version__))
    print('mmdet: {}'.format(mmdet.__version__))
    print('mmdet_path: {}'.format(Path(mmdet.__file__).resolve()))
    print('torch_cuda: {}'.format(torch.version.cuda or 'none'))
    errors = []
    if not torch.__version__.startswith(('2.0.', '2.1.')):
        errors.append('modern stack requires PyTorch 2.0.x or 2.1.x')
    expected_vision = {'2.0': '0.15.', '2.1': '0.16.'}.get(
        '.'.join(torch.__version__.split('+')[0].split('.')[:2]))
    if expected_vision and not torchvision.__version__.startswith(expected_vision):
        errors.append('torchvision {} does not match PyTorch {}'
                      .format(torchvision.__version__, torch.__version__))
    if mmcv.__version__ != '1.7.2':
        errors.append('modern stack requires mmcv-full==1.7.2, not MMCV 2.x')
    if not mmdet.__version__.startswith('2.10.'):
        errors.append('expected bundled MMDetection 2.10.x')
    if LD_ROOT.resolve() not in Path(mmdet.__file__).resolve().parents:
        errors.append('mmdet is not imported from the bundled LD checkout')
    if int(np.__version__.split('.')[0]) >= 2:
        errors.append('NumPy 2.x is unsupported; install the pinned requirements')
    for registry, name in ((HEADS, 'GFLHead'), (HEADS, 'LDHead'),
                           (DETECTORS, 'KnowledgeDistillationSingleStageDetector')):
        if registry.get(name) is None:
            errors.append('missing model registration: ' + name)
    if args.require_cuda and not torch.cuda.is_available():
        errors.append('CUDA is not visible to PyTorch')
    if errors:
        raise SystemExit('; '.join(errors))

    try:
        # Also catches the YAPF verify=True API incompatibility before training.
        config = mmcv.Config(dict(model=dict(type='GFL'), gpu_ids=[0]))
        if 'GFL' not in config.pretty_text:
            raise RuntimeError('MMCV config formatting failed')
        check_ops(torch, nms, roi_align, 'cpu')
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                print('gpu_{}: {}'.format(index, torch.cuda.get_device_name(index)))
            check_ops(torch, nms, roi_align, 'cuda:0')
    except Exception as exc:
        raise SystemExit('Runtime compatibility check failed: {}: {}'.format(
            type(exc).__name__, exc))
    print('environment: operator checks passed; run the VOC smoke test before training')


if __name__ == '__main__':
    main()
