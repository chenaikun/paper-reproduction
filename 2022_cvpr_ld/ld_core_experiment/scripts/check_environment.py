"""Check the runtime packages and CUDA visibility before starting a run."""

import argparse
import platform
import sys


def version_of(name):
    try:
        module = __import__(name)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f'UNAVAILABLE ({type(exc).__name__}: {exc})'
    return getattr(module, '__version__', 'imported')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()

    print(f'python: {sys.version.split()[0]}')
    print(f'platform: {platform.platform()}')
    versions = {name: version_of(name) for name in ('torch', 'mmcv', 'mmengine', 'mmdet')}
    for name, version in versions.items():
        print(f'{name}: {version}')
    missing = [name for name, version in versions.items()
               if version.startswith('UNAVAILABLE')]
    if missing and args.require_cuda:
        raise SystemExit(f'Missing or incompatible packages: {", ".join(missing)}')

    try:
        import torch
    except Exception:
        if args.require_cuda:
            raise SystemExit('torch is required for a CUDA check.')
        return

    print(f'cuda_available: {torch.cuda.is_available()}')
    print(f'torch_cuda: {torch.version.cuda or "none"}')
    if torch.cuda.is_available():
        print(f'gpu_count: {torch.cuda.device_count()}')
        for index in range(torch.cuda.device_count()):
            print(f'gpu_{index}: {torch.cuda.get_device_name(index)}')
    elif args.require_cuda:
        raise SystemExit('CUDA is not visible to PyTorch.')


if __name__ == '__main__':
    main()
