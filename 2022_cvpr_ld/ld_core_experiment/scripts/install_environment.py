"""Install the pinned compatibility stack into the active GPU environment.

Default: require a prebuilt MMCV wheel, never silently start a CUDA build.
Use --build-mmcv only with a matching local CUDA toolkit and C++ compiler.
"""

import argparse
import platform
from pathlib import Path
import shutil
import subprocess
import sys

from apply_ld_compat import apply_patch

ROOT = Path(__file__).resolve().parents[1]
LD_ROOT = ROOT.parent / 'LD'
REQUIREMENTS = ROOT / 'requirements.txt'


def mmcv_wheel_index(torch_version, cuda_version):
    version = torch_version.split('+')[0].split('.')
    if tuple(version[:2]) not in (('2', '0'), ('2', '1')):
        raise ValueError('Use PyTorch 2.0.x or 2.1.x; preferred: 2.0.1 + CUDA 11.8.')
    if not cuda_version:
        raise ValueError('This installer requires a CUDA-enabled PyTorch build.')
    # OpenMMLab wheel indices use x.y.0 for all PyTorch x.y patch releases.
    cuda_tag = 'cu' + cuda_version.replace('.', '')
    return ('https://download.openmmlab.com/mmcv/dist/{}/torch{}.{}.0/index.html'
            .format(cuda_tag, *version[:2]))


def pip(*args):
    command = [sys.executable, '-m', 'pip', *args]
    print('+ ' + ' '.join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-mmcv', action='store_true',
                        help='explicitly compile MMCV instead of requiring a wheel')
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise SystemExit('Use this installer on the Linux x86_64 GPU rental machine.')
    if not (3, 8) <= sys.version_info[:2] <= (3, 10):
        raise SystemExit('Use Python 3.8-3.10 (recommended: 3.8 or 3.9).')
    try:
        import torch
        import torchvision
        index = mmcv_wheel_index(torch.__version__, torch.version.cuda)
        expected_vision = {'2.0': '0.15.', '2.1': '0.16.'}[
            '.'.join(torch.__version__.split('+')[0].split('.')[:2])]
        if not torchvision.__version__.startswith(expected_vision):
            raise ValueError('torchvision {} does not match PyTorch {}; expected torchvision {}x.'
                             .format(torchvision.__version__, torch.__version__, expected_vision))
        if not torch.cuda.is_available():
            raise ValueError('CUDA GPU is not visible. Check nvidia-smi / the GPU image.')
    except (ImportError, ValueError) as exc:
        raise SystemExit(str(exc))
    if args.build_mmcv and not shutil.which('nvcc'):
        raise SystemExit('--build-mmcv requires nvcc and a CUDA development image.')

    try:
        print('LD compatibility patch: ' + apply_patch())
        pip('install', 'pip<25', 'setuptools<70', 'wheel', 'Cython<3')
        # These packages all provide a top-level mmcv package and cannot coexist.
        pip('uninstall', '-y', 'mmcv', 'mmcv-lite', 'mmcv-full')
        # Install constraints first so a source build cannot pull NumPy 2.x.
        pins = [line for line in REQUIREMENTS.read_text().splitlines()
                if line and not line.startswith(('#', 'mmcv-full'))]
        pip('install', *pins)
        if args.build_mmcv:
            pip('install', '--no-build-isolation', '--no-binary=mmcv-full',
                '-c', str(REQUIREMENTS), 'mmcv-full==1.7.2')
        else:
            try:
                pip('install', '--only-binary=mmcv-full', '-f', index,
                    '-c', str(REQUIREMENTS), 'mmcv-full==1.7.2')
            except subprocess.CalledProcessError:
                raise SystemExit(
                    'No usable MMCV wheel was installed. Check network/index above. '
                    'Prefer torch 2.0.1 + cu118; if a wheel is unavailable, select '
                    'a CUDA development image and retry with --build-mmcv. '
                    'No automatic source build was attempted.')
        # setup.py installs all declared runtime requirements; pass our pins as
        # constraints so it cannot replace the tested NumPy/OpenCV versions.
        pip('install', '--no-build-isolation', '-c', str(REQUIREMENTS),
            '-e', str(LD_ROOT))
        subprocess.run([sys.executable, str(ROOT / 'scripts' / 'check_environment.py'),
                        '--require-cuda'], check=True)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc))
    print('Installation and CUDA operator checks passed. Next: run_smoke.sh with VOC data.')


if __name__ == '__main__':
    main()
