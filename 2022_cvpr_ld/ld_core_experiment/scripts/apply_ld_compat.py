"""Apply the tracked LD compatibility patch without changing submodule commits.

No third-party Python dependencies are needed. Repeated runs are safe; a
conflicting checkout fails before any files are changed.
"""

import argparse
from pathlib import Path
import subprocess

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LD_ROOT = EXPERIMENT_ROOT.parent / 'LD'
PATCH_PATH = EXPERIMENT_ROOT / 'patches' / 'ld-mmcv-1.7.2.patch'


def patch_state(ld_root=DEFAULT_LD_ROOT):
    root = Path(ld_root).resolve()
    if not (root / 'mmdet' / '__init__.py').is_file():
        raise RuntimeError('LD source missing. Run git submodule update --init --recursive.')
    # Explicit paths also support exported source trees without Git metadata.
    def check(reverse=False):
        command = ['git', 'apply', '--check', '--unsafe-paths',
                   '--directory=' + str(root)]
        if reverse:
            command.append('--reverse')
        command.append(str(PATCH_PATH))
        return subprocess.run(command, cwd=str(root), capture_output=True, text=True)

    if check(reverse=True).returncode == 0:
        return 'applied'
    result = check()
    if result.returncode == 0:
        return 'needed'
    raise RuntimeError('LD compatibility patch conflicts with this checkout; '
                       'no files changed.\n' + result.stderr.strip())


def apply_patch(ld_root=DEFAULT_LD_ROOT, check_only=False):
    root = Path(ld_root).resolve()
    state = patch_state(root)
    if state == 'needed' and not check_only:
        subprocess.run(['git', 'apply', '--unsafe-paths',
                        '--directory=' + str(root), str(PATCH_PATH)],
                       cwd=str(root), check=True)
        state = 'applied'
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ld-root', type=Path, default=DEFAULT_LD_ROOT)
    parser.add_argument('--check', action='store_true', help='do not modify files')
    args = parser.parse_args()
    try:
        state = apply_patch(args.ld_root, check_only=args.check)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc))
    print('LD compatibility patch: ' + state)
    if args.check and state != 'applied':
        raise SystemExit('Run python scripts/apply_ld_compat.py before installing LD.')


if __name__ == '__main__':
    main()
