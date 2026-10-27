import os
from pathlib import Path
import subprocess
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parent
LD_ROOT = PROJECT_ROOT / 'LD'
CONFIG_ROOT = EXPERIMENT_ROOT / 'configs'
RUNS_ROOT = EXPERIMENT_ROOT / 'runs'
DEFAULT_VOC_ROOT = PROJECT_ROOT / 'data' / 'VOCdevkit'
DEFAULT_TEACHER_CHECKPOINT = RUNS_ROOT / 'teacher_r101' / 'epoch_16.pth'

BASELINE_CONFIG = CONFIG_ROOT / 'gfl_r18_voc_baseline.py'
TEACHER_CONFIG = CONFIG_ROOT / 'gfl_r101_voc_teacher.py'
LD_CONFIGS = {
    'kd-only': CONFIG_ROOT / 'ld_r18_voc_kd_only.py',
    'loc-only': CONFIG_ROOT / 'ld_r18_voc_loc_only.py',
    'full': CONFIG_ROOT / 'ld_r18_voc_full.py',
}

RUNTIME_PROFILES = {
    # The dimensions are deliberately multiples of 32 after the resize step.
    'tiny': dict(width=640, height=384, samples_per_gpu=1, workers=0),
    'low': dict(width=800, height=480, samples_per_gpu=1, workers=0),
    'default': dict(width=1000, height=600, samples_per_gpu=1, workers=2),
    'high': dict(width=1333, height=800, samples_per_gpu=1, workers=4),
}


def _gpu_memory_gib():
    """Return visible GPU memory when torch is available, otherwise None."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception:
        pass
    return None


def resolve_profile(profile=None):
    """Resolve a named or automatic profile to a concrete runtime profile."""
    requested = (profile or os.environ.get('LD_PROFILE', 'auto')).lower()
    if requested != 'auto':
        if requested not in RUNTIME_PROFILES:
            raise ValueError(
                'Unknown LD_PROFILE {!r}; choose auto, tiny, low, default, '
                'or high'.format(requested))
        return requested

    memory_gib = _gpu_memory_gib()
    if memory_gib is None or memory_gib >= 20:
        return 'default'
    if memory_gib >= 12:
        return 'low'
    return 'tiny'


def _normalize_gpu_ids(gpu_ids):
    if isinstance(gpu_ids, int):
        return [gpu_ids]
    if isinstance(gpu_ids, str):
        return [int(item) for item in gpu_ids.replace(',', ' ').split()]
    return [int(item) for item in gpu_ids]


def build_env(data_root=None, teacher_checkpoint=None,
              samples_per_gpu=None, workers_per_gpu=None, profile=None,
              gpu_count=None, fp16=None):
    env = os.environ.copy()
    resolved_profile = resolve_profile(profile)
    profile_settings = RUNTIME_PROFILES[resolved_profile]
    env['VOC_DATA_ROOT'] = str(Path(data_root or env.get(
        'VOC_DATA_ROOT', DEFAULT_VOC_ROOT)).expanduser().resolve())
    env['LD_PROFILE'] = resolved_profile
    env['LD_IMAGE_WIDTH'] = str(env.get('LD_IMAGE_WIDTH', profile_settings['width']))
    env['LD_IMAGE_HEIGHT'] = str(env.get('LD_IMAGE_HEIGHT', profile_settings['height']))
    env['SAMPLES_PER_GPU'] = str(
        env.get('SAMPLES_PER_GPU', profile_settings['samples_per_gpu'])
        if samples_per_gpu is None else samples_per_gpu)
    env['WORKERS_PER_GPU'] = str(
        env.get('WORKERS_PER_GPU', profile_settings['workers'])
        if workers_per_gpu is None else workers_per_gpu)
    if gpu_count is not None:
        env['GPU_COUNT'] = str(gpu_count)
    if fp16 is not None:
        env['USE_FP16'] = '1' if fp16 else '0'
    if teacher_checkpoint:
        env['TEACHER_CHECKPOINT'] = str(
            Path(teacher_checkpoint).expanduser().resolve())
    return env


def require_ld_checkout():
    train_tool = LD_ROOT / 'tools' / 'train.py'
    if not train_tool.is_file():
        raise SystemExit(
            'Official LD submodule is missing. Run '
            '`git submodule update --init --recursive` from the repository root.')
    return train_tool


def run_train(config, work_dir, env, gpu_ids='0', epochs=None,
              resume_from=None, no_validate=False):
    train_tool = require_ld_checkout()
    normalized_gpu_ids = _normalize_gpu_ids(gpu_ids)
    command = [
        sys.executable,
        str(train_tool),
        str(Path(config).resolve()),
        '--work-dir', str(Path(work_dir).expanduser().resolve()),
        '--gpu-ids', *[str(gpu_id) for gpu_id in normalized_gpu_ids],
    ]
    if resume_from:
        command.extend(['--resume-from', str(Path(resume_from).expanduser().resolve())])
    if no_validate:
        command.append('--no-validate')
    if epochs is not None:
        command.extend(['--cfg-options', 'total_epochs={}'.format(epochs)])
    subprocess.run(command, cwd=str(LD_ROOT), env=env, check=True)


def run_test(config, checkpoint, env, output=None):
    test_tool = LD_ROOT / 'tools' / 'test.py'
    if not test_tool.is_file():
        require_ld_checkout()
    command = [
        sys.executable,
        str(test_tool),
        str(Path(config).resolve()),
        str(Path(checkpoint).expanduser().resolve()),
        '--eval', 'mAP',
    ]
    if output:
        command.extend(['--out', str(Path(output).expanduser().resolve())])
    subprocess.run(command, cwd=str(LD_ROOT), env=env, check=True)
