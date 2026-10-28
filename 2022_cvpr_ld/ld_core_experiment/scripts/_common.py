import os
import math
from pathlib import Path
import subprocess
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parent
LD_ROOT = PROJECT_ROOT / 'LD'
CONFIG_ROOT = EXPERIMENT_ROOT / 'configs'
RUNS_ROOT = EXPERIMENT_ROOT / 'runs'


def _default_voc_root():
    candidates = [PROJECT_ROOT / 'data' / 'VOCdevkit', PROJECT_ROOT / 'data']
    for candidate in candidates:
        if (candidate / 'VOC2007' / 'ImageSets' / 'Main' /
                'trainval.txt').is_file():
            return candidate
    return candidates[0]


DEFAULT_VOC_ROOT = _default_voc_root()
DEFAULT_TEACHER_CHECKPOINT = RUNS_ROOT / 'teacher_r101' / 'epoch_16.pth'

BASELINE_CONFIG = CONFIG_ROOT / 'gfl_r18_voc_baseline.py'
TEACHER_CONFIG = CONFIG_ROOT / 'gfl_r101_voc_teacher.py'
LD_CONFIGS = {
    'kd-only': CONFIG_ROOT / 'ld_r18_voc_kd_only.py',
    'loc-only': CONFIG_ROOT / 'ld_r18_voc_loc_only.py',
    'main-ld-only': CONFIG_ROOT / 'ld_r18_voc_main_ld_only.py',
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


def resolve_single_gpu_id(gpu_id):
    """Return one GPU id and reject paths that imply unsupported multi-GPU."""
    normalized = _normalize_gpu_ids(gpu_id)
    if len(normalized) != 1:
        raise ValueError('Exactly one --gpu-id is supported by this runner.')
    return str(normalized[0])


def schedule_decay_epoch(epochs):
    """Use the repository's 75 percent decay point for a custom epoch count."""
    if epochs is None:
        return None
    if epochs < 1:
        raise ValueError('epochs must be at least 1')
    if epochs == 1:
        return None
    return max(1, int(math.ceil(epochs * 0.75)))


def build_env(data_root=None, teacher_checkpoint=None,
              samples_per_gpu=None, workers_per_gpu=None, profile=None,
              gpu_count=None, fp16=None, repeat_times=None):
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
    if repeat_times is not None:
        if repeat_times < 1:
            raise ValueError('repeat_times must be at least 1')
        env['LD_REPEAT_TIMES'] = str(repeat_times)
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
    from apply_ld_compat import apply_patch
    if apply_patch(check_only=True) != 'applied':
        raise SystemExit(
            'LD compatibility patch missing. Run python scripts/apply_ld_compat.py.')
    return train_tool


def run_train(config, work_dir, env, gpu_ids='0', epochs=None,
              resume_from=None, no_validate=False, load_from=None,
              seed=0, deterministic=False):
    train_tool = require_ld_checkout()
    config_path = Path(config).expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit('Config not found: {}'.format(config_path))
    normalized_gpu_ids = _normalize_gpu_ids(gpu_ids)
    if len(normalized_gpu_ids) != 1:
        raise SystemExit(
            'This MMCV runner supports one GPU per process. Select one --gpu-id; '
            'multi-GPU training requires a separate distributed launcher.')
    command = [
        sys.executable,
        str(train_tool),
        str(config_path),
        '--work-dir', str(Path(work_dir).expanduser().resolve()),
        '--gpu-ids', *[str(gpu_id) for gpu_id in normalized_gpu_ids],
    ]
    if load_from and resume_from:
        raise SystemExit('Use either --load-from or --resume-from, not both.')
    cfg_options = []
    if epochs is not None:
        decay_epoch = schedule_decay_epoch(epochs)
        cfg_options.append('total_epochs={}'.format(epochs))
        if decay_epoch is not None:
            cfg_options.append('lr_config.step=[{}]'.format(decay_epoch))
    if load_from:
        load_path = Path(load_from).expanduser().resolve()
        if not load_path.is_file():
            raise SystemExit('Pretrained checkpoint not found: {}'.format(load_path))
        cfg_options.append('load_from={}'.format(load_path))
    if resume_from:
        resume_path = Path(resume_from).expanduser().resolve()
        if not resume_path.is_file():
            raise SystemExit('Resume checkpoint not found: {}'.format(resume_path))
        command.extend(['--resume-from', str(resume_path)])
    if no_validate:
        command.append('--no-validate')
    if cfg_options:
        command.extend(['--cfg-options', *cfg_options])
    if seed is not None:
        command.extend(['--seed', str(seed)])
    if deterministic:
        command.append('--deterministic')
    subprocess.run(command, cwd=str(LD_ROOT), env=env, check=True)


def run_test(config, checkpoint, env, output=None, log_file=None):
    test_tool = LD_ROOT / 'tools' / 'test.py'
    require_ld_checkout()
    config_path = Path(config).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit('Config not found: {}'.format(config_path))
    if not checkpoint_path.is_file():
        raise SystemExit('Checkpoint not found: {}'.format(checkpoint_path))
    command = [
        sys.executable,
        str(test_tool),
        str(config_path),
        str(checkpoint_path),
        '--eval', 'mAP',
    ]
    if output:
        command.extend(['--out', str(Path(output).expanduser().resolve())])
    if log_file is None:
        log_file = checkpoint_path.parent / '{}_eval.log'.format(config_path.stem)
    log_path = Path(log_file).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(
            command,
            cwd=str(LD_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end='')
            stream.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
