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


def build_env(data_root=None, teacher_checkpoint=None,
              samples_per_gpu=1, workers_per_gpu=2):
    env = os.environ.copy()
    env['VOC_DATA_ROOT'] = str(Path(data_root or env.get(
        'VOC_DATA_ROOT', DEFAULT_VOC_ROOT)).expanduser().resolve())
    env['SAMPLES_PER_GPU'] = str(samples_per_gpu)
    env['WORKERS_PER_GPU'] = str(workers_per_gpu)
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
    command = [
        sys.executable,
        str(train_tool),
        str(Path(config).resolve()),
        '--work-dir', str(Path(work_dir).expanduser().resolve()),
        '--gpu-ids', str(gpu_ids),
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
