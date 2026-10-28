import os
import runpy as _runpy  # noqa: E402


_base_ = '../../LD/configs/ld/ld_r18_gflv1_r101_fpn_voc_1x.py'

TEACHER_CHECKPOINT = os.environ.get(
    'TEACHER_CHECKPOINT',
    '{{ fileDirname }}/../runs/teacher_r101/epoch_16.pth')

model = dict(
    teacher_config='{{ fileDirname }}/gfl_r101_voc_teacher.py',
    teacher_ckpt=TEACHER_CHECKPOINT,
    bbox_head=dict(
        loss_kd=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=10, T=2),
        loss_ld=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=10),
        loss_ld_vlr=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=10),
    ))

globals().update(_runpy.run_path(
    '{{ fileDirname }}/_base_/voc_dataset.py')['build_voc_overrides'](
        '{{ fileDirname }}/../..'))
del _runpy
