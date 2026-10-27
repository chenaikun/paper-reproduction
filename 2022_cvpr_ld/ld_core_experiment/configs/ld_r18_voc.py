import os
from pathlib import Path


_base_ = [
    '../../LD/configs/ld/ld_r18_gflv1_r101_fpn_voc_1x.py',
    './_base_/voc_dataset.py',
]

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
LD_ROOT = EXPERIMENT_ROOT.parent / 'LD'
TEACHER_CHECKPOINT = os.environ.get(
    'TEACHER_CHECKPOINT',
    str(EXPERIMENT_ROOT / 'runs' / 'teacher_r101' / 'epoch_16.pth'))

model = dict(
    teacher_config=str(LD_ROOT / 'configs' / 'gfl' / 'gfl_r101_fpn_voc.py'),
    teacher_ckpt=TEACHER_CHECKPOINT,
    bbox_head=dict(
        loss_kd=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=10, T=2),
        loss_ld=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=10),
        loss_ld_vlr=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=10),
    ))
