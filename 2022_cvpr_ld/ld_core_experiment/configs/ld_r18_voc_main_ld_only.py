_base_ = ['./ld_r18_voc.py']

# Corresponds to the main-region-only setting in the paper's VOC Table A4.
model = dict(
    bbox_head=dict(
        loss_kd=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=0, T=2),
        loss_ld=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=10),
        loss_ld_vlr=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=0, T=10),
    ))
