_base_ = ['./ld_r18_voc.py']

model = dict(
    bbox_head=dict(
        loss_kd=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=10, T=2),
        loss_ld=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=0, T=10),
        loss_ld_vlr=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=0, T=10),
    ))
