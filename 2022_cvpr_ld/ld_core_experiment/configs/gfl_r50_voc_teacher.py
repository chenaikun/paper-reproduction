_base_ = ['./gfl_r18_voc_baseline.py']

# Teacher used only to provide detached classification/localization targets.
# It is intentionally kept in the independent core-experiment directory.
model = dict(
    backbone=dict(
        depth=50,
        init_cfg=None),
    neck=dict(in_channels=[256, 512, 1024, 2048]),
    bbox_head=dict(num_classes=20))

# Keep the same data split, image size, optimizer, and schedule as the student
# so that the teacher/student comparison isolates model capacity first.
