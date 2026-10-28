# Exactly one official base; apply runtime data overrides in this child.
_base_ = '../../LD/configs/gfl/gfl_r18_fpn1x_voc.py'

import runpy as _runpy

globals().update(_runpy.run_path(
    '{{ fileDirname }}/_base_/voc_dataset.py')['build_voc_overrides'](
        '{{ fileDirname }}/../..'))
del _runpy
