import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import ndimage


class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self, w0=10.0, sigma=5.0):
        super().__init__()

        self.w0 = w0
        self.sigma = sigma

    def create_weight_map(self, instance_mask):

        """
        instance_mask:
            [B, H, W]

        return:
            weight_map:
            [B, H, W]
        """

        device = instance_mask.device

        instance_mask_np = (
            instance_mask
            .detach()
            .cpu()
            .numpy()
        )

        weight_maps = []

        for mask in instance_mask_np:

            # 找到所有实例
            instance_ids = np.unique(mask)

            # 去掉 background
            instance_ids = instance_ids[
                instance_ids != 0
            ]

            # 如果没有细胞
            if len(instance_ids) == 0:

                weight_map = np.ones_like(
                    mask,
                    dtype=np.float32
                )

                weight_maps.append(weight_map)
                continue

            distance_maps = []

            # --------------------------------
            # 计算每一个 cell 的距离图
            # --------------------------------

            for instance_id in instance_ids:

                cell = (
                    mask == instance_id
                )

                distance = ndimage.distance_transform_edt(
                    ~cell
                )

                distance_maps.append(distance)

            distance_maps = np.stack(
                distance_maps,
                axis=0
            )

            # --------------------------------
            # 每个像素找到最近的两个 cell
            # --------------------------------

            distance_maps.sort(axis=0)

            d1 = distance_maps[0]

            if distance_maps.shape[0] >= 2:
                d2 = distance_maps[1]
            else:
                d2 = d1

            # --------------------------------
            # U-Net boundary weight
            # --------------------------------

            boundary_weight = (
                self.w0
                * np.exp(
                    -(
                        (d1 + d2) ** 2
                    )
                    / (
                        2 * self.sigma ** 2
                    )
                )
            )

            # 基础权重暂设为 1
            weight_map = (
                1.0 + boundary_weight
            )

            weight_maps.append(
                weight_map.astype(
                    np.float32
                )
            )

        weight_maps = np.stack(
            weight_maps,
            axis=0
        )

        return torch.from_numpy(
            weight_maps
        ).to(device)

    def forward(
        self,
        logits,
        target,
        instance_mask
    ):

        # 每个像素自己的 CE
        ce = F.cross_entropy(
            logits,
            target,
            reduction="none"
        )

        # spatial weight map
        weight_map = self.create_weight_map(
            instance_mask
        )

        # weighted CE
        loss = (
            ce * weight_map
        ).mean()

        return loss