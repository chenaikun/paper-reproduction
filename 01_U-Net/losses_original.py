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

    def create_weight_map(self, instance_mask, binary_mask=None):
        """
        instance_mask: [B, H, W]
        binary_mask:   [B, H, W] — 0/1 二值掩膜，用于计算类别平衡权重 w_c(x)
                        如果为 None，则从 instance_mask 推导 (>0 为前景)

        return: weight_map [B, H, W]

        论文公式: w(x) = w_c(x) + w0 * exp(-(d1+d2)^2 / (2*sigma^2))
        其中 w_c(x) 是类别频率平衡权重:
            w_c(fg) = N_bg / N_total
            w_c(bg) = N_fg / N_total
        """

        device = instance_mask.device

        instance_mask_np = (
            instance_mask
            .detach()
            .cpu()
            .numpy()
        )

        if binary_mask is None:
            binary_mask_np = (instance_mask_np > 0).astype(np.int64)
        else:
            binary_mask_np = (
                binary_mask
                .detach()
                .cpu()
                .numpy()
            )

        weight_maps = []

        for i, mask in enumerate(instance_mask_np):

            bmask = binary_mask_np[i]

            # ---- 类别平衡权重 w_c(x) ----
            n_total = bmask.size
            n_fg = bmask.sum()
            n_bg = n_total - n_fg

            # w_c(fg) = N_bg / N_total, w_c(bg) = N_fg / N_total
            class_weight = np.where(
                bmask > 0,
                n_bg / n_total,
                n_fg / n_total
            ).astype(np.float32)

            # ---- 边界权重 ----
            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids != 0]

            if len(instance_ids) == 0:
                weight_maps.append(class_weight)
                continue

            distance_maps = []

            for instance_id in instance_ids:
                cell = (mask == instance_id)
                distance = ndimage.distance_transform_edt(~cell)
                distance_maps.append(distance)

            distance_maps = np.stack(distance_maps, axis=0)
            distance_maps.sort(axis=0)

            d1 = distance_maps[0]
            d2 = distance_maps[1] if distance_maps.shape[0] >= 2 else d1

            boundary_weight = (
                self.w0
                * np.exp(
                    -((d1 + d2) ** 2)
                    / (2 * self.sigma ** 2)
                )
            )

            # w(x) = w_c(x) + boundary_weight
            weight_map = class_weight + boundary_weight

            weight_maps.append(
                weight_map.astype(np.float32)
            )

        weight_maps = np.stack(weight_maps, axis=0)

        return torch.from_numpy(weight_maps).to(device)

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

        # spatial weight map (含类别平衡 w_c + 边界权重)
        weight_map = self.create_weight_map(
            instance_mask,
            binary_mask=target
        )

        # weighted CE
        loss = (
            ce * weight_map
        ).mean()

        return loss