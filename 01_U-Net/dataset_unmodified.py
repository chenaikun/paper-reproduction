"""Dataset loader for the downloaded 30/30 U-Net PNG dataset.

The directory layout is:

    data/unmodified-data/
    ├── train/{imgs,labels}
    └── test/{imgs,labels}

The labels are binary PNGs with values 0 and 255.  The model still receives
the full 512x512 image, while the target is center-cropped to 324x324 to
match the valid-convolution U-Net output.
"""

from pathlib import Path
import random

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class UnmodifiedSegmentationDataset(Dataset):
    """Read the downloaded train/validation/test image-mask pairs.

    The provided ``train`` directory contains 30 labelled images.  To keep
    the test set untouched during model selection, ``split='train'`` and
    ``split='val'`` are deterministic partitions of those 30 images.  The
    provided ``test`` directory is available only through ``split='test'``.
    """

    def __init__(
        self,
        data_root="./data/unmodified-data",
        split="train",
        target_size=324,
        val_fraction=0.2,
        seed=42,
        elastic_deform=False,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1")

        self.data_root = Path(data_root)
        self.split = split
        self.target_size = target_size
        self.is_train = split == "train"
        self.elastic_deform = elastic_deform and self.is_train

        source_split = "test" if split == "test" else "train"
        image_dir = self.data_root / source_split / "imgs"
        label_dir = self.data_root / source_split / "labels"

        image_paths = sorted(image_dir.glob("*.png"))
        pairs = []
        for image_path in image_paths:
            label_path = label_dir / image_path.name
            if not label_path.exists():
                raise FileNotFoundError(
                    f"Missing label for {image_path}: {label_path}"
                )
            pairs.append((image_path, label_path))

        if not pairs:
            raise FileNotFoundError(f"No PNG image pairs found in {image_dir}")

        if split in {"train", "val"}:
            indices = list(range(len(pairs)))
            random.Random(seed).shuffle(indices)
            n_val = max(1, round(len(indices) * val_fraction))
            val_indices = set(indices[:n_val])

            if split == "val":
                pairs = [pair for i, pair in enumerate(pairs) if i in val_indices]
            else:
                pairs = [pair for i, pair in enumerate(pairs) if i not in val_indices]

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, label_path = self.pairs[index]
        image = Image.open(image_path).convert("L")
        mask = Image.open(label_path).convert("L")

        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                mask = TF.rotate(
                    mask,
                    angle,
                    interpolation=TF.InterpolationMode.NEAREST,
                )

            if self.elastic_deform and random.random() > 0.5:
                image, mask = self._elastic_transform(image, mask)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.int64)

        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_array)
        mask_tensor = TF.center_crop(
            mask_tensor, [self.target_size, self.target_size]
        )

        return image_tensor, mask_tensor.long()

    @staticmethod
    def _elastic_transform(image, mask, alpha=40.0, sigma=6.0):
        """Apply one smooth random displacement field to image and mask."""
        image_array = np.asarray(image, dtype=np.float32)
        mask_array = np.asarray(mask, dtype=np.uint8)
        height, width = image_array.shape

        random_x = np.random.rand(height, width) * 2.0 - 1.0
        random_y = np.random.rand(height, width) * 2.0 - 1.0
        displacement_x = gaussian_filter(random_x, sigma=sigma) * alpha
        displacement_y = gaussian_filter(random_y, sigma=sigma) * alpha

        grid_y, grid_x = np.meshgrid(
            np.arange(height), np.arange(width), indexing="ij"
        )
        sample_y = np.clip(grid_y + displacement_y, 0, height - 1)
        sample_x = np.clip(grid_x + displacement_x, 0, width - 1)

        coordinates = [sample_y, sample_x]
        transformed_image = map_coordinates(
            image_array, coordinates, order=1, mode="nearest"
        )
        transformed_mask = map_coordinates(
            mask_array, coordinates, order=0, mode="nearest"
        )

        return (
            Image.fromarray(np.clip(transformed_image, 0, 255).astype(np.uint8)),
            Image.fromarray(transformed_mask.astype(np.uint8)),
        )
