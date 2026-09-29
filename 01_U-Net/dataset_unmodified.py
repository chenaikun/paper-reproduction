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
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1")

        self.data_root = Path(data_root)
        self.split = split
        self.target_size = target_size
        self.is_train = split == "train"

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

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.int64)

        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_array)
        mask_tensor = TF.center_crop(
            mask_tensor, [self.target_size, self.target_size]
        )

        return image_tensor, mask_tensor.long()
