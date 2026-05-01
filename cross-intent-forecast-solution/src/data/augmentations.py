from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_augmentations(height: int, width: int) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(height=height, width=width, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            A.MotionBlur(p=0.2),
            A.GaussNoise(p=0.2),
            A.RandomFog(p=0.15),
            A.RandomRain(p=0.15, blur_value=3),
            A.RandomBrightnessContrast(p=0.2),
            A.CoarseDropout(
                max_holes=5,
                max_height=max(8, height // 8),
                max_width=max(8, width // 8),
                p=0.25,
            ),
            A.Affine(scale=(0.95, 1.05), translate_percent=0.04, rotate=(-5, 5), shear=(-3, 3), p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def build_eval_augmentations(height: int, width: int) -> A.Compose:
    return A.Compose([A.Resize(height=height, width=width), A.Normalize(), ToTensorV2()])
