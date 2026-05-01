from __future__ import annotations

from dataclasses import dataclass

import lightning as L
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import CrossingIntentDataset
from .parser import build_sequence_index, compute_class_weights, load_annotations, train_val_split


@dataclass(slots=True)
class DataConfig:
    root_dir: str
    annotations_file: str
    cache_dir: str
    image_size: tuple[int, int]
    history_frames: int
    future_steps: int
    future_dt: float
    num_workers: int
    batch_size: int
    val_batch_size: int
    train_fraction: float
    use_cache: bool
    weighted_sampler: bool
    scene_crop_scale: float
    ped_crop_size: tuple[int, int]


class CrossingDataModule(L.LightningDataModule):
    def __init__(self, cfg: DataConfig, heatmap_size: tuple[int, int]) -> None:
        super().__init__()
        self.cfg = cfg
        self.heatmap_size = heatmap_size
        self.train_dataset: CrossingIntentDataset | None = None
        self.val_dataset: CrossingIntentDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        annotations = load_annotations(self.cfg.annotations_file)
        train_df, val_df = train_val_split(annotations, self.cfg.train_fraction)
        train_samples = build_sequence_index(train_df, self.cfg.history_frames, self.cfg.future_steps)
        val_samples = build_sequence_index(val_df, self.cfg.history_frames, self.cfg.future_steps)
        cache_dir = self.cfg.cache_dir if self.cfg.use_cache else None
        self.train_dataset = CrossingIntentDataset(
            samples=train_samples,
            image_size=self.cfg.image_size,
            future_steps=self.cfg.future_steps,
            future_dt=self.cfg.future_dt,
            heatmap_size=self.heatmap_size,
            scene_crop_scale=self.cfg.scene_crop_scale,
            ped_crop_size=self.cfg.ped_crop_size,
            is_train=True,
            cache_dir=cache_dir,
        )
        self.val_dataset = CrossingIntentDataset(
            samples=val_samples,
            image_size=self.cfg.image_size,
            future_steps=self.cfg.future_steps,
            future_dt=self.cfg.future_dt,
            heatmap_size=self.heatmap_size,
            scene_crop_scale=self.cfg.scene_crop_scale,
            ped_crop_size=self.cfg.ped_crop_size,
            is_train=False,
            cache_dir=cache_dir,
        )
        self.class_weights = compute_class_weights(train_df)

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        assert self.train_dataset is not None
        sampler = None
        shuffle = True
        if self.cfg.weighted_sampler:
            weights = [float(sample["group"].iloc[int(sample["anchor"])]["intent"]) for sample in self.train_dataset.samples]
            tensor_weights = torch.tensor(
                [self.class_weights.get(int(weight), 1.0) for weight in weights],
                dtype=torch.double,
            )
            sampler = WeightedRandomSampler(tensor_weights, len(tensor_weights), replacement=True)
            shuffle = False
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            pin_memory=True,
        )
