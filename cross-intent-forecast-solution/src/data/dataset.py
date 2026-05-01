from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augmentations import build_eval_augmentations, build_train_augmentations
from .cache import DiskCache
from .targets import compute_heading_velocity, generate_future_heatmaps, generate_future_waypoints


class CrossingIntentDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        image_size: tuple[int, int],
        future_steps: int,
        future_dt: float,
        heatmap_size: tuple[int, int],
        scene_crop_scale: float,
        ped_crop_size: tuple[int, int],
        is_train: bool,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.samples = samples
        self.height, self.width = image_size
        self.future_steps = future_steps
        self.future_dt = future_dt
        self.heatmap_size = heatmap_size
        self.scene_crop_scale = scene_crop_scale
        self.ped_crop_size = ped_crop_size
        self.is_train = is_train
        self.augment = (
            build_train_augmentations(self.height, self.width)
            if is_train
            else build_eval_augmentations(self.height, self.width)
        )
        self.cache = DiskCache(cache_dir) if cache_dir else None

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_rgb(path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _crop_pedestrian(self, image: np.ndarray, row: pd.Series) -> np.ndarray:
        x1, y1, x2, y2 = [int(row[key]) for key in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]]
        crop = image[max(0, y1): max(y1 + 1, y2), max(0, x1): max(x1 + 1, x2)]
        if crop.size == 0:
            crop = image
        return cv2.resize(crop, (self.ped_crop_size[1], self.ped_crop_size[0]))

    def _crop_scene(self, image: np.ndarray, row: pd.Series) -> np.ndarray:
        x1, y1, x2, y2 = [float(row[key]) for key in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        width = (x2 - x1) * self.scene_crop_scale
        height = (y2 - y1) * self.scene_crop_scale
        left = int(max(0, cx - width))
        right = int(min(image.shape[1], cx + width))
        top = int(max(0, cy - height))
        bottom = int(min(image.shape[0], cy + height))
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            crop = image
        return crop

    @staticmethod
    def _load_mask(mask_path: str | None, image_shape: tuple[int, int]) -> np.ndarray:
        if not mask_path:
            return np.zeros(image_shape, dtype=np.uint8)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros(image_shape, dtype=np.uint8)
        return mask

    def _make_sample(self, index: int) -> dict[str, torch.Tensor]:
        spec = self.samples[index]
        group = spec["group"]
        anchor = int(spec["anchor"])
        cache_key = f"{spec['sequence_id']}-{spec['pedestrian_id']}-{anchor}-{self.is_train}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        history = group.iloc[: anchor + 1]
        future = group.iloc[anchor + 1 : anchor + 1 + self.future_steps]
        anchor_row = history.iloc[-1]

        image = self._read_rgb(str(anchor_row["image_path"]))
        h, w = image.shape[:2]
        scene_crop = self._crop_scene(image, anchor_row)
        ped_crop = self._crop_pedestrian(image, anchor_row)
        scene_aug = self.augment(image=scene_crop)
        ped_aug = self.augment(image=ped_crop)

        history_points = history[["center_x", "center_y"]].to_numpy(dtype=np.float32)
        waypoints = generate_future_waypoints(
            centers=list(future[["center_x", "center_y"]].itertuples(index=False, name=None)),
            future_steps=self.future_steps,
            image_width=w,
            image_height=h,
        )
        heatmaps = generate_future_heatmaps(waypoints_norm=waypoints, heatmap_size=self.heatmap_size)
        vx, vy, derived_heading = compute_heading_velocity(history_points, dt=self.future_dt)
        scene_mask = self._load_mask(anchor_row.get("mask_path"), (h, w))
        crosswalk_mask = self._load_mask(anchor_row.get("crosswalk_mask_path"), (h, w))

        features = np.array(
            [
                vx / max(w, 1),
                vy / max(h, 1),
                float(anchor_row["heading"]),
                derived_heading,
                float(anchor_row["curb_distance"]) / max(w, 1),
                float(anchor_row["crosswalk_distance"]) / max(w, 1),
                float(anchor_row["ego_speed"]) / 20.0,
                float(anchor_row["ttc"]) / 10.0,
                float(scene_mask.mean()) / 255.0,
                float(crosswalk_mask.mean()) / 255.0,
            ],
            dtype=np.float32,
        )

        result = {
            "scene": scene_aug["image"].float(),
            "pedestrian": ped_aug["image"].float(),
            "interaction": torch.from_numpy(features),
            "intent": torch.tensor(float(anchor_row["intent"]), dtype=torch.float32),
            "waypoints": torch.from_numpy(waypoints),
            "heatmaps": torch.from_numpy(heatmaps),
            "sample_weight": torch.tensor(2.0 if int(anchor_row["intent"]) == 1 else 1.0, dtype=torch.float32),
        }
        if self.cache:
            self.cache.set(cache_key, result)
        return result

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._make_sample(index)
