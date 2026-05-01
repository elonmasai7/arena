from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> tuple[Path, Path]:
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for sequence_id in ["seq1", "seq2"]:
        for frame_index in range(14):
            image = np.full((128, 128, 3), fill_value=frame_index * 10 % 255, dtype=np.uint8)
            image_path = image_dir / f"{sequence_id}_{frame_index}.png"
            cv2.imwrite(str(image_path), image)
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "frame_index": frame_index,
                    "image_path": str(image_path),
                    "pedestrian_id": "ped1",
                    "bbox_x1": 30,
                    "bbox_y1": 40,
                    "bbox_x2": 70,
                    "bbox_y2": 100,
                    "center_x": 50 + frame_index,
                    "center_y": 60 + frame_index,
                    "ego_speed": 3.0,
                    "heading": 0.2,
                    "crosswalk_distance": 12.0,
                    "curb_distance": 8.0,
                    "ttc": 4.0,
                    "intent": 1 if sequence_id == "seq1" else 0,
                }
            )
    annotation_path = tmp_path / "annotations.csv"
    pd.DataFrame(rows).to_csv(annotation_path, index=False)
    return tmp_path, annotation_path
