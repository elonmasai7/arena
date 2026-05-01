from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "sequence_id",
    "frame_index",
    "image_path",
    "pedestrian_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "center_x",
    "center_y",
    "ego_speed",
    "heading",
    "crosswalk_distance",
    "curb_distance",
    "ttc",
    "intent",
}


def load_annotations(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix == ".parquet":
        df = pd.read_parquet(source)
    else:
        df = pd.read_csv(source)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.sort_values(["sequence_id", "pedestrian_id", "frame_index"]).reset_index(drop=True)
    return df


def build_sequence_index(
    annotations: pd.DataFrame,
    history_frames: int,
    future_steps: int,
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], pd.DataFrame] = {
        key: group.reset_index(drop=True)
        for key, group in annotations.groupby(["sequence_id", "pedestrian_id"], sort=False)
    }
    samples: list[dict[str, object]] = []
    for (sequence_id, pedestrian_id), group in groups.items():
        total = len(group)
        for anchor in range(history_frames - 1, total - future_steps):
            samples.append(
                {
                    "sequence_id": sequence_id,
                    "pedestrian_id": pedestrian_id,
                    "group": group,
                    "anchor": anchor,
                }
            )
    return samples


def train_val_split(
    annotations: pd.DataFrame,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seq_ids = annotations["sequence_id"].drop_duplicates().tolist()
    cutoff = max(1, int(len(seq_ids) * train_fraction))
    train_ids = set(seq_ids[:cutoff])
    train_df = annotations[annotations["sequence_id"].isin(train_ids)].reset_index(drop=True)
    val_df = annotations[~annotations["sequence_id"].isin(train_ids)].reset_index(drop=True)
    if val_df.empty:
        val_df = train_df.iloc[: max(1, len(train_df) // 5)].copy()
    return train_df, val_df.reset_index(drop=True)


def compute_class_weights(annotations: pd.DataFrame) -> dict[int, float]:
    counts = annotations["intent"].value_counts().to_dict()
    total = float(sum(counts.values()))
    return {
        int(label): total / max(1.0, float(count))
        for label, count in counts.items()
    }
