from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FrameRecord:
    sequence_id: str
    frame_index: int
    image_path: str
    pedestrian_id: str
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    center_x: float
    center_y: float
    ego_speed: float
    heading: float
    crosswalk_distance: float
    curb_distance: float
    ttc: float
    intent: int
    mask_path: str | None = None
    crosswalk_mask_path: str | None = None
