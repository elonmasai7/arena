from __future__ import annotations

from src.data.datamodule import CrossingDataModule, DataConfig


def test_datamodule_loads_batches(synthetic_dataset: tuple[str, str]) -> None:
    root_dir, annotations = synthetic_dataset
    cfg = DataConfig(
        root_dir=str(root_dir),
        annotations_file=str(annotations),
        cache_dir=str(root_dir / "cache"),
        image_size=(64, 64),
        history_frames=4,
        future_steps=8,
        future_dt=0.25,
        num_workers=0,
        batch_size=2,
        val_batch_size=2,
        train_fraction=0.5,
        use_cache=True,
        weighted_sampler=True,
        scene_crop_scale=1.8,
        ped_crop_size=(64, 32),
    )
    dm = CrossingDataModule(cfg, heatmap_size=(16, 16))
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert batch["scene"].shape == (2, 3, 64, 64)
    assert batch["pedestrian"].shape == (2, 3, 64, 64)
    assert batch["waypoints"].shape == (2, 8, 2)
    assert batch["heatmaps"].shape == (2, 8, 16, 16)
