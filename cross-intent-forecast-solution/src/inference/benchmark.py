from __future__ import annotations

import time

import torch

from .predictor import ForecastPredictor


def benchmark_predictor(
    predictor: ForecastPredictor,
    scene: torch.Tensor,
    pedestrian: torch.Tensor,
    interaction: torch.Tensor,
    iterations: int = 20,
) -> dict[str, float]:
    for _ in range(5):
        predictor.predict_batch(scene, pedestrian, interaction)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        predictor.predict_batch(scene, pedestrian, interaction)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    ms = (elapsed / iterations) * 1000.0
    return {"latency_ms": ms, "throughput_fps": 1000.0 / ms * scene.shape[0]}
