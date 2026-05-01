# Cross-Intent Forecast

Production-oriented repository for pedestrian crossing-intent classification and 2-second multimodal trajectory prediction for slow-speed autonomous delivery vehicles.

## Features

- ConvNeXt + FPN multi-scale encoders for scene and pedestrian crops
- Temporal transformer + cross-attention interaction fusion
- Binary intent head, multimodal trajectory decoder, and aleatoric uncertainty head
- Joint loss with focal BCE, minADE/minFDE, heatmap KL, and uncertainty-weighted balancing
- PyTorch Lightning training with AMP, DDP-ready config, EMA, SWA, cosine warmup schedule, and curriculum stages
- Hydra configuration, Weights & Biases logging, ONNX export, TensorRT conversion support, and latency benchmarking
- Explainability via Grad-CAM and failure-case analysis
- Tests covering data, model, losses, metrics, inference, and export

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Data Format

Expected annotation columns:

`sequence_id, frame_index, image_path, pedestrian_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, center_x, center_y, ego_speed, heading, crosswalk_distance, curb_distance, ttc, intent`

Optional columns:

`mask_path, crosswalk_mask_path`

Each row represents one pedestrian track observation at one frame. The loader assembles history/future windows per track and anchor frame.

## Train

```bash
python scripts/train.py
```

Override config values with Hydra, for example:

```bash
python scripts/train.py training.max_epochs=30 data.annotations_file=/path/to/annotations.csv
```

## Validate

```bash
python scripts/validate.py
```

## Infer

```bash
python scripts/infer.py
```

## Export

```bash
python scripts/export.py
```

TensorRT engine creation:

```bash
trtexec --onnx=artifacts/model.onnx --saveEngine=artifacts/model.engine --fp16
```

## Benchmark

```bash
python scripts/benchmark.py
```

## Explainability

```bash
python scripts/explain.py
python scripts/analyze_failures.py
```

## Checkpoint Averaging

```bash
python scripts/average_checkpoints.py --checkpoints ckpt1.ckpt ckpt2.ckpt --output artifacts/averaged.ckpt
```

## Docker

```bash
docker compose up --build
```
