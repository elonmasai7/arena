from __future__ import annotations

import subprocess
from pathlib import Path


def export_tensorrt(onnx_path: str | Path, engine_path: str | Path) -> Path:
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
    ]
    subprocess.run(command, check=True)
    return engine_path
