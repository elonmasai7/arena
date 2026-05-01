from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * resized / max(resized.max(), 1e-6)), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 0.65, colored, 0.35, 0)


def grad_cam(
    model: torch.nn.Module,
    scene: torch.Tensor,
    pedestrian: torch.Tensor,
    interaction: torch.Tensor,
) -> np.ndarray:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    target_layer = model.scene_backbone.output[-1][0]

    def forward_hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations.append(output)

    def backward_hook(_module: torch.nn.Module, grad_input: tuple[torch.Tensor, ...], grad_output: tuple[torch.Tensor, ...]) -> None:
        del grad_input
        gradients.append(grad_output[0])

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    outputs = model(scene, pedestrian, interaction)
    score = outputs["intent_prob"].sum()
    model.zero_grad(set_to_none=True)
    score.backward()
    handle_f.remove()
    handle_b.remove()
    activation = activations[-1]
    gradient = gradients[-1]
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1).relu()
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-6)
    return cam[0].detach().cpu().numpy()


def save_attention_map(attention: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.uint8(255 * attention / max(attention.max(), 1e-6))
    cv2.imwrite(str(path), image)
