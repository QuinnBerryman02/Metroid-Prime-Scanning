"""
Monocular depth estimation (Depth Anything V2) - the "z" for lifting.

Depth Anything V2 predicts *relative* depth, so the absolute scale is unknown.
That's fine: the phone recovers scale from an AR hit-test and multiplies
(see docs/LIFTING.md section 4).
"""
from __future__ import annotations

import logging

import numpy as np
from PIL import Image

log = logging.getLogger("inference.depth")

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._torch = torch
        self.device = device
        log.info("loading depth model %s on %s", model_id, device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
        )

    def predict(self, image: Image.Image) -> np.ndarray:
        """Return a float32 HxW depth map (relative units; larger = farther)."""
        torch = self._torch
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            predicted = self.model(**inputs).predicted_depth
            predicted = torch.nn.functional.interpolate(
                predicted.unsqueeze(1),
                size=(image.height, image.width),
                mode="bicubic",
                align_corners=False,
            )
        return predicted.squeeze().cpu().numpy().astype(np.float32)
