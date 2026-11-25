"""
Promptable segmentation (SAM 2.1) for the object under the reticle.

We only segment the *aimed* object (one box prompt), not every detection, so the
cost is paid once per keyframe.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("inference.segmenter")


class Segmenter:
    def __init__(self, model: str = "sam2.1_t.pt", device: str = "cpu") -> None:
        from ultralytics import SAM

        path = Path(model)
        # A local file wins; otherwise let Ultralytics resolve/download the name.
        self.model_arg = str(path) if path.exists() else path.name
        self.device = device
        log.info("loading SAM %s on %s", self.model_arg, device)
        self.model = SAM(self.model_arg)

    def mask_from_box(self, image: np.ndarray, box: list[float]) -> np.ndarray | None:
        """image: HxWx3 RGB uint8. box: [x1,y1,x2,y2]. Returns HxW bool or None."""
        result = self.model(image, bboxes=[box], verbose=False, device=self.device)[0]
        masks = masks_from_result(result, image.shape[:2])
        if not masks:
            return None
        return max(masks, key=lambda m: int(m.sum()))


def masks_from_result(result, hw: tuple[int, int]) -> list[np.ndarray]:
    """Pull boolean HxW masks out of an Ultralytics segmentation result."""
    if result.masks is None:
        return []
    data = result.masks.data.detach().cpu().numpy()
    if data.ndim == 2:
        data = data[None]
    out = []
    for m in data:
        if m.shape != hw:
            m = cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
        out.append(m > 0.5)
    return out
