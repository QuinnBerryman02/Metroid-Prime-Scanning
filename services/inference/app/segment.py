"""
Promptable segmentation for the object under the reticle.

Two backends:
  * "sam2"   - SAM 2.1, box-prompted. Best masks, runs on CPU (~2.5 s).
  * "yolo11" - YOLO11n-seg, class-aware, runs on the Intel iGPU via OpenVINO
               (~0.1 s). We pick the detection whose box best matches the aim.

Only the aimed object is segmented, so the cost is paid once per keyframe.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("inference.segmenter")

INTEL_DEVICES = {"xpu", "gpu", "npu", "intel:gpu", "intel:npu"}


class Segmenter:
    def __init__(
        self,
        backend: str = "sam2",
        sam_model: str = "sam2.1_t.pt",
        yolo_model: str = "yolo11n-seg.pt",
        device: str = "cpu",
        export: bool = False,
        imgsz: int = 640,
        conf: float = 0.25,
    ) -> None:
        self.backend = backend
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.sam = None
        self.yolo = None

        if backend == "yolo11":
            from ultralytics import YOLO

            path = Path(yolo_model)
            model_arg = str(path) if path.exists() else path.name
            if export and device in INTEL_DEVICES:
                ov_dir = path.with_name(f"{path.stem}_openvino_model")
                if not ov_dir.exists():
                    log.info("exporting %s -> OpenVINO (one-off)…", model_arg)
                    ov_dir = Path(YOLO(model_arg).export(format="openvino", half=False))
                model_arg = str(ov_dir)
            log.info("loading YOLO11-seg %s on %s", model_arg, device)
            self.yolo = YOLO(model_arg, task="segment")
        else:
            from ultralytics import SAM

            path = Path(sam_model)
            model_arg = str(path) if path.exists() else path.name
            log.info("loading SAM %s on %s", model_arg, device)
            self.sam = SAM(model_arg)

    def warmup(self, size: int = 640) -> None:
        """One dummy call so the first real request isn't a cold OpenVINO compile."""
        dummy = np.zeros((size, size, 3), dtype=np.uint8)
        self.mask_from_box(dummy, [16, 16, size - 16, size - 16])

    def mask_from_box(self, image: np.ndarray, box: list[float]) -> np.ndarray | None:
        """image: HxWx3 RGB uint8. box: [x1,y1,x2,y2]. Returns HxW bool or None."""
        if self.backend == "yolo11":
            return self._yolo_mask(image, box)
        return self._sam_mask(image, box)

    def _sam_mask(self, image: np.ndarray, box: list[float]) -> np.ndarray | None:
        result = self.sam(image, bboxes=[box], verbose=False, device=self.device)[0]
        masks = masks_from_result(result, image.shape[:2])
        return max(masks, key=lambda m: int(m.sum())) if masks else None

    def _yolo_mask(self, image: np.ndarray, box: list[float]) -> np.ndarray | None:
        result = self.yolo(
            image, imgsz=self.imgsz, conf=self.conf, verbose=False, device=self.device
        )[0]
        if result.masks is None or result.boxes is None:
            return None
        masks = masks_from_result(result, image.shape[:2])
        if not masks:
            return None
        idx = _best_box(box, result.boxes.xyxy.detach().cpu().numpy())
        return masks[idx] if idx is not None else None


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


def _best_box(box: list[float], boxes: np.ndarray) -> int | None:
    """Index of the box with the highest IoU against `box` (None if no overlap)."""
    if len(boxes) == 0:
        return None
    x1, y1, x2, y2 = box
    area = max((x2 - x1) * (y2 - y1), 1e-6)
    best, best_iou = None, 0.0
    for i, b in enumerate(boxes):
        ix = max(0.0, min(x2, b[2]) - max(x1, b[0]))
        iy = max(0.0, min(y2, b[3]) - max(y1, b[1]))
        inter = ix * iy
        union = area + (b[2] - b[0]) * (b[3] - b[1]) - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best, best_iou = i, iou
    return best
