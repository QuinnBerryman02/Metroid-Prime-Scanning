"""
YOLO object detector, OpenVINO-accelerated when targeting an Intel device.

Wraps Ultralytics so we get free letterboxing + NMS. On first run against an
Intel accelerator it exports the `.pt` to an OpenVINO IR (a few seconds) and
reuses it afterwards.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .schemas import Detection

log = logging.getLogger("inference.detector")

INTEL_DEVICES = {"xpu", "gpu", "npu", "intel:gpu", "intel:npu"}


class Detector:
    def __init__(
        self,
        model: str,
        device: str = "xpu",
        imgsz: int = 640,
        conf: float = 0.25,
        export_openvino: bool = True,
    ) -> None:
        from ultralytics import YOLO

        self.device = device
        self.imgsz = imgsz
        self.conf = conf

        path = Path(model)
        model_arg = str(path) if path.exists() else path.name
        if export_openvino and device in INTEL_DEVICES:
            ov_dir = path.with_name(f"{path.stem}_openvino_model")
            if not ov_dir.exists():
                log.info("exporting %s -> OpenVINO (one-off)…", model_arg)
                ov_dir = Path(YOLO(model_arg).export(format="openvino", half=False))
            model_arg = str(ov_dir)

        log.info("loading model %s on device %s", model_arg, device)
        self.model = YOLO(model_arg, task="detect")
        self.names = self.model.names

        # First inference triggers the OpenVINO compile (seconds on GPU). Absorb
        # it here so the first real /scan request is fast.
        t0 = time.perf_counter()
        self.detect(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))
        log.info("warmed up in %.0f ms", (time.perf_counter() - t0) * 1000)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """image: HxWx3 RGB uint8. Returns boxes in pixel coords."""
        result = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            cls = int(box.cls)
            detections.append(
                Detection(
                    classId=cls,
                    label=result.names[cls],
                    score=float(box.conf),
                    box=[x1, y1, x2, y2],
                )
            )
        # biggest first (most likely what the user cares about)
        detections.sort(
            key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]),
            reverse=True,
        )
        return detections
