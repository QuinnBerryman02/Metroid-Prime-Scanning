"""
The selectable methods behind the client's radial wheel.

Each method has a `kind`:
  * "prompt" - takes the reticle as a point prompt and returns one mask
               (SAM 1 / SAM 2).
  * "detect" - returns 1..N labelled masks for the whole frame
               (FastSAM, YOLO11-seg).
  * "depth"  - returns the depth map.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import config

log = logging.getLogger("inference.methods")

INTEL_DEVICES = {"xpu", "gpu", "npu", "intel:gpu", "intel:npu"}


@dataclass
class MaskItem:
    mask: np.ndarray
    label: str = "object"
    score: float = 1.0
    box: list[float] = field(default_factory=list)


# order = the order they appear around the wheel
METHOD_SPECS: list[dict] = [
    {"id": "sam", "label": "SAM1", "family": "sam", "kind": "prompt", "model": config.SAM_MODEL},
    {"id": "sam2", "label": "SAM2", "family": "sam", "kind": "prompt", "model": config.SAM2_MODEL},
    {"id": "fastsam", "label": "FastSAM", "family": "fastsam", "kind": "detect", "model": config.FASTSAM_MODEL},
    {"id": "yolo", "label": "YOLO", "family": "yolo", "kind": "detect", "model": config.YOLO_SEG_MODEL},
    {"id": "depth", "label": "DEPTH", "family": "depth", "kind": "depth"},
]

METHODS: dict[str, dict] = {m["id"]: m for m in METHOD_SPECS}


def device_for(spec: dict) -> str:
    """SAM/SAM2 are PyTorch-only; FastSAM/YOLO use the OpenVINO device."""
    if spec["family"] == "sam":
        return "cpu"
    return config.DEVICE


def describe(spec: dict) -> dict:
    model = spec.get("model")
    return {
        "id": spec["id"],
        "label": spec["label"],
        "kind": spec["kind"],
        "device": "GPU" if device_for(spec) in INTEL_DEVICES else "CPU",
        "available": spec["family"] == "depth" or model is None or Path(model).exists(),
    }


class Segmenter:
    """Wraps an Ultralytics model for one method spec."""

    def __init__(self, spec: dict, device: str) -> None:
        self.spec = spec
        self.family = spec["family"]
        self.kind = spec["kind"]
        self.device = device
        self.imgsz = spec.get("imgsz", config.IMGSZ)
        self.conf = spec.get("conf", config.CONF)
        self.model = self._load(spec["model"])

    def _load(self, model: str):
        path = Path(model)
        model_arg = str(path) if path.exists() else path.name
        if self.family == "sam":
            from ultralytics import SAM

            return SAM(model_arg)
        if self.family == "fastsam":
            from ultralytics import FastSAM

            return FastSAM(self._maybe_export(FastSAM, model_arg, path))
        if self.family == "yolo":
            from ultralytics import YOLO

            return YOLO(self._maybe_export(YOLO, model_arg, path), task="segment")
        raise ValueError(f"unknown family {self.family!r}")

    def _maybe_export(self, cls, model_arg: str, path: Path) -> str:
        if not (config.EXPORT_OPENVINO and self.device in INTEL_DEVICES and path.exists()):
            return model_arg
        ov_dir = path.with_name(f"{path.stem}_openvino_model")
        if not ov_dir.exists():
            log.info("exporting %s -> OpenVINO (one-off)…", model_arg)
            ov_dir = Path(cls(model_arg).export(format="openvino", half=False))
        return str(ov_dir)

    def warmup(self) -> None:
        self.run(np.zeros((640, 640, 3), dtype=np.uint8), prompt=(320, 320))

    def run(self, image: np.ndarray, prompt=None) -> list[MaskItem]:
        if self.kind == "prompt":
            return self._prompt(image, prompt or self._centre(image))
        return self._detect(image)

    @staticmethod
    def _centre(image: np.ndarray) -> tuple[int, int]:
        h, w = image.shape[:2]
        return w // 2, h // 2

    def _prompt(self, image: np.ndarray, prompt) -> list[MaskItem]:
        px, py = int(prompt[0]), int(prompt[1])
        result = self.model(
            image, points=[[px, py]], labels=[1], verbose=False, device=self.device
        )[0]
        items = []
        for mask in masks_from_result(result, image.shape[:2]):
            items.append(
                MaskItem(mask=mask, label="object", score=1.0, box=box_of(mask))
            )
        return items

    def _detect(self, image: np.ndarray) -> list[MaskItem]:
        result = self.model(
            image, imgsz=self.imgsz, conf=self.conf, verbose=False, device=self.device
        )[0]
        masks = masks_from_result(result, image.shape[:2])
        if not masks:
            return []
        boxes = result.boxes
        xyxy = boxes.xyxy.detach().cpu().numpy() if boxes is not None else []
        cls = boxes.cls.detach().cpu().numpy() if boxes is not None else []
        conf = boxes.conf.detach().cpu().numpy() if boxes is not None else []
        items = []
        for i, mask in enumerate(masks):
            label = result.names.get(int(cls[i]), "object") if i < len(cls) else "object"
            score = float(conf[i]) if i < len(conf) else 1.0
            box = [float(v) for v in xyxy[i]] if i < len(xyxy) else box_of(mask)
            items.append(MaskItem(mask=mask, label=label, score=score, box=box))
        return items


def masks_from_result(result, hw: tuple[int, int]) -> list[np.ndarray]:
    """Boolean HxW masks from an Ultralytics result, in ORIGINAL image pixels.

    We rasterise `masks.xy` (always original-pixel polygons) rather than resize
    `masks.data`, whose spatial size differs by backend: PyTorch returns the
    unpadded letterbox size, OpenVINO returns the padded square. Stretching the
    padded square back to the frame is what squished the YOLO/FastSAM masks.
    """
    if result.masks is None:
        return []
    out = []
    for poly in result.masks.xy:
        mask = np.zeros(hw, dtype=np.uint8)
        pts = np.asarray(poly, dtype=np.float32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
        out.append(mask.astype(bool))
    return out


def box_of(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
