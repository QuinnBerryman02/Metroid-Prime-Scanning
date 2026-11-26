"""
Monocular depth estimation (Depth Anything V2) - the "z" for lifting.

Runs through OpenVINO on the Intel iGPU when an IR exists (it is exported on
first use). Depth Anything V2 predicts *relative* depth, so the phone recovers
absolute scale from an AR hit-test and multiplies (see docs/LIFTING.md s4).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger("inference.depth")

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

# The model is fully convolutional; we export it at one fixed size. The image
# is resized here and the depth map resized back, so any camera aspect works.
INPUT_SIZE = 518
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DepthEstimator:
    """OpenVINO backend when an IR is available, else PyTorch (CPU)."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str = "GPU",
        ir: str | Path | None = None,
        export: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.ir = Path(ir) if ir is not None else None
        self.size = INPUT_SIZE
        self._compiled = None
        self._output = None
        self._torch = None
        self._model = None

        if export and self.ir is not None and not self.ir.exists():
            try:
                self._export_openvino(self.ir)
            except Exception as e:  # noqa: BLE001
                log.warning("depth OpenVINO export failed (%s); using PyTorch", e)

        if self.ir is not None and self.ir.exists():
            try:
                self._load_openvino(self.ir)
                return
            except Exception as e:  # noqa: BLE001
                log.warning("depth OpenVINO load failed (%s); using PyTorch", e)

        self._load_torch()

    # -- backends ----------------------------------------------------------
    def _export_openvino(self, ir: Path) -> None:
        import openvino as ov
        import torch
        from transformers import AutoModelForDepthEstimation

        log.info("exporting depth model to OpenVINO (one-off)…")
        t0 = time.perf_counter()
        model = AutoModelForDepthEstimation.from_pretrained(self.model_id).eval()
        dummy = torch.randn(1, 3, self.size, self.size)
        onnx = ir.with_suffix(".onnx")
        onnx.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model,
            dummy,
            str(onnx),
            input_names=["pixel_values"],
            output_names=["predicted_depth"],
            opset_version=18,
            dynamo=True,
        )
        ov.save_model(ov.convert_model(str(onnx)), str(ir))
        onnx.unlink(missing_ok=True)
        log.info("depth exported in %.1fs -> %s", time.perf_counter() - t0, ir)

    def _load_openvino(self, ir: Path) -> None:
        import openvino as ov

        compiled = ov.Core().compile_model(str(ir), self.device.upper())
        self._compiled = compiled
        self._output = compiled.output(0)
        log.info("depth model: OpenVINO %s on %s", self.model_id, self.device)

    def _load_torch(self) -> None:
        import torch
        from transformers import AutoModelForDepthEstimation

        self._torch = torch
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id).eval()
        log.info("depth model: PyTorch on cpu")

    # -- inference ---------------------------------------------------------
    def _preprocess(self, image: Image.Image) -> np.ndarray:
        img = np.asarray(
            image.resize((self.size, self.size), Image.BILINEAR), dtype=np.float32
        )
        img = (img / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(img.transpose(2, 0, 1)[None])

    def predict(self, image: Image.Image) -> np.ndarray:
        """Raw model output as float32 HxW.

        NOTE: Depth Anything V2 predicts *inverse* depth (disparity), so here
        **larger = nearer**. Use `predict_depth()` when you need depth.
        """
        pixel_values = self._preprocess(image)
        if self._compiled is not None:
            depth = np.asarray(
                self._compiled([pixel_values])[self._output]
            ).squeeze()
        else:
            with self._torch.no_grad():
                tensor = self._torch.from_numpy(pixel_values)
                depth = self._model(tensor).predicted_depth.squeeze().numpy()
        depth = depth.astype(np.float32)
        if depth.shape != (image.height, image.width):
            depth = cv2.resize(
                depth, (image.width, image.height), interpolation=cv2.INTER_LINEAR
            )
        return depth

    def predict_depth(self, image: Image.Image) -> np.ndarray:
        """True depth (larger = farther), i.e. `1 / disparity`.

        Unprojecting the raw disparity would invert the object inside-out, so
        the lifting stage always goes through this.
        """
        return 1.0 / np.maximum(self.predict(image), 1e-6)
