"""Service configuration (env-overridable)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # services/inference

# Path to the YOLO weights (.pt) - the OpenVINO export is derived from it.
MODEL_PATH = os.environ.get("MODEL_PATH", str(ROOT / "yolo11n.pt"))

# Ultralytics device: "xpu" (Intel iGPU), "npu", "intel:cpu", or "cpu".
DEVICE = os.environ.get("OV_DEVICE", "xpu")

# Export to OpenVINO on startup if we're targeting an Intel accelerator.
EXPORT_OPENVINO = os.environ.get("EXPORT_OPENVINO", "1") == "1"

IMGSZ = int(os.environ.get("IMGSZ", "640"))
CONF = float(os.environ.get("CONF", "0.25"))
