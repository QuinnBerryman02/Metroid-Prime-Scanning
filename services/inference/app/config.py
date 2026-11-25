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

# --- Stage B: segmentation + depth (lifting) ---
# SAM weights (a local path wins over a bare name, which Ultralytics downloads).
SAM_MODEL = os.environ.get("SAM_MODEL", str(ROOT / "notebooks" / "sam2.1_t.pt"))
SAM_DEVICE = os.environ.get("SAM_DEVICE", "cpu")

DEPTH_MODEL = os.environ.get(
    "DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf"
)
DEPTH_DEVICE = os.environ.get("DEPTH_DEVICE", "cpu")

# Horizontal field of view used to build intrinsics (overridable per request).
FOV_DEG = float(os.environ.get("FOV_DEG", "60"))

# Toggle the extra models. Detection still works with both off.
SEGMENT_ENABLED = os.environ.get("SEGMENT", "1") == "1"
DEPTH_ENABLED = os.environ.get("DEPTH", "1") == "1"
