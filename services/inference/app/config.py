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
# Which segmenter to use for the aimed object:
#   "sam2"   - SAM 2.1 (best masks, CPU, ~2.5 s)
#   "yolo11" - YOLO11n-seg (class-aware, OpenVINO iGPU, ~0.1 s)
SEGMENTER = os.environ.get("SEGMENTER", "sam2")

# SAM weights (a local path wins over a bare name, which Ultralytics downloads).
SAM_MODEL = os.environ.get("SAM_MODEL", str(ROOT / "notebooks" / "sam2.1_t.pt"))
SAM_DEVICE = os.environ.get("SAM_DEVICE", "cpu")

# YOLO11 segmentation weights for the "yolo11" segmenter (runs on DEVICE).
YOLO_SEG_MODEL = os.environ.get("YOLO_SEG_MODEL", str(ROOT / "yolo11n-seg.pt"))

# Depth Anything V2 (exported to OpenVINO on first run).
DEPTH_MODEL = os.environ.get(
    "DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf"
)
DEPTH_DEVICE = os.environ.get("DEPTH_DEVICE", "GPU")
DEPTH_IR = os.environ.get(
    "DEPTH_IR",
    str(ROOT / "depth_anything_v2_small_openvino_model" / "depth_anything_v2_small.xml"),
)

# Horizontal field of view used to build intrinsics (overridable per request).
FOV_DEG = float(os.environ.get("FOV_DEG", "60"))

# Toggle the extra models. Detection still works with both off.
SEGMENT_ENABLED = os.environ.get("SEGMENT", "1") == "1"
DEPTH_ENABLED = os.environ.get("DEPTH", "1") == "1"
