"""Service configuration (env-overridable)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # services/inference


def _has_openvino_gpu() -> bool:
    try:
        import openvino as ov

        return "GPU" in ov.Core().available_devices
    except Exception:  # noqa: BLE001
        return False


def resolve_device() -> str:
    """Ultralytics device string: 'xpu' when the Intel iGPU is usable, else 'cpu'.

    `OV_DEVICE` forces a value (e.g. "cpu", "npu"); the default "auto" picks the
    iGPU when OpenVINO can see it.
    """
    explicit = os.environ.get("OV_DEVICE", "auto")
    if explicit.lower() not in ("", "auto"):
        return explicit
    return "xpu" if _has_openvino_gpu() else "cpu"


# Ultralytics/OpenVINO device: "xpu" (Intel iGPU) / "npu" / "cpu" / "auto".
DEVICE = resolve_device()

# Export YOLO-style models to OpenVINO when targeting an Intel accelerator.
EXPORT_OPENVINO = os.environ.get("EXPORT_OPENVINO", "1") == "1"

IMGSZ = int(os.environ.get("IMGSZ", "640"))
CONF = float(os.environ.get("CONF", "0.25"))
DETECT_MODEL = os.environ.get("MODEL_PATH", str(ROOT / "yolo11n.pt"))

# --- selectable methods (the radial wheel) -------------------------------
# Paths to the weights. Missing local files fall back to an Ultralytics
# download (bare name). SAM/SAM2 run on CPU (PyTorch); FastSAM/YOLO are
# OpenVINO-accelerated on the iGPU.
# MobileSAM - a SAM1 variant that's actually usable on CPU (sam_b.pt is ~15 s/scan).
SAM_MODEL = os.environ.get("SAM_MODEL", str(ROOT / "notebooks" / "mobile_sam.pt"))
SAM2_MODEL = os.environ.get("SAM2_MODEL", str(ROOT / "notebooks" / "sam2.1_t.pt"))
FASTSAM_MODEL = os.environ.get("FASTSAM_MODEL", str(ROOT / "notebooks" / "FastSAM-s.pt"))
YOLO_SEG_MODEL = os.environ.get("YOLO_SEG_MODEL", str(ROOT / "yolo11n-seg.pt"))

# --- depth (Depth Anything V2, exported to OpenVINO) ---------------------
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
