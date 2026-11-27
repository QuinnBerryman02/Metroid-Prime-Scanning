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
# Metric (indoor) checkpoint: outputs depth in metres directly, so the recon
# path needs no per-frame scale fit. See docs/METRIC_DEPTH.md.
DEPTH_MODEL = os.environ.get(
    "DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
)
# Focal length (px) the metric model was trained at. 0 = no correction.
# Metric depth scales with focal length, so if the phone's focal differs, the
# output is off by f/f0. Calibrate once against a known distance; don't fit.
DEPTH_FOCAL_CANON = float(os.environ.get("DEPTH_FOCAL_CANON", "0"))
DEPTH_DEVICE = os.environ.get("DEPTH_DEVICE", "GPU")
DEPTH_IR = os.environ.get(
    "DEPTH_IR",
    str(ROOT / "depth_anything_v2_metric_indoor_small_openvino_model" / "depth_anything_v2_metric_indoor_small.xml"),
)

# Horizontal field of view used to build intrinsics (overridable per request).
FOV_DEG = float(os.environ.get("FOV_DEG", "60"))

# --- dense reconstruction (MS3.1/MS3.2, docs/RECONSTRUCTION.md) -----------
# Fusion voxel size. 2 cm is the fidelity target but costs ~180 ms/frame on its
# own (services/inference/scripts/bench_voxel.py); 5 cm is ~55 ms. Start coarse.
RECON_VOXEL = float(os.environ.get("RECON_VOXEL", "0.05"))
RECON_CAPACITY = int(os.environ.get("RECON_CAPACITY", "3000000"))
# Unproject every Nth pixel of the depth map.
RECON_STRIDE = int(os.environ.get("RECON_STRIDE", "2"))
# Ignore depth outside this metric range (the depth model has no idea).
RECON_MIN_DEPTH = float(os.environ.get("RECON_MIN_DEPTH", "0.2"))
RECON_MAX_DEPTH = float(os.environ.get("RECON_MAX_DEPTH", "12.0"))
# Send a cloud snapshot to viewers every Nth keyframe (keeps the wire bounded).
RECON_CLOUD_EVERY = int(os.environ.get("RECON_CLOUD_EVERY", "4"))
RECON_CLOUD_VOXEL = float(os.environ.get("RECON_CLOUD_VOXEL", "0.12"))
RECON_CLOUD_MAX = int(os.environ.get("RECON_CLOUD_MAX", "30000"))
# The TSDF method (recon_methods "tsdf") is the only one with a surface.

# Record every keyframe (JPEG + full request JSON) to disk so a walk-around can
# be replayed offline with different scales / voxel sizes / methods. This is the
# tuning loop: on-device iteration is far too slow to explore a parameter space.
RECON_RECORD = os.environ.get("RECON_RECORD", "1") == "1"
RECON_DIR = Path(os.environ.get("RECON_DIR", str(ROOT / "recon_log")))
# Reject a plane fit whose residual is this large in metres: on real data the
# bad fits show rms ~1-2 m over a ~1.6 m spread, i.e. they are noise.
RECON_SCALE_RMS_MAX = float(os.environ.get("RECON_SCALE_RMS_MAX", "0.25"))
