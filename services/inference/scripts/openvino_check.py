#!/usr/bin/env python
"""
OpenVINO sanity check + device benchmark.

Answers two questions:
  1. Which OpenVINO devices does this machine expose (CPU / GPU / NPU)?
  2. Does YOLO11n actually run on the iGPU, and how much faster than CPU?

Usage (from services/inference):
    .venv/Scripts/python.exe scripts/openvino_check.py
    .venv/Scripts/python.exe scripts/openvino_check.py --runs 30
"""
from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SAMPLES = ROOT / "samples"
SAMPLES.mkdir(exist_ok=True)


def print_devices():
    import openvino as ov

    core = ov.Core()
    print(f"OpenVINO {ov.__version__}")
    print(f"available_devices: {core.available_devices}")
    for d in core.available_devices:
        try:
            full = core.get_property(d, "FULL_DEVICE_NAME")
        except Exception:
            full = "?"
        print(f"  {d:<6} {full}")
    return core


def ensure_image() -> Path:
    path = SAMPLES / "bus.jpg"
    if not path.exists():
        print("downloading sample image…")
        urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", path)
    return path


def export_yolo() -> Path:
    from ultralytics import YOLO

    # Ultralytics writes `<model>_openvino_model/` next to the weights.
    existing = ROOT / "yolo11n_openvino_model"
    if existing.exists():
        return existing
    model = YOLO(str(ROOT / "yolo11n.pt"))
    return Path(model.export(format="openvino", half=False))


def benchmark(ov_dir: Path, img: np.ndarray, device: str, runs: int):
    from ultralytics import YOLO

    model = YOLO(str(ov_dir), task="detect")
    for _ in range(3):  # warmup (compile on first call)
        model.predict(img, imgsz=640, device=device, verbose=False)

    boxes = 0
    t0 = time.perf_counter()
    for _ in range(runs):
        r = model.predict(img, imgsz=640, device=device, verbose=False)[0]
        boxes = len(r.boxes)
    dt = (time.perf_counter() - t0) / runs * 1000.0
    return dt, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()

    print_devices()
    img_path = ensure_image()
    img = np.array(Image.open(img_path).convert("RGB"))

    print("\nexporting YOLO11n -> OpenVINO (once)…")
    ov_dir = export_yolo()
    print("  model:", ov_dir)

    print(f"\nYOLO11n @ 640, {args.runs} runs each (lower is better):")
    print(f"  {'device':<12} {'ms/run':>8}   {'boxes':>5}")
    results = {}
    for device in ["intel:cpu", "xpu", "npu"]:
        try:
            dt, boxes = benchmark(ov_dir, img, device, args.runs)
            results[device] = dt
            print(f"  {device:<12} {dt:8.1f}   {boxes:5d}")
        except Exception as e:  # noqa: BLE001
            print(f"  {device:<12} {'n/a':>8}   ({type(e).__name__}: {e})")

    if "intel:cpu" in results and "intel:gpu" in results:
        speedup = results["intel:cpu"] / results["intel:gpu"]
        print(f"\niGPU speedup over CPU: {speedup:.2f}x")
    print("\nIf GPU shows a latency, the iGPU is working ✓")


if __name__ == "__main__":
    main()
