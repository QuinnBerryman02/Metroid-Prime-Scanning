"""
Smoke-test `/scan` in-process (no server needed).

    .venv/Scripts/python.exe scripts/test_scan.py
"""
from __future__ import annotations

import base64
import io
import sys
import urllib.request
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402

SAMPLES = ROOT / "samples"


def ensure_image() -> Path:
    path = SAMPLES / "bus.jpg"
    if not path.exists():
        SAMPLES.mkdir(exist_ok=True)
        urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", path)
    return path


def main() -> int:
    buf = io.BytesIO()
    Image.open(ensure_image()).convert("RGB").save(buf, format="JPEG", quality=90)
    payload = base64.b64encode(buf.getvalue()).decode()
    body = {"image": payload, "frameId": "test", "aimX": 0.5, "aimY": 0.5}

    with TestClient(app) as client:
        health = client.get("/health").json()
        print("health:", health)
        if not health.get("ok"):
            return 1

        data = {}
        for i in range(3):
            response = client.post("/scan", json=body)
            response.raise_for_status()
            data = response.json()
            t = data["timing"]
            parts = " ".join(f"{k}={v:.0f}" for k, v in t.items())
            print(f"  scan {i + 1}: {parts} ms")

    print(f"image {data['width']}x{data['height']}  timing {data['timing']}")
    for d in data["detections"]:
        box = ", ".join(f"{v:.0f}" for v in d["box"])
        flag = " mask" if d.get("mask") else ""
        print(f"  {d['label']:<12} {d['score']:.2f}  [{box}]{flag}")

    aimed = data["aimed"]
    if aimed:
        obb = aimed.get("obbCam")
        obb_str = "-"
        if obb:
            size = "x".join(f"{v:.2f}" for v in obb["size"])
            centre = ",".join(f"{v:.2f}" for v in obb["center"])
            obb_str = f"size {size} @ ({centre})"
        depth = aimed.get("depthAtCentre")
        depth_str = f"{depth:.3f}" if depth is not None else "-"
        print(
            f"aimed: {aimed['name']}  points={aimed.get('pointCount')} "
            f"contour={len(aimed.get('contour2d') or [])}pts "
            f"depthAtCentre={depth_str} "
            f"scale={aimed.get('scaleHint')}"
        )
        print(f"       obbCam {obb_str}")
    return 0 if data["detections"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
