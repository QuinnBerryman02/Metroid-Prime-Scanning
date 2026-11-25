"""
Smoke-test `/scan` in-process (no server needed).

    .venv/Scripts/python.exe scripts/test_scan.py

Runs every available wheel method (SAM1, SAM2, FastSAM, YOLO, DEPTH).
"""
from __future__ import annotations

import base64
import io
import sys
import time
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


def encode() -> str:
    buf = io.BytesIO()
    Image.open(ensure_image()).convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def fmt(timing: dict) -> str:
    return " ".join(f"{k}={v:.0f}" for k, v in timing.items())


def run_scan(client: TestClient, image: str, **extra):
    body = {"image": image, "frameId": "test", "aimX": 0.5, "aimY": 0.5, **extra}
    t0 = time.perf_counter()
    response = client.post("/scan", json=body)
    response.raise_for_status()
    return response.json(), (time.perf_counter() - t0) * 1000


def main() -> int:
    image = encode()
    ok = True
    with TestClient(app) as client:
        health = client.get("/health").json()
        print("health :", health)
        registry = client.get("/methods").json()
        print("device :", registry["device"])

        for method in registry["methods"]:
            if not method["available"]:
                print(f"[{method['label']:<7}] unavailable - skipped")
                continue
            data, rtt = run_scan(client, image, method=method["id"])
            if method["kind"] == "depth":
                result = data["depth"]
                print(
                    f"[{method['label']:<7}] {method['device']} {method['kind']:<6} "
                    f"rtt={rtt:6.0f} ms | {fmt(data['timing'])} | png={len(result['png'])}B"
                )
                continue
            aimed = data["aimed"]
            labels = ", ".join(d["label"] for d in data["detections"][:6])
            print(
                f"[{method['label']:<7}] {method['device']} {method['kind']:<6} "
                f"rtt={rtt:6.0f} ms | {fmt(data['timing'])} | masks={len(data['detections'])} [{labels}]"
            )
            if aimed:
                obb = aimed["obbCam"]
                size = "x".join(f"{v:.2f}" for v in obb["size"]) if obb else "-"
                print(
                    f"           aimed={aimed['name']} points={aimed['pointCount']} "
                    f"contour={len(aimed.get('contour2d') or [])}pts obb={size}"
                )
            elif method["kind"] == "prompt":
                ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
