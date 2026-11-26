"""POST a real keyframe (with planes consistent with the image) to a RUNNING
/recon/frame and check a cloud comes back. Requires the service on :8000.

    python scripts/test_recon_http.py [base-url]
"""
import base64
import io
import json
import sys
import urllib.request

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, ".")

from app import config  # noqa: E402
from app.depth import DepthEstimator  # noqa: E402

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
A_TRUE = 1.0  # we pretend rel_depth = Z / A_TRUE

pil = Image.open("samples/bus.jpg").convert("RGB")
w, h = pil.size
f = 0.5 * w / np.tan(np.radians(60) / 2.0)
fx, fy, cx, cy = f, f, w / 2.0, h / 2.0

est = DepthEstimator(
    config.DEPTH_MODEL, config.DEPTH_DEVICE, config.DEPTH_IR, config.EXPORT_OPENVINO
)
Z = A_TRUE * cv2.medianBlur(est.predict_depth(pil), 5)


def quad(u0, u1, v0, v1):
    def p(u, v):
        z = float(Z[v, u])
        return [(u - cx) / fx * z, (v - cy) / fy * z, z]

    return [p(u0, v0), p(u1, v0), p(u1, v1), p(u0, v1)]


planes = [
    {"id": 1, "action": "added", "orientation": "horizontal",
     "polygon": quad(60, 240, int(0.62 * h), int(0.80 * h)),
     "center": [0, 0, 0], "normal": [0, 1, 0]},
    {"id": 2, "action": "added", "orientation": "vertical",
     "polygon": quad(int(0.30 * w), int(0.75 * w), 10, int(0.30 * h)),
     "center": [0, 0, 0], "normal": [0, 0, -1]},
]

buf = io.BytesIO()
pil.save(buf, format="JPEG", quality=85)
body = {
    "sessionId": "http-test", "frameId": "kf-1",
    "position": [0, 1.6, 0], "quaternion": [0, 0, 0, 1],
    "pose": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    "fovDeg": 60, "width": w, "height": h,
    "image": base64.b64encode(buf.getvalue()).decode(),
    "reason": "first", "sharpness": 40.0, "planes": planes,
}
req = urllib.request.Request(
    URL + "/recon/frame", data=json.dumps(body).encode(),
    headers={"content-type": "application/json"},
)
res = json.loads(urllib.request.urlopen(req, timeout=180).read())
print({k: v for k, v in res.items() if k not in ("cloud", "timing")})
print("timing:", res.get("timing"))
c = res.get("cloud")
if c:
    print("cloud:", {"count": c["count"], "voxel": c["voxel"],
                       "method": c["scaleMethod"], "a": round(c["scaleA"], 3),
                       "b": round(c["scaleB"], 3)})
assert c and c["count"] > 0, "no cloud came back"
print("\nHTTP E2E OK")
