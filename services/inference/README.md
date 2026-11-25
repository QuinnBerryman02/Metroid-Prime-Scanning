# Inference service

FastAPI service that runs the CV models. See `docs/LIFTING.md` for how detection
feeds into the 3D overlay, and `docs/MODELS.md` for model choices.

## Status

- **Stage A** - `POST /scan` returns YOLO detections (boxes + labels) for a
  keyframe, plus which detection is under the reticle (`aimed`). ✅
- **Stage B** - the aimed object is segmented (SAM 2.1), given depth (Depth
  Anything V2) and lifted into camera-space geometry: `contour2d`, `contourCam`,
  `obbCam`, `depthAtCentre`. ✅ (see `docs/LIFTING.md`)

## Setup

Uses the same venv as the notebooks (`services/inference/.venv`):

```bash
cd services/inference
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows
# macOS/Linux: .venv/bin/pip install -r requirements.txt
```

The YOLO weights (`yolo11n.pt`) download on first use and are exported to
OpenVINO IR automatically when `OV_DEVICE` is an Intel accelerator.

## Run

```bash
cd services/inference
.venv/Scripts/python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Environment:

| var | default | meaning |
|---|---|---|
| `OV_DEVICE` | `xpu` | `xpu` (Intel iGPU) / `npu` / `intel:cpu` / `cpu` |
| `MODEL_PATH` | `./yolo11n.pt` | weights; OpenVINO IR derived next to it |
| `EXPORT_OPENVINO` | `1` | auto-export the `.pt` to OpenVINO |
| `IMGSZ` | `640` | inference resolution |
| `CONF` | `0.25` | detection confidence threshold |
| `SEGMENT` | `1` | enable Stage-B segmentation (SAM) |
| `DEPTH` | `1` | enable Stage-B depth (Depth Anything V2) |
| `SAM_MODEL` | `./notebooks/sam2.1_t.pt` | SAM weights (local path or Ultralytics name) |
| `DEPTH_MODEL` | `depth-anything/Depth-Anything-V2-Small-hf` | HF depth model |
| `FOV_DEG` | `60` | default horizontal FOV for intrinsics |

Stage B models are optional: if they fail to load the service still serves
boxes (it logs a warning). Segment/depth run on CPU by default; set
`SAM_DEVICE` / `DEPTH_DEVICE` to move them.

### Latency (Intel Core Ultra 7 165U)

Per `/scan` on a keyframe: detect ~22 ms (iGPU), segment ~2.5 s (SAM 2.1 tiny,
CPU), depth ~1.2 s (Depth Anything V2 Small, CPU). Segmentation is only run for
the aimed object. Exporting SAM/depth to OpenVINO is the next speed-up.

## Endpoints

### `GET /health`
```json
{ "ok": true, "device": "xpu", "model": "…/yolo11n.pt", "loaded": true, "classes": 80 }
```

### `POST /scan`
JSON body (`image` is base64, no data-URI prefix):

```json
{ "image": "/9j/4AAQ…", "frameId": "abc", "aimX": 0.5, "aimY": 0.5 }
```

Response (Stage A):

```jsonc
{
  "frameId": "abc",
  "width": 640, "height": 360,
  "detections": [
    { "classId": 5, "label": "bus", "score": 0.92,
      "box": [27, 232, 802, 737], "mask": null }
  ],
  "aimed": {
    "detectionIndex": 0, "classId": 5, "name": "bus",
    "contour2d": [[u, v], ...],             // mask boundary, image pixels
    "contourCam": [[x, y, z], ...],          // same points, camera space
    "obbCam": { "center": [x, y, z], "size": [w, h, d], "quat": [x, y, z, w] },
    "depthAtCentre": 3.02, "scaleHint": "hit-test", "pointCount": 16273
  },
  "timing": { "detect": 22, "segment": 2533, "depth": 1155 }
}
```

`box` is `[x1, y1, x2, y2]` in pixels of the submitted image. Camera space has
origin at the camera, `+z` forward, in *relative* depth units - the phone
recovers metric scale from an AR hit-test (see `docs/LIFTING.md`).

## Test

```bash
.venv/Scripts/python.exe scripts/test_scan.py      # in-process, 3 scans
```

Measured on this machine (Intel Core Ultra 7 165U): steady-state ~20 ms/scan on
the iGPU (`xpu`); the first call after startup is warm (handled at load).
