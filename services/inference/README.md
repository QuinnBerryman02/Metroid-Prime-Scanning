# Inference service

FastAPI service that runs the CV models. See `docs/LIFTING.md` for how masks
become 3D anchors, and `docs/MODELS.md` for model choices.

## Status

- **Stage A** - `POST /scan` returns YOLO detections (boxes + labels) for a
  keyframe, plus which detection is under the reticle (`aimed`). ✅
- **Stage B** - masks are lifted to camera-space geometry: `contour2d`,
  `contourCam`, `obbCam`, `depthAtCentre`. ✅

The phone's radial wheel calls one of five **methods** (`GET /methods`):

| id | label | kind | model | device | notes |
|---|---|---|---|---|---|
| `sam` | SAM1 | prompt | MobileSAM | CPU | SAM1 variant; `sam_b.pt` is ~15 s/scan |
| `sam2` | SAM2 | prompt | SAM 2.1 tiny | CPU | best masks, slow on CPU |
| `fastsam` | FastSAM | detect | FastSAM-s | OpenVINO iGPU | class-agnostic, many fragments |
| `yolo` | YOLO | detect | YOLO11n-seg | OpenVINO iGPU | class-aware, fast |
| `depth` | DEPTH | depth | Depth Anything V2 | OpenVINO iGPU | colourised depth PNG |

- **prompt** methods use the reticle as a point prompt and return one mask.
- **detect** methods return 1..N labelled masks for the frame.
- **depth** returns the depth map for the frame.

> SAM1/SAM2 can't be exported to OpenVINO (Ultralytics' exporter doesn't support
> SAM/SAM2, and optimum-intel rejects `sam2_video`), so the fast paths use
> YOLO11-seg / FastSAM instead.

## Setup

Uses the same venv as the notebooks (`services/inference/.venv`):

```bash
cd services/inference
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows
# macOS/Linux: .venv/bin/pip install -r requirements.txt
```

The YOLO weights download on first use and are exported to OpenVINO IR
automatically when the resolved device is an Intel accelerator.

## Run

```bash
cd services/inference
.venv/Scripts/python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Environment:

| var | default | meaning |
|---|---|---|
| `OV_DEVICE` | `auto` | `auto` (iGPU if OpenVINO sees one) / `xpu` / `npu` / `cpu` |
| `MODEL_PATH` | `./yolo11n.pt` | detection weights (for the prompt-method label) |
| `EXPORT_OPENVINO` | `1` | auto-export YOLO-style models to OpenVINO |
| `IMGSZ` / `CONF` | `640` / `0.25` | detection resolution / threshold |
| `SAM_MODEL` | `./notebooks/mobile_sam.pt` | SAM1 weights |
| `SAM2_MODEL` | `./notebooks/sam2.1_t.pt` | SAM2 weights |
| `FASTSAM_MODEL` | `./notebooks/FastSAM-s.pt` | FastSAM weights |
| `YOLO_SEG_MODEL` | `./yolo11n-seg.pt` | YOLO11-seg weights |
| `DEPTH_MODEL` | `depth-anything/Depth-Anything-V2-Small-hf` | HF depth model |
| `DEPTH_DEVICE` | `GPU` | OpenVINO device for the depth IR |
| `DEPTH_IR` | `./depth_anything_v2_small_openvino_model/…xml` | exported depth IR |
| `FOV_DEG` | `60` | default horizontal FOV for intrinsics |

Methods are optional: a model that fails to load is reported as
`available: false` and simply greyed out on the wheel.

The depth model is **exported to OpenVINO on first run** (one-off, ~30 s). Every
method is then loaded in the background at startup and warmed up, so the first
selection is not a cold compile.

### Latency (Intel Core Ultra 7 165U)

Per keyframe, warm, excluding the one-off model load:

| method | segment | + depth | notes |
|---|---|---|---|
| YOLO | ~40 ms | ~180 ms | iGPU |
| FastSAM | ~500 ms | ~650 ms | iGPU, ~100 masks |
| DEPTH | - | ~150 ms | iGPU |
| SAM2 | ~2-6 s | - | CPU |
| SAM1 (MobileSAM) | ~1-3 s | - | CPU |

## Endpoints

### `GET /health`
```json
{ "ok": true, "device": "xpu", "loaded": true,
  "loadedSegmenters": ["yolo", "fastsam"], "depth": true }
```

### `GET /methods`
```json
{ "device": "xpu",
  "methods": [
    { "id": "yolo", "label": "YOLO", "kind": "detect", "device": "GPU", "available": true },
    { "id": "depth", "label": "DEPTH", "kind": "depth", "device": "GPU", "available": true }
  ] }
```

### `POST /scan`
JSON body (`image` is base64, no data-URI prefix):

```json
{ "image": "/9j/4AAQ…", "frameId": "abc", "aimX": 0.5, "aimY": 0.5,
  "method": "yolo", "fovDeg": 60 }
```

Response (mask method):

```jsonc
{
  "frameId": "abc", "width": 640, "height": 360,
  "method": "yolo", "kind": "detect",
  "detections": [
    { "classId": 0, "label": "bus", "score": 0.92,
      "box": [27, 232, 802, 737], "contour2d": [[u, v], ...] }
  ],
  "aimed": {
    "detectionIndex": 0, "classId": 0, "name": "bus",
    "contour2d": [[u, v], ...],             // mask boundary, image pixels
    "contourCam": [[x, y, z], ...],          // same points, camera space
    "obbCam": { "center": [x, y, z], "size": [w, h, d], "quat": [x, y, z, w] },
    "depthAtCentre": 3.02, "scaleHint": "hit-test", "pointCount": 16273
  },
  "timing": { "segment": 40, "depth": 140 }
}
```

Response (`method="depth"`) - `detections` is empty and `depth` is set:

```jsonc
{
  "frameId": "abc", "width": 640, "height": 360,
  "method": "depth", "kind": "depth",
  "detections": [], "aimed": null,
  "depth": { "png": "iVBORw0…", "width": 640, "height": 360,
             "min": 0.30, "max": 7.74 },
  "timing": { "depth": 150 }
}
```

`box` is `[x1, y1, x2, y2]` in pixels of the submitted image. Camera space has
origin at the camera, `+z` forward, in *relative* depth units - the phone
recovers metric scale from an AR hit-test (see `docs/LIFTING.md`).

## Test

```bash
.venv/Scripts/python.exe scripts/test_scan.py   # every available method
```
