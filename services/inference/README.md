# Inference service

FastAPI service that runs the CV models. See `docs/LIFTING.md` for how detection
feeds into the 3D overlay, and `docs/MODELS.md` for model choices.

## Status

- **Stage A** - `POST /scan` returns YOLO detections (boxes + labels) for a
  keyframe, plus which detection is under the reticle (`aimed`). ✅
- **Stage B** - the aimed object is segmented, given depth (Depth Anything V2)
  and lifted into camera-space geometry: `contour2d`, `contourCam`, `obbCam`,
  `depthAtCentre`. ✅ (see `docs/LIFTING.md`)

`POST /scan` takes a `mode`:

- `"scan"` (default) - detection + segmentation + lifting.
- `"depth"` - just the depth map (colourised PNG) for the frame.

Two segmentation backends (`segmenter`, request-overridable):

| backend | model | device | latency | notes |
|---|---|---|---|---|
| `sam2` (default) | SAM 2.1 tiny | CPU | ~2.0 s | best masks |
| `yolo11` | YOLO11n-seg | OpenVINO iGPU | ~40 ms | class-aware, good enough |

> SAM 2.1 can't be exported to OpenVINO (Ultralytics' exporter doesn't support
> SAM/SAM2, and optimum-intel rejects `sam2_video`), so the fast path uses
> YOLO11-seg instead.

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
| `SEGMENT` | `1` | enable Stage-B segmentation |
| `DEPTH` | `1` | enable Stage-B depth (Depth Anything V2) |
| `SEGMENTER` | `sam2` | default backend: `sam2` \| `yolo11` |
| `SAM_MODEL` | `./notebooks/sam2.1_t.pt` | SAM weights (local path or Ultralytics name) |
| `SAM_DEVICE` | `cpu` | device for SAM (PyTorch) |
| `YOLO_SEG_MODEL` | `./yolo11n-seg.pt` | YOLO11-seg weights (OpenVINO-exported) |
| `DEPTH_MODEL` | `depth-anything/Depth-Anything-V2-Small-hf` | HF depth model |
| `DEPTH_DEVICE` | `GPU` | OpenVINO device for the depth IR |
| `DEPTH_IR` | `./depth_anything_v2_small_openvino_model/…xml` | where the exported IR lives |
| `FOV_DEG` | `60` | default horizontal FOV for intrinsics |

Stage B models are optional: if they fail to load the service still serves
boxes (it logs a warning).

The depth model is **exported to OpenVINO on first run** (one-off, ~30 s) and
runs on the iGPU. The `yolo11` segmenter also exports/compiles to OpenVINO on
first use and is warmed up, so the first request isn't the slow one.

### Latency (Intel Core Ultra 7 165U)

Per keyframe, warm (`mode="scan"`):

| stage | `sam2` (quality) | `yolo11` (fast) |
|---|---|---|
| detect (iGPU) | ~20 ms | ~20 ms |
| segment | ~2000 ms (CPU) | ~40 ms (iGPU) |
| depth (iGPU) | ~150 ms | ~150 ms |
| **total** | **~2.2 s** | **~0.2 s** |

`mode="depth"` is ~150 ms. Depth on CPU (PyTorch fallback) is ~1.2 s.

## Endpoints

### `GET /health`
```json
{ "ok": true, "device": "xpu", "model": "…/yolo11n.pt", "loaded": true,
  "classes": 80, "segmenter": "sam2", "loadedSegmenters": ["sam2"], "depth": true }
```

### `POST /scan`
JSON body (`image` is base64, no data-URI prefix):

```json
{ "image": "/9j/4AAQ…", "frameId": "abc", "aimX": 0.5, "aimY": 0.5,
  "mode": "scan", "segmenter": "yolo11", "fovDeg": 60 }
```

`mode` is `"scan"` (default) or `"depth"`. `segmenter` overrides the server
default for this request.

Response (`mode="scan"`):

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
  "timing": { "detect": 22, "segment": 2000, "depth": 150 }
}
```

Response (`mode="depth"`) - `detections` is empty and `depth` is set:

```jsonc
{
  "frameId": "abc", "width": 640, "height": 360,
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
.venv/Scripts/python.exe scripts/test_scan.py   # both segmenters + depth mode
```
