# Model Selection (integrated-GPU friendly)

Target hardware: **Intel integrated GPU** (Iris Xe / UHD, shared RAM, no CUDA).
Everything below is chosen to run on CPU or Intel iGPU via **OpenVINO** or quantised
**ONNX** (onnxruntime). Prefer `*-n`/`*-s`/`small`/`base` variants and INT8/FP16.

## Recommended stack

| Stage | Primary pick | Fallback / lighter | Runtime | Notes |
|---|---|---|---|---|
| Object detection | **YOLO11n / YOLOv8n** | YOLOv8n, RT-DETR-r18 | OpenVINO / ONNX | ~3-10 ms on iGPU; class + box |
| Instance segmentation | **FastSAM-s** (YOLOv8-seg) | MobileSAM, EfficientSAM-Ti, YOLOv8n-seg | OpenVINO / ONNX | FastSAM = YOLO backbone, *much* faster than SAM ViT-H |
| Promptable segmentation | **MobileSAM** | EfficientSAM, SAM ViT-T | ONNX | For the reticule-picked region; ~10x lighter than SAM-H |
| Monocular depth | **Depth Anything V2 Small** | MiDaS `MiDaS_small`, **WebXR Depth API** | ONNX / OpenVINO | Converts 2D mask -> 3D points |
| Metric depth (optional) | **UniDepth / Metric3D-small** | ZoeDepth-small | ONNX | Better absolute scale if needed |
| Classification | **CLIP ViT-B/32** (OpenCLIP) | MobileCLIP-S0, SigLIP-base | ONNX / OpenVINO | Zero-shot labels, change prompts live |
| Caption / "explanation" | **Florence-2-base** | BLIP-base, Moondream2 (int4) | ONNX / llama.cpp | Florence-2 does caption + detection + grounding in one small model |
| Open-vocab detect (optional) | **GroundingDINO-tiny** | YOLO-World-s | ONNX | "scan for anything" |
| Features / tracking | **ORB** / SuperPoint+LightGlue | - | OpenCV / ONNX | Cross-frame association, dedupe |

### Why not plain SAM (ViT-H)?
SAM ViT-H is ~2.4 GB and seconds-per-image even on a discrete GPU. On an iGPU it is a
non-starter. Use **FastSAM** (runs at YOLO speed) or **MobileSAM** for the interactive,
reticule-driven mask. There is no practical, reliable "3D SAM" we want to run locally;
instead we lift 2D masks into 3D (see ARCHITECTURE) - cheaper and good enough.

### Why OpenVINO?
- Built by Intel specifically for Intel CPU/iGPU/NPU. `device="GPU"` / `"AUTO"` / `"MULTI:GPU,CPU"`.
- Covers YOLO, SAM-family, CLIP, depth models with ready conversion + INT8 quantisation.
- `AUTO` device selection and model caching (seconds -> ms on first-load).

Fallbacks if OpenVINO proves awkward: `onnxruntime-directml` (Windows) or
`onnxruntime` CPU.

## Model sizes / rough expectations (iGPU, ~640px input)

| Model | Params | Latency (rough) |
|---|---|---|
| YOLO11n | 2.6 M | 5-15 ms |
| FastSAM-s | ~11 M | 20-50 ms |
| MobileSAM | ~10 M | 20-60 ms |
| Depth Anything V2 small | 24 M | 40-100 ms |
| CLIP ViT-B/32 | 88 M | 15-40 ms |
| Florence-2-base | 0.23 B | 300 ms - 1.5 s |

Design the pipeline so **detection + depth run on every keyframe**, while the expensive
**classifier/captioner runs once per new object** (cached by object id).

## Measured on the target machine (Intel Core Ultra 7 165U, iGPU)

`services/inference/samples/openvino_check.py` — YOLO11n @ 640, 15 runs:

| device | ms/run |
|---|---|
| `intel:cpu` | 50.4 |
| `xpu` | **21.7** (2.3x) |
| `npu` | 25.1 |

So the plan is sound: OpenVINO + iGPU gives a ~2.3x win, NPU is a nice backup.

Notebook findings (`services/inference/notebooks`):

- YOLO11n, FastSAM-s, MobileSAM, Depth Anything V2, CLIP all run (CPU).
- **SAM 2.1 tiny** gives the best smooth object mask (box or point prompt,
  ~2 s CPU) — the recommended "scan action" model. SAM ViT-B is better still
  but ~10 s CPU, so only worth it once exported to OpenVINO.
- FastSAM is class-agnostic and fragments large objects; good for whole-frame
  masks without a prompt, not for isolating one object.
- Unprojecting **mask** pixels (not the whole box) removes the background and
  makes the 3D shell recognisable. Depth is still *relative*, so scale must be
  fixed from AR planes / known size.

## Model files & fetching

Weights are **not** committed (see `.gitignore`). Add a `services/inference/scripts/fetch_models.*`
that downloads/export to `services/inference/models/`. Use the OpenVINO `optimum-intel`
exporters or `ultralytics export format=openvino`.

## Quantisation guidance
- Export FP16 first, validate, then try INT8 (NNCF / OpenVINO PTQ) on a small calibration set.
- Keep the classifier FP16 if INT8 hurts CLIP zero-shot quality noticeably.
