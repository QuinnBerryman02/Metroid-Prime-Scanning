# Architecture

## 1. Goals / non-goals

**Goals**
- Scan objects with a phone camera and get back labelled, 3D-anchored outlines.
- A Metroid-Prime-style HUD: centre reticule, outlines that stick to the world,
  info cards for the scanned object under the reticule.
- A desktop viewer showing the reconstructed environment with all scanned objects
  highlighted + selectable.
- Everything runs locally on a desktop with an **integrated GPU**.

**Non-goals**
- Photoreal NeRF / 3D Gaussian Splatting reconstruction.
- Metric-accurate, survey-grade geometry.
- iOS parity. Target **Android Chrome** for AR; desktop for the viewer.
- Multi-user / cloud scale.

## 2. System topology

```
┌──────────────────────────┐        ┌─────────────────────────────┐
│  Mobile AR client        │        │  Desktop viewer client      │
│  (Babylon.js + WebXR)    │        │  (Babylon.js)               │
│  - camera passthrough    │        │  - 3D reconstruction render │
│  - anchored outlines     │        │  - highlight / select       │
│  - reticule + info cards │        │  - object inspector         │
└───────┬──────────────────┘        └──────────────┬──────────────┘
        │  WS (frames, poses)                      │ WS (world state)
        │  WS (anchors, labels)                    │
        └──────────────┬───────────────────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  apps/server  (TS / Fastify) │
        │  - sessions & world state    │
        │  - frame queue + dedupe      │
        │  - broadcasts world deltas   │
        └──────────────┬───────────────┘
                       │ HTTP / gRPC
                       ▼
        ┌──────────────────────────────┐
        │  services/inference (Python) │
        │  YOLO-seg → depth → CLIP     │
        │  OpenVINO (Intel iGPU/CPU)   │
        └──────────────────────────────┘
```

- **Client** = one Vite app with two routes/modes: `/ar` (phone) and `/viewer` (desktop).
- **Server** = TypeScript (Fastify) gateway. Owns session + authoritative world model,
  fan-out to viewers. Keeps the "TS backend" requirement.
- **Inference** = Python FastAPI + OpenVINO. Kept in Python because the ML tooling
  (OpenVINO, PyTorch export, Ultralytics, transformers) is a decade ahead of Node/`onnxruntime-node`.
  This split is deliberate and worth calling out in the report as a design decision.

## 3. Why Babylon.js + WebXR

| Option | Verdict |
|---|---|
| **Babylon.js + WebXR** | ✅ **Chosen.** TS-first, first-class WebXR: `WebXRAnchors`, `WebXRHitTest`, `WebXRPlaneDetection`, `WebXRDepthSensing`, `WebXRFeaturePointSystem`, `WebXRRawCamera`. Rich importers (glTF/OBJ/STL), materials, outlines (`renderOutline`), GUI/`AdvancedDynamicTexture` for HUD. Covers both mobile AR and desktop viewer. |
| Three.js + `@react-three/xr` | Good; more assembly required for anchors/plane detection; great if you want React. Solid alternative. |
| A-Frame + AR.js | Too high-level; hard to do custom HUD/anchors. |
| 8th Wall / Zappar | Commercial, paid; overkill for a course demo. |
| MindAR | Marker/image tracking, not general environment anchoring. |
| native ARCore/ARKit | Not web, platform-locked. |

**Key trick:** we do **not** implement SLAM ourselves. WebXR gives us the device 6DoF
pose, plane detection, hit-test, feature points, and (on supported devices) depth. That is
the "SLAM" layer for free. Our job is fusing camera frames + pose + depth into a map.

## 4. Coordinate frames (must be nailed down early)

1. **Camera frame** - right-handed, from WebXR `XRRigidTransform` / camera matrix.
2. **World frame** - the WebXR reference space. `local-floor` / `unbounded` on device.
   Reconstruction + anchors live here.
3. **Server world frame** - a single canonical frame per session. For a single phone we
   can treat the first anchor as origin, but plan an explicit `worldFromDevice` transform.

Rule: every pose/frame sent to the server is tagged with the world transform at capture
time. The server never guesses; the viewer applies the same transform. Keep a
`WorldTransform` on every `WorldObject`.

## 5. Pipeline: 2D detection -> 3D anchor

> Full detail, maths, API shape and limitations: [`docs/LIFTING.md`](LIFTING.md).

The central problem the brief calls out: *2D masks don't anchor*. Solution - **lift to 3D**
with depth + pose, instead of relying on a 3D SAM.

```
camera frame (RGB) ─┬─> detector (YOLO11n) ──> boxes + classes
                    ├─> FastSAM / MobileSAM ─> masks
                    └─> depth (DepthAnythingV2-s) ─> per-pixel depth map
                                     │
        pose (WebXR) ────────────────┤
                                     ▼
        back-project mask ∩ depth  ->  3D point cloud cluster (camera frame)
                                     │
                    transform by worldFromCamera  ->  points in world frame
                                     │
        cluster/fuse with existing objects (IoU / centroid distance)  ->  WorldObject
```

> **Superceded for the *environment* model.** The single-view path below stays as the
> **close-up object scan** (crisp, deliberate). The world/environment is now built by
> **fusing streamed keyframes into a world voxel map** (pose from ARCore, depth from
> DA V2, scale from ARCore planes) and segmenting that. See
> [`docs/RECONSTRUCTION.md`](RECONSTRUCTION.md) and [`docs/SLAM_SURVEY.md`](SLAM_SURVEY.md).

**Lifting a mask to 3D** (per object):
- Take masked pixels, sample their depth, unproject with intrinsics
  `p_world = T_world_cam · K⁻¹ · [u,v,1]ᵀ · d(u,v)`.
- Robust-ish reduction: RANSAC plane fit or quantile cropping to drop background depth
  bleed, then take an **oriented bounding box (OBB)** or convex hull of the remaining
  points. Use the OBB/hull as the **scan border** in 3D.
- The 2D mask silhouette is also kept and stored in **object space** (mask pixels relative
  to the detection box) so the mobile HUD can draw a textured/billboard outline that
  matches the object's appearance from the current view angle.

**Anchoring & persistence:**
- Attach the 3D OBB/hull + centroid to a **WebXR anchor** (`XRAnchor`) so it is stable
  under tracking drift; the runtime corrects the anchor pose as the map refines.
- Re-detect and **fuse by proximity** so repeated scans update, not duplicate, objects.

**Occlusion (bonus):** use `WebXRDepthSensing` or the stored point cloud to write depth
so outlines are correctly occluded by nearer real geometry.

## 6. Reconstruction (the "3D map")

Keep it simple and iGPU-friendly. In priority order:

1. **Point-cloud fusion (recommended, MVP).** Accumulate unprojected depth frames in world
   space into a voxel-hashed point cloud (e.g. 1-3 cm voxels). Render with splat/disc
   impostors. Cheap, incremental, no training.
2. **TSDF mesh fusion.** Feed RGB-D + poses to Open3D `ScalableTSDFVolume` or a small
   custom voxel-hash TSDF; extract a triangle mesh on demand (`MarchingCubes` /
   `extract_triangle_mesh`). Gives solid surfaces for the desktop viewer.
3. **Plane reconstruction.** Use WebXR plane detection + ARCore/ARKit planes directly, then
   colour them. Cheapest, coarsest.

**Deliberately avoid full NeRF / 3DGS** in the real-time loop: training is too heavy for an
iGPU and adds latency with no benefit at this scope. If a pretty offline reconstruction is
wanted for the report, run it once offline (e.g. nerfstudio / gsplat on a few hundred
frames) and load it into the viewer as a static asset.

## 7. Mobile HUD design (the Metroid Prime bit)

- **Reticule**: fixed at screen centre, screen-space canvas/DOM overlay.
- **Outlines**: for each `WorldObject`, project its OBB corners to screen space and draw a
  coloured outline (per-class colour) in a 2D overlay canvas, or render a 3D outline. 2D
  overlay is simpler and reads well; 3D gives correct occlusion. Start 2D, add 3D later.
- **Hit-testing the reticule**: raycast from the camera through screen centre against each
  object's 3D OBB (or against a screen-space distance of the outline to the reticule).
  On hover/proximity -> highlight + show its **info card** (name, class, confidence,
  description, an icon).
- **Info card**: centre-screen DOM panel. Also list all scanned info per the brief.
- **Scan action**: a button (or dwell timer) captures the frame under the reticule, sends
  it for segmentation+classification, then materialises the outline with a Metroid-style
  animation.

## 8. Client/server protocol (indicative)

Defined in `packages/shared/src/protocol.ts`. Sketch:

```ts
// client -> server
type ScanFrameMsg   = { type: "scan_frame"; frameId: string; pose: number[16];
                        intrinsics: Intrinsics; jpeg: ArrayBuffer; depth?: ArrayBuffer };
type RescanMsg      = { type: "rescan"; objectId: string; jpeg: ArrayBuffer };
// server -> client
type WorldDeltaMsg  = { type: "world_delta"; added: WorldObject[]; updated: WorldObject[] };
type ScanResultMsg  = { type: "scan_result"; frameId: string; objects: WorldObject[] };
```

```ts
interface WorldObject {
  id: string;
  label: string;            // classifier output
  confidence: number;
  description?: string;     // VLM caption / explanation
  color: string;
  // world-space geometry (scan border)
  center: [number, number, number];
  size:   [number, number, number];
  quaternion: [number, number, number, number];
  hull?: number[];          // optional point cloud / hull
  // 2D appearance for HUD outlines
  maskBox?: [number, number, number, number];
  anchorUuid?: string;      // WebXR anchor id on the scanning device
}
```

Transport: **WebSocket** for frames (binary JPEG + JSON control) and world deltas; **HTTP**
for model health / one-shot classify. Frames are downscaled (long side ~640-960) and
JPEG/WebP-compressed; dedupe by pose + perceptual hash so we don't segment the same view twice.

## 9. Desktop viewer

- Receives `WorldObject[]` over WS, renders point cloud / TSDF mesh + each object's OBB.
- Hover/click -> highlight + info panel. Orbit camera (ArcRotateCamera).
- Optional: fly around, toggle reconstruction layers, toggle labels.
- Useful debug surface: shows world-frame alignment, duplicates/fusion, drift.

## 10. Non-functional considerations

- **Latency budget**: aim < 300 ms scan -> outline. Budget: capture+encode ~30 ms,
  network ~5 ms (localhost/LAN), detect+segment ~80 ms, depth ~80 ms, classify async.
  Do NOT block the outline on the classifier - show outline immediately, fill the card
  when the label returns.
- **Bandwidth**: keyframes only, aggressive dedupe, no raw video upload.
- **Frame sync**: include the exact WebXR pose *at capture*, not "now" - otherwise anchors
  swim. Snapshot `viewer.pose` with the texture and send together.
- **Drift & relocalisation**: anchor each object; if the session relocalises, anchors move
  with the map. Don't anchor to raw world coordinates only.
- **iOS reality check**: iOS Safari's WebXR is partial (no full AR session / camera
  passthrough for arbitrary WebXR pages as of writing). Demo on Android Chrome; document
  the limitation. WebXR Viewer (Mozilla) is an option but limited.
- **HTTPS required**: camera + WebXR need a secure context. Use `vite --https` or a LAN
  cert/tunnel for phone testing.
- **Privacy**: everything stays local; note it in the report.
- **Reproducibility**: model fetch script + pinned versions; commit `package-lock.json`
  and `requirements.txt`.

## 11. Risks

| Risk | Mitigation |
|---|---|
| iGPU too slow for depth + segment every frame | Only on keyframes; cache by object; OpenVINO AUTO device |
| iGPU model export friction | Start CPU-only to get pipeline working, then move to GPU |
| Anchors drift / no relocalisation in WebXR | Accept approximate; fuse by proximity; re-anchor on re-scan |
| iOS doesn't work | Demo on Android; document |
| "3D SAM" doesn't exist cheaply | Use 2D mask + depth lifting + OBB (documented above) |
| Coordinate-frame bugs | Single canonical world frame, explicit transforms, debug gizmos |
