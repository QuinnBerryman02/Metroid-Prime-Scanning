# Implementation Plan

## 0. Guiding principle

Build a **thin vertical slice first**, then deepen it. The riskiest unknowns are
(a) WebXR camera + anchors on the phone and (b) 2D-mask -> 3D-anchor lifting. Prove both
early with placeholder ML (hardcoded detections) before touching real models.

Suggested overall effort for a course-sized scope: ~4-6 focused weeks.

## Priorities & scope decisions

1. **Mobile AR scan HUD is the priority.** It carries the most marks (interaction +
   innovation) and is the most distinctive part of the demo.
2. **3D viewer can be coarse.** Room walls + scanned objects as boxes is enough; a full
   NeRF/3DGS house scan is *nice-to-have, not essential*. Do not sink time into it before
   the HUD works.
3. **Occlusion of AR overlays is out of scope.** Overlays draw on top (2D overlay canvas);
   no depth-based occlusion needed.
4. **Depth source is TBD** - depends on whether the target phone exposes WebXR
   `depth-sensing`. Use `apps/client/capabilities.html` to find out (see M1). If it does,
   we can skip the monocular depth model for lifting on that device.

---

## Milestone 0 - Repo & tooling (0.5 week)

- [x] `git init`, monorepo layout (`apps/`, `services/`, `packages/`).
- [ ] Root npm workspaces; Vite + TS client; Fastify + TS server; FastAPI inference stub.
- [ ] Lint/format (eslint + prettier), `tsconfig` base, `.gitignore`, `.env.example`.
- [ ] Dev HTTPS story for phone testing (mkcert or `vite --https` + LAN).
- [ ] Model fetch script placeholder; `requirements.txt` with pinned versions.

**Done when:** `npm run dev` serves a blank Babylon scene, a WS connect succeeds, and the
inference `/health` responds.

---

## Milestone 1 - WebXR skeleton on the phone (1 week)

- [x] **Capability/depth diagnostic page** (`apps/client/capabilities.html`): reports
      secure context, WebXR support, enabled AR features (`session.enabledFeatures`),
      live hit-test reticle and depth-map preview. **Run this on the phone first** - the
      result decides whether we need monocular depth at all.
- [x] Babylon `WebXR` default experience: `immersive-ar`, `local-floor`, DOM overlay.
- [x] Camera passthrough via immersive-ar (alpha-blend).
- [x] Hit-test reticle tracks real surfaces; tap / **Place** drops a marker.
- [x] Anchors: placed box is held by a WebXR `XRAnchor` (walk around it - stays put).
- [x] Send pose to server; server relays to viewers; viewer shows a moving phone frustum.
- [x] On-screen debug overlay for phones without a debugger.
- [x] Headless smoke test for the pose relay (`npm run smoke:ws`).

**Done when:** a static cube placed in AR stays anchored while moving, and a desktop
viewer shows the phone's live pose in a shared world. *(ready for device test)*

**De-risk here:** secure context, Android Chrome quirks, anchor lifetime, frame sync.

---

## Milestone 2 - Detection + segmentation service (1 week)

Stage A (done):

- [x] FastAPI `/scan` endpoint: accepts a base64 keyframe, returns detections.
- [x] YOLO11n detection via OpenVINO on the iGPU (`xpu`), auto-export + warmup.
- [x] Stable JSON schema (`Detection {box, mask?, score, label}`) + `aimed` object
      with Stage-B fields (`contourCam`/`obbCam`) already present.
- [x] Test harness (`scripts/test_scan.py`); measured ~20 ms/scan on the iGPU.
- [x] Gateway `POST /scan` proxy + `/scan.html` phone verification page.

Stage B (done):

- [x] Segment the aimed object -> `contour2d`; lift to camera space ->
      `contourCam` + `obbCam` (PCA + robust extents). See `docs/LIFTING.md`.
- [x] Depth Anything V2 exported to OpenVINO (iGPU, ~150 ms). SAM can't be
      exported, so added OpenVINO FastSAM / YOLO11-seg backends.
- [x] `GET /methods` + `method` on `/scan`: a registry of SAM1 / SAM2 / FastSAM /
      YOLO / DEPTH with auto GPU-or-CPU device selection. The client renders it
      as a long-press radial wheel; depth replaces the camera feed.
- [ ] Optional: CLIP zero-shot classifier on crops; Florence-2 caption (async).

Stage B (phone side) - anchoring:

- [x] AR: capture a frame (`camera-access`), hit-test distance, `/scan`, scale by
      `hitTestDistance / depthAtCentre`, world-transform, create an `XRAnchor`
      with the 3D outline + OBB, and relay it to the desktop viewer.
- [ ] Tune the camera-image FOV (intrinsics aren't exposed by camera-access).
- [ ] Move the wheel into AR (currently a Scan button).

**Stage A done when:** a photo gives labels back in < ~150 ms. (achieved, ~20 ms)
**Stage B done when:** the aimed object yields a camera-space contour + OBB. (achieved)

**De-risk here:** OpenVINO export/quantisation path, iGPU vs CPU device selection.

---

## Milestone 3 - Dense world model (2-2.5 weeks)

> **Re-scoped.** Single-view lifting is proven (`docs/LIFTING.md`) but has a hard
> ceiling (one-sided shell, frozen noise). We are moving to a **fused** map:
> stream keyframes, fuse depth into a world voxel/TSDF, then segment objects.
> Full plan + milestones: [`docs/RECONSTRUCTION.md`](RECONSTRUCTION.md).
> Why we do **not** bolt on ORB-SLAM3 / SVO Pro (ARCore already does VIO, and
> classic SLAM maps are sparse): [`docs/SLAM_SURVEY.md`](SLAM_SURVEY.md).

Single-view lifting (Stage B) - **done, now the close-up path**:

- [x] Depth Anything V2 (OpenVINO iGPU, ~150 ms), inverse-depth fix, robust
      depth so outlines aren't jaggy.
- [x] Unproject mask ∩ depth -> camera-space `contourCam` + `obbCam`; phone
      applies hit-test scale and anchors it. See `docs/LIFTING.md`.

Fused reconstruction (the new work):

- [x] ARCore **planes** streamed device -> gateway -> viewer and rendered as
      coloured polygons (toggle with `P`). This is the `polygon`/`orientation`
      data MS3.3 needs, plus a live sanity check on the shared world frame.
- [x] **MS3.0** keyframe streaming: motion/rate/blur-gated captures (≤3 Hz) with
      pose + fov are sent device -> gateway -> inference; the gateway broadcasts
      metadata to viewers (breadcrumb frustums) and `GET /recon/status` reports
      counts. Images ride as base64 JSON (see the protocol note).
- [x] **MS3.1** metric scale: robust affine fit `Z = a·(1/d_rel) + b` against
      ARCore's metric planes, with a pure-scale fallback when only one depth
      level is visible and a cached fit when nothing is. (`app/fusion.py`)
- [x] **MS3.2** voxel fusion in the ARCore frame (running-mean `VoxelMap`), with
      a decimated snapshot streamed to the viewer, which renders it as points.
      Verified end-to-end on the sample image; no GPU needed (see `docs/FUSION.md`).
- [ ] **MS3.3** structure: ARCore + RANSAC planes, excluded from the object pool.

> **Design note - Metroid Prime fidelity (affects MS3.4/MS3.6 ordering).** In the
> game the visor *reveals* what is scannable: a thing lights up with its scan
> outline and only then can it be scanned. So the order is
> **fused map -> object candidates -> reveal each candidate as scannable
> (outline + label) -> the user scans it for detail**. It is *not* "user aims at
> a point and we check whether it happens to be an object". The reveal is
> therefore part of MS3.4's output, not a later polish step.
- [ ] **MS3.4** DBSCAN clustering + scoring + tracking -> candidate objects with IDs.
- [ ] **MS3.5** multi-view render -> CLIP labelling.
- [ ] **MS3.6** anchor promotion on the phone (fused objects -> scanable HUD targets).

**Done when:** walking around a room produces a recognisable point cloud on the
desktop, a chair/desk appears as a single tracked candidate with a label, and the
phone can anchor + scan it.

**De-risk here:** metric scale (the plane fit), pose/frame sync, fusion memory,
depth warping.

---

## Milestone 4 - Mobile HUD / scan visor (1 week)

- [ ] Screen-space overlay canvas drawing projected OBB edges in class colours.
- [ ] Centre reticule; raycast against OBBs; hover -> highlight + info card.
- [ ] "Scan" button: capture frame under reticule -> server -> materialise outline + card.
- [ ] Info card UI (name, class, confidence, description) in `AdvancedDynamicTexture` or DOM.
- [ ] Animations / SFX for the Metroid feel (optional but great for marks).

**Done when:** walking around a scanned object keeps its outline glued in place, aiming at
it re-shows its card, and the list of scans is visible.

---

## Milestone 5 - Desktop viewer + reconstruction (1 week)

- [ ] WS client subscribes to world deltas; renders OBBs + labels.
- [ ] Basic point-cloud/TSDF accumulation and render (splat discs or mesh).
- [ ] Click to select -> info panel; ArcRotateCamera navigation.
- [ ] Visual polish: grid, axes, scan-highlight pulse.

**Done when:** the desktop shows the reconstructed scene with all scanned objects
highlighted and selectable, updating live as the phone scans.

---

## Milestone 6 - Polish, demo, report (0.5-1 week)

- [ ] Tune lighting/materials/outlines; handle empty/error states.
- [ ] Capture demo video (phone + desktop side by side).
- [ ] Write-up: architecture, model choices, evaluation (latency, accuracy anecdotes),
      limitations, future work (3DGS/NeRF, multi-device, iOS).

---

## Work breakdown by area

### Frontend (`apps/client`)
| Task | Notes |
|---|---|
| Babylon scene + engine | shared between AR and viewer |
| WebXR AR session | `WebXRFeaturePointSystem`, `WebXRHitTest`, `WebXRAnchors` |
| Pose streaming | snapshot at capture time, not on send |
| HUD overlay | 2D canvas outlines + reticule + cards |
| Viewer | orbit camera, OBB render, selection, point cloud |

### Server (`apps/server`)
| Task | Notes |
|---|---|
| WS hub + session mgmt | Fastify + `ws` |
| Frame intake + dedupe | pose + pHash key |
| Inference client | HTTP to Python, timeouts, retries |
| World model | authoritative objects, fusion, deltas |
| Broadcast | to viewers + back to device (labels/anchors) |

### Inference (`services/inference`)
| Task | Notes |
|---|---|
| Model loading | OpenVINO, `AUTO` device, caching |
| `/scan` | detect + segment |
| `/depth` | monocular depth |
| `/classify` | CLIP zero-shot |
| `/describe` | Florence-2 / BLIP caption |
| Bench + accuracy harness | latency + qualitative gallery |

---

## Suggested first sprint (next 1-2 sessions)

1. Scaffold the three packages + shared types, get `npm run dev` green (M0).
2. Stand up a Babylon `immersive-ar` session served over LAN HTTPS, with hit-test +
   one anchored cube (M1 core).
3. Add a `/health` + `/scan` stub that returns a *hardcoded* detection, and render it as a
   screen-space box on the phone HUD. End-to-end loop closed *before* real ML.
4. Then swap the stub for YOLO11n + FastSAM.

---

## Open questions

- Is WebXR Depth Sensing available on the target phone? (Check `/capabilities.html`.)
  If yes -> use it for lifting and skip monocular depth on-device. If no -> Depth Anything
  V2 small on the server.
- Do we want 3D outlines rendered in-scene (correct occlusion) or 2D overlay (simpler)?
  **Decided: 2D overlay first, occlusion out of scope.**
- How much reconstruction fidelity is needed? **Decided: coarse boxes/planes are fine;**
  prioritise the scan-visor HUD and the segmentation/classification pipeline.
