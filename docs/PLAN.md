# Implementation Plan

## 0. Guiding principle

Build a **thin vertical slice first**, then deepen it. The riskiest unknowns are
(a) WebXR camera + anchors on the phone and (b) 2D-mask -> 3D-anchor lifting. Prove both
early with placeholder ML (hardcoded detections) before touching real models.

Suggested overall effort for a course-sized scope: ~4-6 focused weeks.

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

- [ ] Babylon `WebXR` default experience: `immersive-ar`, `local-floor`, DOM overlay.
- [ ] Camera passthrough + `<video>`/raw-camera fallback; confirm it works over LAN HTTPS.
- [ ] Hit-test / sphere at tapped feature points; draw debug spheres.
- [ ] Anchors: place a static test marker in the world, walk around it, confirm it stays put.
- [ ] Send pose (world matrix) to server each frame; server echoes; viewer shows a moving
      frustum gizmo (proves coordinate frames + transport end to end).

**Done when:** a static cube placed in AR stays anchored while moving, and a desktop
viewer shows the phone's live pose in a shared world.

**De-risk here:** secure context, Android Chrome quirks, anchor lifetime, frame sync.

---

## Milestone 2 - Detection + segmentation service (1 week)

- [ ] FastAPI `/scan` endpoint: accepts JPEG (+ optional intrinsics), returns detections.
- [ ] YOLO11n detection via OpenVINO; then FastSAM-s (or MobileSAM) for masks.
- [ ] Post-process to a stable JSON schema (`Detection {box, mask{rle|png}, score, label}`).
- [ ] Test harness: run over a folder of stills, visualise overlays, measure latency.
- [ ] Optional: CLIP zero-shot classifier on crops; Florence-2 caption (async).

**Done when:** a photo gives masks + labels in < ~150 ms on the target machine and results
look sensible across a handful of test photos.

**De-risk here:** OpenVINO export/quantisation path, iGPU vs CPU device selection.

---

## Milestone 3 - 2D -> 3D lifting & world model (1-1.5 weeks)

- [ ] Depth service (Depth Anything V2 small) or consume WebXR depth if available.
- [ ] On the server: unproject mask ∩ depth using pose + intrinsics -> world points.
- [ ] Reduce points to an OBB/hull; create a `WorldObject`.
- [ ] Fusion: match new detections to existing objects (centroid + IoU) -> update vs add.
- [ ] Persist session world state; broadcast deltas.

**Done when:** scanning the same object twice updates one entry (no duplicates), and its
OBB sits on the object in the world frame.

**De-risk here:** intrinsics (WebXR `XRWebGLLayer.getViewport` + projection matrix
inversion), depth scale, robust point reduction.

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

- Is WebXR Depth Sensing available on the target phone? Determines whether we need the
  monocular depth model at all for lifting.
- Do we want 3D outlines rendered in-scene (correct occlusion) or 2D overlay (simpler)?
  Plan: 2D first, upgrade if time.
- How much reconstruction fidelity is actually needed to score on the rubric's
  "detailed 3D scene / custom models"? If not much, invest in the scan-visor HUD and
  model pipeline instead (more distinctive + more marks via interaction/innovation).
