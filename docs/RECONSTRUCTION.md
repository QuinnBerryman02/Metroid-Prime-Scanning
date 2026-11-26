# Dense reconstruction plan (single-view → fused map)

> Supersedes the "Stage C: fuse" sketch in [`docs/LIFTING.md`](LIFTING.md §7).
> The SLAM/odometry decision behind it is in [`docs/SLAM_SURVEY.md`](SLAM_SURVEY.md).

## 1. TL;DR

Stop unprojecting one frame. Instead:

1. The phone **streams keyframes** (JPEG + metric pose + intrinsics) while the
   user walks around - the pose comes from **ARCore via WebXR** (we already read it).
2. The inference service runs **monocular depth** on each keyframe, recovers
   **metric scale** from ARCore's planes, and **fuses** the points into a
   world-space **voxel map** (→ TSDF later).
3. Every couple of seconds the map is **segmented**: RANSAC/ARCore **planes** =
   structure; the residual is **Euclidean-clustered** into object candidates.
4. Candidates are **tracked** across frames; stable ones are **classified** by
   rendering the cluster from several views and running **CLIP**.
5. Confirmed objects are sent to the phone as `WorldObject`s in the **same
   ARCore frame**, and the phone drops an `XRAnchor` on each → aim at it → scan.

Why this is easier than it sounds: fusion with **accurate poses** is a standard,
well-behaved operation (weighted averaging in a voxel grid). The single-view
jaggedness we just fought is *depth noise*, and **averaging many views of the
same surface cancels most of it**. Fusion is the cure, not the complication.

## 2. Why single-view has a ceiling

- One frame = one direction → a **one-sided shell**, never a closed object.
- Monocular depth is **relative and noisy**; `1/disparity` amplifies far
  outliers (this is exactly the mesh-chair spikiness).
- No way to decide "is this a *thing*?" from one view - a box could be part of a
  wall, a shadow, a person.
- Every artefact is frozen into the anchor at scan time.

A fused map fixes all four: multiple directions, noise averaging, **structure vs
object** reasoning, and re-observation before promotion.

## 3. Architecture / data flow

```
PHONE (AR)                          GATEWAY (TS)                 INFERENCE (Python)
──────────                          ────────────                 ──────────────────
XRView pose ──────┐
ARCore planes ────┤  keyframe (≤3 Hz)
 camera JPEG ─────┴──────────────►  recon_frame  ──────────►  ┌──────────────────────┐
                                 (JSON + JPEG)               │ depth (DA V2, iGPU)  │
                                                             │ scale fit (planes)   │
                                                             │ unproject + fuse →   │
                                                             │   voxel map          │
                                                             │ every ~2 s:           │
                                                             │   planes + DBSCAN →   │
                                                             │   candidates → track  │
                                                             │   render views → CLIP │
                                                             └──────────┬───────────┘
                                                                        │ WorldObject[]
  XRAnchor per object ◄──────────── world_delta ◄──────────────────────┘
                                          │
                                          └──────────►  VIEWER (TS/Web) renders the live
                                                        point cloud + object boxes
```

Two data planes, as now: **WS** for the live loop, **HTTP** for one-shot `/scan`.

## 4. The shared world frame

**ARCore's `local` reference space is the canonical world frame.** No alignment
math is needed:

- the phone streams `worldFromCamera` **in `local`** (it already does for the
  pose relay);
- the server fuses, segments and returns `WorldObject`s **in `local`**;
- the phone anchors them at the returned transform, directly.

Rules (from `ARCHITECTURE.md §4`): every keyframe carries the pose **at capture
instant**; every `WorldObject` carries its `WorldTransform`. For a single device
we never need a `worldFromDevice` correction; add one field later if we ever run
two phones.

## 5. Streaming & keyframe selection

Don't stream every frame - gate to keep the iGPU fed but not saturated.

A frame is a **keyframe** when all hold:

- ≥ 250 ms since the last one (rate cap ≈ 4 Hz, target 2-3 Hz);
- camera moved ≥ 2 cm **or** rotated ≥ 1.5° since the last keyframe (adds
  baseline → real parallax, which is what makes fusion useful);
- **blur gate:** Laplacian variance of the greyscale frame > threshold (skip
  motion-blurred frames - they poison depth).

Transport: a `recon_frame` JSON header + a binary JPEG body, over the existing
WS hub (mirrors the current `scan_frame` design). Payload ≈ 30-60 KB at 640 px;
@ 3 Hz ≈ 150 KB/s on LAN. Fine.

Keyframe gating lives on the **phone** (it knows motion + blurriness for free).

## 6. Metric scale (the one hard bit)

Depth Anything V2 is scale-invariant, so `1/disparity` is proportional to depth
*up to an affine transform*. Fit it per keyframe against **ARCore's metric
planes** - no new model, and ARCore is already our best metric source:

For a pixel `(u,v)` on a detected plane with metric normal `n`, offset `c`:

```
ray      r = K⁻¹·[u, v, 1]            # camera-space direction
Z_metric = c / (n · r)                # exact metric depth of the plane point
d_rel    = disparity(u, v)            # model output
```

Collect plane samples across all visible planes and least-squares fit

```
Z_metric ≈ a · (1 / d_rel) + b        # a, b per keyframe
```

The affine term matters: DA V2's output is inverse-depth-like, so a pure scale
(`b = 0`) drifts with distance. This fit is 2 unknowns from hundreds of samples.

Fallbacks, best first:

1. ≥ 2 planes with different depths → affine fit (above).
2. 1 plane → pure scale `a`, `b = 0`.
3. no plane → the **hit-test** scale we already use (`s = D / depth@reticle`);
   reuse the last good `(a,b)` if the user hasn't moved far.
4. keep the phone's existing 1.5 m default as a last resort.

Store `(a,b)` per keyframe; if the fit residual (median |error|) is bad, drop the
keyframe.

> Future upgrade: a **metric** depth model (e.g. Depth Anything V2 Metric /
> Metric3D) conditioned on the known focal length would remove this step
> entirely. Deferred: it is a different checkpoint + focal plumbing, and the
> plane fit is already robust.

## 7. Dense map representation

**Start: voxel accumulator** (~40 lines, no exotic deps).

- Quantise world points at a voxel size (start 2 cm).
- Each voxel keeps a **running mean** position, **running mean** RGB, and a
  **weight** (observation count).
- `voxel = floor(p / vs)` → dict key (or a hashed integer key).
- Rendering = the voxel means; segmentation = the same array.
- Cap the total voxels (e.g. 500 k) → drop the lowest-weight voxels.

This *is* the noise-averaging: two noisy observations of the same surface land in
the same voxel and average out, so the fused map is dramatically smoother than
any single frame.

**Upgrade: TSDF** via `open3d.pipelines.integration.ScalableTSDFVolume`:

```python
vol.integrate(rgbd, intrinsic, extrinsic_world_to_cam)   # per keyframe
cloud = vol.extract_point_cloud()
mesh  = vol.extract_triangle_mesh()
```

Better surfaces + hole filling + a mesh for the viewer. Switch once the voxel
path proves the loop end-to-end (same inputs, so it is a drop-in swap).

> Dependency note: `open3d` 0.20 is installable here (Python 3.11), but verify it
> against **numpy 2.4** before committing - pin if needed. It is optional for
> MS3.2 (the voxel accumulator is ours) and required for the TSDF upgrade.

## 8. Structure extraction

Two complementary sources:

1. **ARCore planes** (`XRPlane` from plane-detection, already enabled): metric
   polygons with orientation (horizontal/vertical). Free, tracked, labelled
   floor/wall/table. **Stream them** with each keyframe.
2. **RANSAC planes on the fused cloud** (`open3d` or a 30-line SVD fit): catches
   surfaces ARCore did not promote to a plane.

Product: a **structure set** (floor + walls + big horizontals). Those points are
removed from the object pool - this is the "planes act as edges" idea, made
concrete: walls/floor delimit what can be an object.

## 9. Object segmentation & tracking

On the structure-free residual:

1. **Voxel-downsample** (2-3 cm) and estimate normals.
2. **Euclidean clustering** = `sklearn.cluster.DBSCAN(eps≈0.03-0.04, min_samples≈30)`
   (already a dependency; `open3d.cluster_dbscan` if we take Open3D). Density
   threshold = your "density then edge" intuition, exactly.
3. **Score** each cluster: point count, bounding-box volume, mean density,
   height above the floor plane, distance to the camera trajectory.
4. **Track** clusters frame-to-frame: greedy match on **centroid distance + box
   IoU** → stable `id`; update `lastSeen`, `scanCount`, running centre/size.
5. **Promote** to a `WorldObject` when: `pointCount ≥ N` (e.g. 400),
   `observations ≥ 3` (seen from ≥ 2 distinct viewpoints), size in a plausible
   range, and (from §10) classification confidence above threshold.
6. **Prune**: objects unseen for a while, or that turned out to be structure.

This is the classic robotics recipe (plane segmentation + Euclidean clustering),
which is exactly why it is trustworthy: it is boring, robust, and tunable.

## 10. Classification (multi-view → CLIP)

For a promoted cluster:

1. Pick K viewpoints (K ≈ 6) on a ring around the cluster: radius ≈ 1.5× box
   diagonal, at the object's height, all looking at its centre, plus one top-down.
2. **Render** each view by projecting the cluster's coloured voxels into a small
   image with numpy/cv2 (back-to-front splats, ~10 px discs, neutral background,
   cropped to the projected bbox). No OpenGL needed → works headless.
   *(Fallback: `open3d` OffscreenRenderer if a GL context is available.)*
3. Run **CLIP** zero-shot against a prompt list per view; **average** the text
   similarities across views → label + confidence. Multi-view voting kills the
   pose-sensitive CLIP errors a single crop would make.
4. Optional: Florence-2 / BLIP one-liner description for the info card.
5. Clip per-object views as the `WorldObject`'s thumbnail for the visitor HUD.

CLIP is already validated in `notebooks/`; this reuses it.

## 11. Anchoring back to the phone

Server → phone (`world_delta`, in the ARCore frame):
`WorldObject { id, center, size, quaternion, label, color, confidence }`.

Phone, on receiving:

1. Build a rigid transform from `center` + `quaternion` in the reference space.
2. `frame.createAnchor(transform, refSpace)` → an `XRAnchor` that survives
   relocalisation (we already use this API).
3. Attach the OBB wireframe + label to the anchor; render on the HUD.
4. Aiming at it (existing reticle raycast) shows the info card and offers a
   high-quality **`/scan`** close-up, which is still useful for a crisp
   segmentation of one object.

Because the server works **in the ARCore frame**, no transform fiddling is
needed - the anchor lands exactly where the fused geometry is.

## 12. Compute budget (Intel Core Ultra 7 165U, iGPU)

| Step | Cost | Rate | Load |
|---|---|---|---|
| Depth (DA V2, OpenVINO iGPU) | ~150 ms | 2-3 Hz | ~40% |
| Scale fit + unproject + voxel fuse | ~20-40 ms | 2-3 Hz | ~10% |
| Plane RANSAC + DBSCAN + scoring | ~200-500 ms | every ~2 s | ~20% |
| CLIP on ~6 crops (only on promotion) | ~100 ms | rare | - |

Comfortable. Levers if tight: lower JPEG/working resolution, drop the depth rate,
reduce the voxel size only when needed.

## 13. Milestones

| # | Deliverable | Done when |
|---|---|---|
| **MS3.0** | Keyframe streaming: phone sends JPEG + pose + intrinsics + planes at ≤3 Hz; gateway relays to inference; a `GET /recon/status` reports keyframe count | viewer shows the streamed keyframe cameras in the world; counts match |
| **MS3.1** | Metric scale: affine `(a,b)` fit from ARCore planes; reported + validated against hit-test | fused cloud is metrically consistent (a 1 m ruler measures ≈1 m) |
| **MS3.2** | Voxel fusion in the ARCore frame; viewer renders the **live growing point cloud** | walk around a room → a recognisable cloud appears on the desktop |
| **MS3.3** | Structure: ARCore planes + RANSAC planes extracted and visualised distinctly; structural points excluded from the object pool | floor/walls appear as planes, not noise |
| **MS3.4** | Object clustering + scoring + tracking → candidate objects with stable IDs | a chair/desk appears as one candidate with one ID across views, no duplicates |
| **MS3.5** | Multi-view CLIP labelling of candidates | candidates carry correct-ish labels in the viewer |
| **MS3.6** | Anchor promotion on the phone: aim at a fused object → info card + a crisp `/scan` close-up | the Metroid HUD works on the *fused* world model, not just on fresh scans |

MS3.0-MS3.2 are the demo money shot ("walk around → 3D model"). MS3.3-MS3.5 are
the segmentation/classification payload. MS3.6 ties it back to the HUD.

## 14. Risks & fallbacks

| Risk | Mitigation |
|---|---|
| Monocular depth is *approximate* - the cloud is warped | affine scale fit; voxel averaging; accept approximate geometry (non-goal per ARCHITECTURE.md) |
| ARCore pose drift → smeared cloud | short effective baselines, drop keyframes on bad fits, re-anchor after relocalisation; pose-graph later |
| Dynamic objects (people) smear into the map | visibility/motion filtering (hard); or accept + document |
| Thin/transparent geometry (mesh chairs!) | depth is genuinely poor there; robust voxel weights + min-observation gating |
| `open3d` vs numpy 2.4 | verify/pin early; TSDF is optional (MS3.2 is dependency-free) |
| Memory growth | voxel cap by weight |
| Plane fit underdetermined (one plane visible) | pure-scale fallback → hit-test fallback (already shipped) |

## 15. Protocol changes (sketch)

Add to `packages/shared/src/protocol.ts`:

```ts
// device -> server
interface ReconFrameMsg {
  type: "recon_frame";
  sessionId: string;
  frameId: string;
  pose: Mat4;                 // worldFromCamera, ARCore `local`
  intrinsics: Intrinsics;     // from the XR projection matrix + viewport
  planes: PlaneMsg[];         // metric ARCore planes visible now
  jpegBytes: number;          // binary body follows
}
interface PlaneMsg {
  id: string;
  center: Vec3;
  normal: Vec3;               // world space
  polygon: Vec3[];            // world space, metric
  orientation: "horizontal" | "vertical";
}

// server -> viewer / device
interface ReconStatusMsg {
  type: "recon_status";
  keyframes: number;
  voxels: number;
  candidates: number;
  scale: { a: number; b: number } | null;
}
// reuse WorldDeltaMsg for objects (already defined)
```

## 16. What we keep

- **`/scan`** stays - it is the deliberate "aim + scan close-up" path and the
  fallback when no fused map exists yet.
- The **AR HUD**, radial wheel, anchors, outline rendering, viewer, diagnostics -
  all unchanged.
- `camera-access` capture, FOV/pose handling, `report.ts` telemetry.
- The depth model + OpenVINO path (now called per keyframe instead of per scan).

What is **replaced**: the single-view `contourCam`/`obbCam` lifting as the
*primary* world-modelling path. It becomes the close-up refinement on top of the
fused map.
