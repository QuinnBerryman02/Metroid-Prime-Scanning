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

**Identifiability matters more than the fit.** A least-squares line through
samples that barely vary in depth returns a *technically positive* `(a, b)` pair
with a tiny `a` and a huge `b`. That passes any `a > 0` sanity check, and then
warps every frame fused with it. Measured on-device: `a=0.081 b=1.543` - the
whole room collapsed to ~1.6 m and the map smeared out to ~1 M voxels. So:

- `b` is only fitted when the visible plane samples span **>= 1 m** of depth;
  otherwise assume `b = 0`. That is not a loss of information: a single
  fronto-parallel wall still pins `a = Z / rel` exactly.
- an affine fit whose `b` swamps `a` is refused and replaced by the `b = 0` one.
- accepted fits go into a short history and the scale used for fusion is the
  running **median** (`ScaleTracker`). This is the important one: even with good
  per-frame fits, fusing with a scale that *jumps frame to frame* smears every
  surface into the space around it. Stability beats per-frame optimality.

A frame with no usable planes keeps the tracked scale rather than failing - but
the viewer is told, so "no planes yet" never looks like a bug.

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

### The fusion loop, concretely

For each keyframe `k` (RGB `I_k`, pose `T_k` = world←camera, intrinsics `K`):

```
d = 1 / disparity(I_k)                      # relative inverse-depth
d = median_blur(d, 5)                       # the same robustness as /scan
Z = a_k * d + b_k                           # metric depth (fit in section 6)

for (u, v) in subsample(pixels):            # every 2nd pixel ~ 150k samples
    z = Z[v, u]
    if not plausible(z): continue           # skip sky / far / NaN / huge steps
    p_cam   = ( (u-cx)/fx * z, (v-cy)/fy * z, z )
    p_world = T_k @ p_cam

    key = floor(p_world / voxel)            # 2 cm voxels
    w, mu, c = voxels[key]
    mu = (mu*w + p_world) / (w+1)           # running mean position
    c  = (c *w + I_k[v,u]) / (w+1)          # running mean colour
    voxels[key] = (w+1, mu, c)
```

**Density is free; consistency is the hard part.** A 640x480 frame is ~300k
depth samples, so a single keyframe is already dense - it is just noisy and
one-sided. Fusion does not add samples, it makes them *agree*: N observations of
the same patch land in the same voxel and their errors average down ~`1/sqrt(N)`.
That is why a fused wall is smooth where a single-frame wall was jagged.

### Non-axis-aligned surfaces (your sloped attic ceiling)

ARCore's plane detector is **binary: horizontal or vertical**. A pitched roof is
neither, so ARCore will never report it as a plane - which is exactly why the top
half of your room is missing from the plane set. Depth fusion has no such
assumption: monocular depth is just geometry, so it reconstructs the slope
naturally. **The dense layer is strictly richer than the plane layer for anything
not axis-aligned** - which, in an attic, is most of the interesting geometry.

### What actually limits quality

1. **Metric scale drift** (`a_k, b_k`) - the biggest one; a bad fit warps the
   whole frame relative to the map.
2. **Pose error** - a small rotation error smears a surface over `dist * angle`.
3. **Depth-model bias** - DA V2 is trained on real photos; it degrades on
   textureless walls, glass, and strong reflections.
4. **Coverage** - surfaces only exist where the camera has actually looked.

Note the asymmetry: 1-3 are *smooth* errors (they warp the map), while 4 is a
*gap*. Warps are visually forgivable in an AR overlay; gaps read as holes.

### Cost (measured)

`scripts/bench_voxel.py` integrates a synthetic room (6 surfaces, one real
diagonal roof), 150 k points/frame, 90 frames, 3 cm per-frame depth noise:

| voxel | ms / frame | % of one core @3 Hz | voxels | arrays | obs / voxel |
|---|---|---|---|---|---|
| 2 cm | ~180 | ~54% | 1.64 M | 59 MB | 8.3 |
| 5 cm | ~55 | ~17% | 148 k | 5.3 MB | 91 |
| 10 cm | ~36 | ~11% | 25 k | 0.9 MB | 535 |

**No GPU is needed.** `integrate()` is pure vectorised numpy - `np.unique` to
bucket points into voxels, `np.bincount` for the per-voxel position/colour sums,
then `np.searchsorted` + `np.insert` to merge into the sorted global key array.
The iGPU is already busy with the depth model, and the CPU has headroom: at 3 Hz
even the 2 cm case is about half of one core.

Note the *shape* of the trade: finer voxels cost super-linearly, because a
surface's voxel footprint grows as `1/voxel²` while the noisy observations spread
it *thicker*. So 2 cm is the fidelity target, and 5 cm is the pragmatic fallback
if the budget tightens (still fine for segmentation).

Caveat: the benchmark's point-to-truth residual is a crude proxy and came out
non-monotonic (2 cm 3.5 cm, 5 cm 5.6 cm, 10 cm 2.7 cm) because larger voxels
straddle more surface *edges*, where a voxel mean legitimately sits inside the
corner. Trust the timing / memory / obs-per-voxel columns, not that one.

### Tuning loop: record a run, replay it offline

On-device iteration is far too slow to explore a parameter space (one tap, one
cloud, one opinion). So the service records every keyframe:

```
services/inference/recon_log/<session>/NNNN.{jpg,json}    (RECON_RECORD=1)
```

There are two ways to consume it:

1. **In the viewer** (fastest loop): the bottom-left panel lists recorded runs.
   *Replay* drives them through the **live** gateway pipeline, so the viewer sees
   them exactly as if a phone were walking the room - keyframes, planes, the
   fused cloud and the pose breadcrumbs, using the same messages as a real
   session. Backed by `GET /recon/logs`, `POST /recon/replay`,
   `POST /recon/replay/stop` on the gateway. A replay never re-records itself.
2. **Offline** for parameter sweeps: `scripts/replay_recon.py <dir>` re-fuses the
   same frames under different scales, voxel sizes, strides and methods, caches
   the depth maps as `NNNN.npy` so repeat runs are fast, and can dump a coloured
   PLY (`--ply`) of any variant.

> `recon_log/` is **user data, not scratch space**. Never `rm -rf` it in tests or
> scripts - use a separate directory (e.g. `RECON_DIR=/tmp/recon_test`) if you
> need a throwaway.

**A caution that cost us a wrong turn.** The obvious objective - score a scale by
how compact the fused map is - is *degenerate*: shrinking the scale always packs
points into fewer voxels, so it monotonically prefers small `a`. That is not a
bug but a property of the problem: under a purely geometric objective the global
scale of a monocular reconstruction is **unobservable** (the same reason
monocular SLAM has scale ambiguity), and a hand-held walk-around has too little
parallax to break it. **Global scale must come from a metric reference** -
ARCore's planes or its hit-test. Replay is for comparing *consistency* of
per-frame scale sources, and for eyeballing methods - not for searching for scale.

### Why TSDF is the real target (not just the accumulator)

The measurement above shows the naive accumulator **does** average (8 obs/voxel
at 2 cm). The problem is what it averages *to*: at 2 cm it produced **1.6 M
voxels for a room with roughly 160 k of true surface** - a ~10x-thick shell.
Averaging shrinks the noise but never collapses the shell, because a voxel is
filled by points that legitimately belong to the surface on either side.

TSDF fixes that structurally: every depth measurement votes a *truncated signed
distance* over a band of voxels along its ray, and the surface is recovered as
the **zero crossing** of the weighted average. So:

- resolution and averaging decouple (no need for voxel > noise),
- the output is a **surface**, not a thick point shell,
- `extract_triangle_mesh()` gives an actual mesh for the viewer.

Cost is comparable (single pass, C++ in Open3D), and the inputs are identical to
the accumulator - so it is a drop-in swap once the loop works end-to-end.

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
