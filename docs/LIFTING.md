# 2D mask -> 3D anchor ("lifting")

The central design question: segmentation gives us a **2D mask**, but we need
something that lives in the **3D world** and stays put when the user moves. This
doc explains how we get from one to the other, what we actually anchor, and
where the API boundary sits.

## 1. The one-line version

> A mask tells us *which pixels*. Depth + intrinsics turn those pixels into 3D
> points in the **camera's** frame. The camera **pose** (from WebXR) turns those
> into the **world** frame. We anchor the world-space geometry, not the 2D mask.

Anchoring a flat 2D billboard would be "basic" because it does not know how far
away the object is, so it would slide/scale wrongly as you move. The moment we
unproject through depth we get real 3D positions, and those behave correctly
under head/camera motion.

## 2. The pipeline

```
 keyframe (RGB) ─┬─► detector (YOLO11n) ──────────► box + label
                 ├─► segmenter (SAM 2.1, prompted) ► mask
                 └─► depth (Depth Anything V2) ────► depth map
                                                       │
 mask + depth + intrinsics ──► unproject ──────────────┘
                                 │
                                 ▼
                     points in CAMERA space (x, y, z)
                                 │
        pose (WebXR, world<-camera) ──► points in WORLD space
                                 │
                                 ▼
              reduce to an anchorable shape (see §3)
                                 │
                                 ▼
                 XRAnchor at the object centroid
```

### The unprojection maths

For each masked pixel `(u, v)` with depth `d(u,v)` and intrinsics
`(fx, fy, cx, cy)`:

```
z = s * d(u, v)                     # s = scale from relative -> metres
x = (u - cx) * z / fx
y = (v - cy) * z / fy
P_cam = (x, y, z)
```

then, with the pose captured **at the same instant** as the frame:

```
P_world = world_from_camera · P_cam
```

The mask is only used to pick *which* pixels. Everything downstream is 3D.

> **Depth Anything V2 outputs inverse depth (disparity, larger = nearer).**
> The service must use `d = 1 / disparity` before this step
> (`DepthEstimator.predict_depth`). Using the raw output as depth inverts the
> object inside-out and puts it at the wrong distance.

## 3. What do we actually anchor?

The raw point cloud is heavy and noisy. We reduce it to something compact and
stable. Three candidate representations, in increasing order of fidelity:

| Representation | What it is | Used for |
|---|---|---|
| **OBB** | oriented bounding box: centroid + size + quaternion | reticle hit-testing, desktop viewer highlight, the "boundary" |
| **3D contour** | the mask boundary, unprojected to a 3D polyline | the smooth glowing outline on the phone |
| **surface patch** | downsampled coloured point cloud | optional: a textured "shell" for the viewer |

Recommendation: **anchor the centroid with an OBB and a 3D contour.** The OBB is
what the reticle raycasts against; the contour is what we render as the coloured
outline. The full point cloud is optional (nice for the desktop viewer).

All of these are stored **relative to the object centroid**, so when the anchor's
pose updates (relocalisation, tracking refinement) the whole object moves with
it.

### Making the OBB
1. Take the masked world points.
2. Robust-centre them (median / trimmed mean).
3. PCA to get principal axes -> orientation.
4. Extent along each axis = a low/high quantile (e.g. 2%-98%) so a few stray
   points don't blow up the box.

### Making the 3D contour
1. `cv2.findContours(mask)` -> ordered boundary pixels.
2. Simplify (`approxPolyDP`, e.g. ~24 points).
3. Unproject just those boundary pixels (same maths) -> `Vector3[]`.
4. Store relative to the centroid.

## 4. The scale problem (important)

Depth Anything V2 gives **relative** depth, so `s` is unknown and the object
will be the right *shape* but the wrong *size*. Options to fix it, best first:

1. **AR hit-test ray.** The user aims the reticle at the object, so we can cast
   a ray from the camera through screen-centre and ask WebXR for the distance
   `D` to the real surface. Then `s = D / d(centre)`. This gives a metric scale
   for free, per scan.
2. **Resting plane.** If the object sits on a detected plane, match the object's
   base to the plane's world height and solve for `s`.
3. **Known size.** "A mug is ~10cm" - works for a fixed taxonomy only.
4. **Metric depth model** (UniDepth / Metric3D) - better, heavier; a later
   upgrade.

For the MVP we use (1), falling back to a default distance (~1.5 m) when the
hit-test misses.

> Note: the phone (not the server) should apply the scale, because the phone is
> the one that can hit-test.

## 5. Where the API boundary is

Key insight: **the server can stay pose-agnostic.** It works in camera space and
does not need to know where the phone is. The phone owns the pose and the XR
anchor API.

```
POST /scan  (phone -> server)
   image (jpeg), intrinsics, aim=(0.5, 0.5)
server:
   detect -> for each box: label, confidence
   segment (prompted at aim) -> mask
   depth -> relative depth
   lift the mask to CAMERA-space geometry (using intrinsics only)
   reduce -> { box, mask, contour3d, obbCandidates, depthScaleHint }
returns:
   detections + camera-space geometry for the aimed object
```

```
phone (client-side, no server):
   s = hitTestDistance / depthAt(centre)
   P_world = world_from_camera * (s * P_cam)
   create XRAnchor at centroid
   attach OBB mesh + 3D contour line
```

So: **lifting is a distinct processing stage, but not necessarily a separate
HTTP endpoint.** It needs the same frame as detection/segmentation, so it should
ride back in the `/scan` response. A separate `/lift` endpoint would only make
sense if the client wanted to lift many objects from one frame, or re-lift with a
different scale - both are cheap to do client-side, so we keep it in `/scan`.

### Indicative response
```jsonc
{
  "frameId": "…",
  "detections": [
    { "classId": 39, "label": "bottle", "score": 0.91,
      "box": [x, y, w, h],
      "mask": { "encoding": "rle", "data": "…" } }
  ],
  "aimed": {                          // the object under the reticle
    "classId": 39,
    "centroidCam": [x, y, z],         // camera space, relative depth units
    "contourCam": [[x,y,z], …],       // N boundary points, camera space
    "obbCam": { "center": [x,y,z], "size": [w,h,d], "quat": [x,y,z,w] },
    "depthAtCentre": 3.42,            // relative; phone rescales to metres
    "scaleHint": { "method": "hit-test" }
  }
}
```

## 6. Known limitations (be honest about these)

- **Single view => a shell from one direction.** Unprojecting one frame only
  ever yields the visible surface. That is exactly what we want for an outline,
  but it is not a full 3D model. Re-scanning from another angle should *fuse*
  (mesh in the project's priority order: point cloud first).
- **Relative depth + scale.** Handled by §4, but drift/errors are real.
- **Mask edges bleed.** Thin structures and transparent objects segment poorly.
- **Pose/frame sync.** The pose sent with a frame must be the pose *at capture*,
  or the anchor will swim. Snapshot it with the texture.
- **Occlusion** is out of scope for now (overlays draw on top).

## 7. Staged plan

| Stage | Server returns | Phone does | Result |
|---|---|---|---|
| **A. boxes** | box + label (+mask) | anchor a billboard/OBB at the hit-test point | end-to-end loop, blocky |
| **B. contour** | + 3D contour (camera space) | scale, world-transform, anchor, draw outline | smooth Metroid outline |
| **C. fuse** | + point patch | accumulate patches across views | better shell in the viewer |

We start at **A** (the user's call) and the server API is designed so **B** and
**C** are additive fields, not a rewrite.

### Implemented (Stage B, phone side)

`ar.ts` now runs the whole loop. On **Scan**:

1. **capture** the camera frame via WebXR `camera-access`
   (`XRWebGLBinding.getCameraImage(view)` -> `readPixels` -> JPEG).
2. hit-test at the reticle -> metric distance `D` from the camera.
3. `POST /scan` (`method`, `fovDeg` from the view projection matrix).
4. `s = D / aimed.depthAtCentre`.
5. `P_world = camPos + s·(x·right - y·up + z·forward)` using the camera's
   world basis, then create an `XRAnchor` at `obbCam.center` and attach the
   outline + OBB wireframe as children of a node (relative to the centroid).
6. relay the anchor + outline to the desktop viewer over the gateway.

> Caveat: `camera-access` does not expose camera intrinsics, so `fovDeg` comes
> from the AR view's projection matrix. If the camera image is wider than the AR
> view the outline's horizontal spread will be off - tunable via `FOV_DEG` / the
> request's `fovDeg`.
