# How ARCore does SLAM (and why it already does what you want)

> Companion to [`docs/SLAM_SURVEY.md`](SLAM_SURVEY.md) and
> [`docs/RECONSTRUCTION.md`](RECONSTRUCTION.md). This one is the *mechanism*
> write-up: what ARCore actually computes, from which sensors, and how that
> reaches us through WebXR.

## 0. The one-line answer

ARCore is a full **visual-inertial SLAM** system - Google calls it *concurrent
odometry and mapping* (COM). It runs on the phone, on every frame, and gives us
four things: a **metric 6DoF trajectory**, a **sparse feature map**, **metric
planes**, and **relocalisable anchors**. WebXR is a thin window onto it. The
only piece it does not hand to the web is a **dense surface** - and that is the
single stage we are adding ourselves.

## 1. "SLAM" and "ARCore" are the same words

| SLAM concept | What ARCore does | What we receive |
|---|---|---|
| Odometry (local pose) | per-frame pose from camera + IMU | `XRView.transform` / `getViewerPose` |
| Mapping | feature points + plane hypotheses | `XRPlane.polygon` (planes only) |
| Localisation | pose in the maintained map | the same pose |
| Loop closure | relocalisation against the stored map | implied - the pose snaps back |
| Map persistence | anchors | `XRAnchor` |
| Sensor fusion | IMU preintegration + bundle adjustment | opaque |

So the request "switch to a SLAM algorithm" is really "stop hand-rolling the
*map*; the *tracker* is already ARCore."

## 2. The sensors it fuses

| Sensor | Rate | Role |
|---|---|---|
| Camera | 30-60 Hz | visual features, plane evidence |
| Gyroscope | ~100-1000 Hz | orientation, between-frame propagation |
| Accelerometer | ~100-1000 Hz | translation **and gravity** |
| Depth sensor (ToF/lidar) | device-dependent | direct depth - **not on our phone** |

The high IMU rate is the whole trick: the camera only answers 30 times a second,
but the IMU fills in every millisecond in between, so the pose stays smooth and
metric. This is exactly what a server-side SLAM fed a 3 Hz *video* stream can
never have.

## 3. The tracking pipeline

### 3.1 Visual front end

1. Detect **feature points** - corners / high-contrast patches in the image
   (think FAST/ORB-style interest points, tracked rather than matched).
2. **Track** them frame-to-frame (optical flow / patch correlation).
3. Each point seen in ≥2 views is **triangulated** into a 3D landmark.
4. Landmarks reproject into new frames -> **reprojection residuals** that
   constrain the camera pose.

Monocular vision alone is **scale-free**: a doll's-house and a real room give
identical images. So at this point the trajectory is correct up to an unknown
scale.

### 3.2 Inertial back end

- Integrate the gyro -> orientation.
- Integrate the accelerometer, subtract gravity -> translation.
- **The accelerometer measures gravity (~9.81 m/s²), which is a known metric
  quantity.** That is what pins the metric scale of the otherwise scale-free
  visual trajectory.

This is why *ARCore can be metric with one camera*: not the camera, the IMU.
It is also why "ORB-SLAM3 monocular" cannot be - it has no IMU input on the web.

### 3.3 Fusion (the COM optimiser)

Visual reprojection residuals + IMU preintegration residuals are solved jointly
- a sliding-window bundle adjustment (or an EKF, depending on history) - for the
pose **and** the landmark positions. Run every frame, it yields a refined,
smooth, metric pose at display rate.

### 3.4 Relocalisation and map refinement

ARCore keeps a **sparse map of features + planes** for the session. When visual
tracking degrades (fast motion, low light, textureless wall) it keeps going on
the IMU, then **relocalises** by matching the live view back into the stored map.
The map is also refined continuously, which is why anchors move slightly as
tracking improves (a feature, not a bug).

## 4. The map: what is actually in it

### 4.1 Feature points - sparse

Thousands of 3D points, each with a confidence. Not a surface. (And on Android
Chrome they do **not** reach us - see §6.)

### 4.2 Planes - the exposed environment model

ARCore looks for clusters of feature points that lie on a common **horizontal or
vertical** surface, fits a plane, and grows a **polygon** in it. Over time
polygons are merged/extended. Each plane arrives with:

- `polygon` - an ordered list of 3D points (**metric, in world space**)
- `orientation` - horizontal / vertical
- (on newer ARCore) a **semantic label**: floor, wall, ceiling, table, …

This is a genuine environment model - metric, tracked, already segmented into
structure. It is the "planes act as edges" idea from your message, but computed
by three years of Google engineering instead of RANSAC we write.

### 4.3 Depth-from-motion (the ARCore Depth API)

On devices with no ToF sensor, ARCore's **Depth API** produces a per-frame depth
map by triangulating **across frames** - i.e. it is itself a multi-view SLAM
step. This is precisely the dense signal we want... but (a) it is a **native
ARCore** feature, and (b) WebXR `depth-sensing` is **not available on our
device** (confirmed in `capabilities.html`). So we get its *output* nowhere, and
re-derive depth with Depth Anything V2 instead.

> Worth a moment: our server-side "depth + fuse" stage is a hand-rolled copy of
> what ARCore does internally. We are rebuilding it because the web does not
> expose it - not because it is a better design.

## 5. Anchors: pose that sticks

An `XRAnchor` is a position + orientation, **plus a binding to the map region it
was created from**. Consequences:

- As the map refines, the runtime **corrects** the anchor pose; the anchor does
  not wander with drift.
- If tracking is lost and later relocalised, the anchor is **found again**.
- On native ARCore, anchors can be **persisted across sessions** and shared
  across devices (Cloud Anchors); on the web they live for the session.

This is why our scanned objects "glue" to the world even though we only post a
raw transform: the transform is expressed *relative to an anchor*, and ARCore
keeps the anchor honest. Our `ar.ts` already does `frame.createAnchor(...)`.

## 6. What reaches us through WebXR (and what does not)

Verified against Babylon's XR feature set in `node_modules/@babylonjs/core/XR`:

| ARCore capability | WebXR / Babylon | On our phone |
|---|---|---|
| Metric 6DoF pose | `getViewerPose` / `XRView.transform` | ✅ used |
| Reference spaces | `local`, `local-floor` | ✅ |
| Plane detection | `WebXRPlaneDetector` (`XRPlane.polygon`) | ✅ enabled |
| Anchors | `WebXRAnchorSystem` (`XRAnchor`) | ✅ used |
| Hit-test | `WebXRHitTest` (raycast vs the map) | ✅ used |
| Raw camera frames | `camera-access` | ✅ used |
| Light estimation | `WebXRLightEstimation` | ✅ enabled |
| **Feature point cloud** | `WebXRFeaturePointSystem` - **"only supported on BabylonNative, should not be used in the browser"** | ❌ |
| **Scene mesh** | `WebXRMeshDetector` - **BabylonNative only** | ❌ |
| **Depth map** | `WebXRDepthSensing` - not exposed on our device | ❌ |
| Persistent / Cloud anchors | native ARCore only | ❌ |

The three ❌ in bold are exactly the "dense map" column. Hence our pipeline.

## 7. So what does our layer actually add?

Two things ARCore's web surface does not give us:

1. **Density.** Planes are planar; objects are not. We unproject a depth model
   through the ARCore pose and accumulate a **voxel/TSDF surface** - the fused
   point cloud.
2. **Semantics.** ARCore labels *structure* (floor/wall). We label *objects*
   (chair/laptop/cup) by rendering clusters from several views and asking CLIP.

Everything else is reused as-is:

- the **pose** comes from ARCore (we never estimate it),
- the **planes** become our structure set (walls/floor, "the edges"),
- the **anchors** make the resulting objects stick,
- the **hit-test** remains our scale fallback,
- the **camera** gives us the pixels to depth-ify.

## 8. Why this beats bolting on ORB-SLAM3 / SVO Pro

- **Same information, worse:** a server-side SLAM sees a 3 Hz video stream and
  no IMU. ARCore sees 30-60 Hz **and** the IMU. You cannot beat it by
  re-running SLAM with less data.
- **Different job:** those systems map *sparse landmarks*; we need a *surface*.
  Adopting one would not remove the fusion stage - it would add a second
  trajectory to reconcile.
- **Scale:** monocular ORB-SLAM3 is scale-free; ARCore is metric, for free.
- **Engineering:** C++/Pangolin/g2o builds, or a ROS stack, versus an API we are
  already calling successfully on-device.

## 9. Honest limits

- **Drift.** Featureless walls, fast motion, low light, or a pure rotation about
  the camera centre all degrade tracking. Drift is metres over a long walk.
- **Scale is good, not perfect.** Metric within roughly percent-level over a
  room, worse over distance; the accelerometer biases show up as slow scale
  error.
- **Planes are conservative.** Cluttered/textured surfaces may never become a
  plane; planes lag behind reality by seconds.
- **No dense geometry on the web** - the whole reason for our stage.
- **Dynamic scenes confuse it.** People moving are tracked as features and
  corrupt both pose and any fused map.
- **Anchor lifetimes** are session-scoped on the web.

## 10. The native escape hatch (know it, don't take it yet)

A native Android ARCore app would expose the **Depth API** (metric, temporally
consistent, depth-from-motion), the **raw feature point cloud**, the **scene
mesh** (on supported devices), and **semantic plane labels** - a strictly richer
input than WebXR. The cost: we would lose the browser stack, the shared desktop
viewer, the `npm run dev` story, and the WebXR demo, and we would own an Android
build. For this project that trade is not worth it - but it is the honest
"if we wanted maximum quality" answer, and a good future-work paragraph.

## 11. How to milk ARCore for all it is worth

In rough priority order:

1. **Pose** - read it every frame; snapshot it *with* the captured image, never
   later (or anchors swim).
2. **Planes** - enable plane detection, stream `polygon` + `orientation` to the
   server; they are our walls/floor and our metric scale reference.
3. **Anchors** - anchor the fused objects by transform; let ARCore correct them.
4. **Hit-test** - for scale fallback and for "place" interactions.
5. **Motion** - derive keyframe gating (moved/rotated enough) from the pose; we
   get that for free.
6. **Reference space** - consider `unbounded` for long walks (avoids the local
   origin drifting out from under a far-away object).
