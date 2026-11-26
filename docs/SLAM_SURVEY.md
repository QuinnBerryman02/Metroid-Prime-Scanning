# SLAM / odometry survey (for the dense-environment upgrade)

> **Question:** should we drop the single-view unprojection and adopt a SLAM
> system (ORB-SLAM3, SVO Pro, …) to build the 3D environment as we walk around?
>
> **Answer:** we already have a SLAM estimator running - it is **ARCore, on the
> phone, exposed through WebXR**. What we are missing is not SLAM, it is a
> **dense map**. This doc surveys the options and justifies the recommendation.

## 1. Split the problem in two

"SLAM" is two different things, and they have very different costs:

| Problem | What it produces | Who solves it for us |
|---|---|---|
| **Pose estimation (odometry)** | 6DoF trajectory, metric, real-time | **ARCore** (device VIO) via WebXR `getViewerPose` |
| **Dense mapping** | a surface / point cloud of the scene | *nobody yet* - **this is the gap** |

Classic SLAM systems (ORB-SLAM3, SVO, stella_vslam) bundle the two, but their
maps are **sparse** - they map a few thousand ORB feature *landmarks*, not a
surface. You cannot run DBSCAN / segmentation on 5 000 scattered points. So even
if we adopted one, we would **still** have to write the dense-fusion layer. The
SLAM detour buys us a second trajectory, not the thing we actually want.

## 2. Requirements

1. **Monocular** - a phone gives us one RGB stream (no stereo, no lidar).
2. **No CUDA** - target is an Intel Core Ultra 7 165U iGPU; OpenVINO only.
3. **Windows dev**, Git Bash. (Anything ROS-shaped is a non-starter.)
4. **Coexist with WebXR** - the phone must keep rendering the AR session while
   estimation runs; we cannot hand the camera to a second consumer on-device.
5. **Python**, or at least callable from the FastAPI sidecar.
6. **Real-time-ish** (~2-5 Hz is plenty for a reconstruction session).
7. **Dense output** we can cluster/segment.
8. **Metric scale** - otherwise a fused map has no consistent size and anchors
   float.

## 3. Candidates

### 3.1 ARCore device VIO (via WebXR)  - *the incumbent*

- **Type:** visual-inertial odometry + relocalisation + plane fusion.
- **Inputs:** high-rate camera **+ IMU** (gyro/accel, ~200 Hz) - WebXR hides it,
  but it is what makes ARCore's pose far more stable than any video-only SLAM.
- **Output:** metric 6DoF pose at display rate; metric **plane** polygons
  (floor/wall/table); `XRAnchor`s that survive relocalisation.
- **Pros:** already working in our app; metric; zero build cost; runs on-device
  at 30-60 Hz; loop closure/relocalisation included; anchors are a first-class
  WebXR API we already use.
- **Cons:** does **not** expose a dense map on our device (`depth-sensing` is
  unavailable - see `capabilities.html`); pose only exists on-device.
- **Verdict:** **use it as the pose source.** It is strictly better than any
  server-side monocular SLAM we could bolt on, because it has the IMU and the
  full-rate frame stream.

### 3.2 ORB-SLAM3  (UZ-SLAMLab/ORB_SLAM3) - *requested*

- **Type:** indirect feature-based SLAM (ORB). Mono / stereo / RGB-D, optionally
  inertial. Modern classic; excellent accuracy benchmarks.
- **Deps (C++11):** OpenCV, Eigen, **Pangolin** (GUI/GL), DBoW2, **g2o**,
  Sophus. Build is notorious.
- **Output:** trajectory **+ sparse landmark cloud** (~10³-10⁴ points) +
  keyframes. **No dense surface.**
- **Windows:** effectively painful (Pangolin/GLFW, g2o). Everyone runs it on
  Linux/WSL.
- **Python:** no official bindings; community wrappers exist but rot quickly.
- **Monocular scale:** unobservable. Its metric mode needs **IMU** or a known
  baseline. WebXR does not expose raw IMU, so we would get a scale-free
  trajectory - worse than ARCore, which is already metric.
- **Real-time:** yes on a desktop CPU, single-threaded-ish, best with stereo.
- **Verdict:** **poor fit.** Duplicates ARCore, loses the IMU, is scale-free in
  the mode we could use, produces a sparse map we cannot segment, and is a
  Windows build nightmare.

### 3.3 rpg_svo_pro_open  (UZH-RPG) - *requested*

- **Type:** semi-direct **visual odometry** (SVO) + a SLAM back-end.
- **Deps:** **ROS / catkin** (Vikit, Ceres, Sophus, rviz plugins), optional IMU.
- **Output:** trajectory + sparse map. Embedded/drone oriented.
- **Verdict:** **worst fit for our environment.** ROS on Windows is not
  practical; porting it off ROS is more work than the whole reconstruction
  layer. Not recommended.

### 3.4 stella_vslam  (successor to OpenVSLAM)

- **Type:** indirect SLAM (ORB + custom), mono/stereo/RGB-D.
- **Deps:** C++ with CMake; **optional pybind11 Python bindings**; YAML config,
  many camera models.
- **Output:** trajectory + sparse map + keyframes.
- **Windows:** buildable, but still Eigen/OpenCV/SuiteSparse-grade deps.
- **Verdict:** **the most practical "real SLAM" if we ever want one**, but still
  sparse and still redundant with ARCore. Keep as a fallback, not a plan.

### 3.5 PySLAM  (luigifreda/pyslam)

- **Type:** pure-Python SLAM framework; pluggable ORB / SuperPoint / etc.;
  mono/stereo/RGB-D; optional dense volume + semantic fusion.
- **Pros:** Python, pip-friendly, readable, good for experiments.
- **Cons:** not built for real-time on an iGPU; research-grade code.
- **Verdict:** great for an **offline comparison** in the report; not for the
  live loop.

### 3.6 Open3D pipelines (odometry + TSDF)

- **Type:** Python library, **not** a full SLAM, but ships exactly the mapping
  half: `compute_rgbd_odometry`, `ScalableTSDFVolume`, pose-graph optimisation,
  `cluster_dbscan`, RANSAC planes, normals, offscreen rendering.
- **Input:** RGB-D (we synthesise the D with our depth model - fine).
- **Verdict:** **adopt it for the dense-mapping half.** Its odometry is a useful
  sanity check, but ARCore stays the pose authority.

### 3.7 Deep SLAM - DPVO / DROID-SLAM / GO-SLAM / MASt3R-SLAM

- State of the art accuracy, dense by construction.
- **All require CUDA.** Our iGPU cannot run them. Excluded.
- (Optional: run one on a rented GPU **offline** on a recorded clip, purely as a
  report comparison / upper bound.)

### 3.8 ROS-centric others (RTAB-Map, OpenVINS, Kimera, Basalt, …)

Stereo/IMU/ROS-centric, or need depth sensors. Same conclusions as 3.3.

## 4. Comparison

| System | Language / deps | Mono | Output | Dense? | Metric? | Runs on our iGPU/CPU | Windows | Python | Fit |
|---|---|---|---|---|---|---|---|---|---|
| **ARCore via WebXR** | already integrated | - | metric pose + planes + anchors | ✗ (add fusion) | ✅ (IMU) | ✅ on-device | n/a | n/a | ⭐ **best** |
| ORB-SLAM3 | C++ / Pangolin, g2o, DBoW2, Sophus | ✅ | trajectory + sparse | ✗ | ⚠️ IMU only | ⚠️ CPU, heavy | ❌ | ❌ | ✗ |
| rpg_svo_pro_open | C++ / **ROS** | ✅ | trajectory + sparse | ✗ | ⚠️ | ⚠️ | ❌ | ❌ | ✗ |
| stella_vslam | C++ (+pybind) | ✅ | trajectory + sparse | ✗ | ⚠️ | ✅ | ⚠️ | ✅ | ⚠️ fallback |
| PySLAM | Python | ✅ | trajectory + sparse (+dense) | ~ | ⚠️ | ⚠️ slow | ✅ | ✅ | ⚠️ offline only |
| Open3D pipelines | Python | ✗ (RGB-D) | **dense TSDF / surfaces** | ✅ | from D | ✅ | ✅ | ✅ | ⭐ **use for mapping** |
| DPVO / DROID-SLAM / MASt3R | PyTorch | ✅ | dense | ✅ | ✅ | ❌ CUDA | ❌ | ✅ | ✗ |

## 5. Recommendation

1. **Pose:** keep **ARCore via WebXR**. It is a real VIO+SLAM with an IMU, it is
   metric, and it is already in the app. Do **not** rebuild it server-side from a
   3 Hz video stream - that is strictly less information.
2. **Dense map:** add a **server-side fusion** stage: every keyframe (RGB + pose
   + intrinsics) is unprojected through a monocular depth model and integrated
   into a world-space voxel/TSDF volume (Open3D for the heavy lifting).
3. **Scale:** recover metric scale per frame by fitting `Z = a·(1/d_rel) + b`
   against **ARCore's metric planes** (least squares); hit-test as fallback.
4. **Report material:** optionally run ORB-SLAM3 / stella_vslam **offline** on a
   recorded clip and compare its trajectory to ARCore's, and/or DROID-SLAM on a
   GPU as a quality upper bound. This shows field awareness without betting the
   build on a fragile C++/ROS stack.

> The one thing classic SLAM would give us that ARCore does not is a
> **pose estimate we control** (e.g. to fuse in a frame that isn't tied to the
> device). We do not need it: a single device means ARCore's `local` space *is*
> our canonical world frame (see `docs/RECONSTRUCTION.md` §3).
