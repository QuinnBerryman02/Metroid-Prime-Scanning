# Fusion techniques: what we use, and what to read

Companion to [`docs/RECONSTRUCTION.md`](RECONSTRUCTION.md) section 7 (the plan)
and `services/inference/scripts/bench_voxel.py` (the measurements).

There are **two** techniques in our plan, and they are different families:

1. **Voxel point accumulation** - what MS3.2 ships first. A *point-based*
   representation.
2. **TSDF fusion** - the upgrade. A *volumetric field* representation.

Understanding why (2) is strictly better is the whole reason we plan to swap.

---

## 1. Terminology cheat-sheet (search these)

| Term | Means |
|---|---|
| **SDF** | signed distance function: distance to the nearest surface, negative inside |
| **TSDF** | *truncated* SDF: clamped to a band `[-mu, +mu]` around the surface |
| **truncation** (`mu`, `sdf_trunc`) | the half-width of that band |
| **zero crossing** | the iso-surface `F(p) = 0` - where TSDF says the surface is |
| **marching cubes** | the algorithm that turns a sampled field into a triangle mesh |
| **voxel hashing** / **voxel block hashing** | sparse storage so you only pay for the surface band |
| **projective data association** | finding a voxel's measurement by *projecting* it into the depth image |
| **weighted running average** | `(F*W + f*w)/(W+w)` - how repeated observations are combined |
| **occlusion carving** / **free space** | voxels between camera and surface are empty; voxels behind are unseen |
| **surfel** | a disc-shaped surface element (position + normal + radius + colour) |
| **raycasting** | rendering the field to get a depth/normal map (used for tracking) |
| **active block** | a voxel block near the camera, allocated/updated this frame |

---

## 2. Technique 1 - voxel point accumulation (what we build first)

This is what `bench_voxel.py` implements. Per observation:

```
key      = floor(p_world / voxel)          # spatial hash
W[key]  += 1
mu[key]  = (mu[key]*(W-1) + p_world) / W   # running mean position
col[key] = (col[key]*(W-1) + rgb) / W      # running mean colour
```

Two things to know about it:

- It is **not** a TSDF - there is no field, no truncation, no zero crossing. It is
  a decimated, averaged **point cloud**. You could feed it straight to Poisson
  reconstruction if you wanted a mesh (see section 5).
- Its resolution is **coupled to its accuracy**: a single depth sample is one
  noisy point, so a voxel only averages if it is *large enough to contain
  several samples of the same surface patch*. The benchmark showed exactly this
  - at 2 cm it produced 1.6 M voxels (~10x the true surface) for a 3 cm-noise
  input, i.e. a thick fuzzy shell.

**Closest literature name:** a degenerate case of **point-based fusion**
(Keller et al. 2013), or of **surfel fusion** (ElasticFusion), minus the normals
and minus the data association. If you want the *proper* version of stage 1,
that is where to go.

---

## 3. Technique 2 - TSDF fusion (the upgrade)

### 3.1 The field

Give every voxel `p` a value. Cast a ray from the camera through `p`; let `D` be
the depth the sensor reports for that pixel, and `z` the depth of `p` itself
along the same ray. Then:

```
sdf(p) = D - z                  # signed distance along the view ray
                                #   > 0  in FRONT of the surface (free space)
                                #   < 0  BEHIND it (occluded / unseen)

tsdf(p) = clamp(sdf(p) / mu, -1, +1)
```

`mu` is the truncation distance. Note this is a **projective** signed distance
(along the camera's z axis), *not* a true Euclidean distance to the nearest
surface - that is what makes it cheap.

### 3.2 The update

```
w    = weight of this measurement (often 1; better: ~cos(theta) ray-vs-normal)
F(p) = (F(p)*W(p) + tsdf(p)*w) / (W(p) + w)      # weighted running average
W(p) = W(p) + w
```

and **only** for voxels where `|sdf(p)| <= mu` (inside the band). Voxels with
`sdf < -mu` are treated as **unseen** and are *not* updated - that is how the
method refuses to carve away things it has merely not seen yet.

### 3.3 Why this beats the point accumulator

Every measurement writes a **band of voxels**, not a single point. So each voxel
in the band gets a vote from *every* frame that sees that surface, and the
surface is recovered as the **zero crossing** of the averaged field.

Consequences:

- resolution and averaging **decouple** - you can use 2 cm voxels with 5 cm of
  depth noise and still average properly;
- the output is a **surface** (a zero level set), not a thick cloud;
- noise falls as `1/sqrt(N)` about that zero crossing, so the surface genuinely
  sharpens as you revisit it;
- occlusion is handled explicitly by the sign, for free.

### 3.4 Cost / shape of the computation

There are two equivalent ways to write it, and the choice matters for speed:

- **voxel-driven**: for each voxel, project into the image, read `D`, update.
  Simple, but you must enumerate candidate voxels.
- **image-driven ("splatting")**: for each *pixel*, walk the ray through the
  truncation band `z-mu .. z+mu` and update the voxels it crosses. This is what
  real-time systems do, because it parallelises over pixels and only touches
  voxels near the surface.

Sparse storage is what makes it practical: **voxel hashing** stores only blocks
that contain surface, so memory scales with *surface area*, not room volume.

### 3.5 Mesh extraction

Run **marching cubes** over the field (usually only over allocated blocks) to get
triangles. That is where the mesh for the desktop viewer comes from.

---

## 4. Reading list, in the order I would read them

1. **Curless & Levoy, *A Volumetric Method for Building Complex Models from Range
   Images*, SIGGRAPH 1996.** - *The* TSDF paper. Volumetric fusion, truncation,
   weighted running average, marching cubes output. Read this first; it is short
   and everything else is an engineering refinement of it.
2. **Newcombe et al., *KinectFusion: Real-Time Dense Surface Mapping and
   Tracking*, ISMAR 2011.** - The modern reference implementation. Fixed-grid
   TSDF on a GPU, projective data association, raycasting for tracking. If you
   read one paper, read this one.
3. **Niessner et al., *Real-time 3D Reconstruction at Scale using Voxel Hashing*,
   SIGGRAPH Asia 2013.** - TSDF in an unbounded scene with sparse hash storage
   and streaming. Conceptually what Open3D's `ScalableTSDFVolume` does, and what
   our packed-integer voxel keys are a crude version of.
4. **Keller et al., *Real-time 3D Reconstruction in Dynamic Scenes using
   Point-based Fusion*, 3DV 2013.** - the *point/surfel* family: running averages
   of position + normal + confidence, with model-to-frame data association and
   stability counters. The formal version of our stage 1, and worth reading
   precisely because it shows what the naive accumulator is missing.
5. **Keller et al., *ElasticFusion*, ISMAR 2017.** - the same idea as a mature,
   practical system (surfels + deformation graph for loop closure).
6. **Dai et al., *BundleFusion*, SIGGRAPH 2017.** - TSDF fusion plus global
   consistency over a whole building.
7. **Kahler et al., *InfiniTAM v3*, 2016 (and InfiniTAM 2015).** - a clean,
   readable open-source implementation of the pipeline above; good for seeing how
   the pieces fit in practice.
8. **Lorensen & Cline, *Marching Cubes*, SIGGRAPH 1987.** - mesh extraction.

Background that removes most of the fog:

- **SDF / level sets**: any intro to implicit surfaces. You only need the idea
  that a surface is the zero set of a scalar field.
- **Curless & Levoy also define the weighting** by the angle between the ray and
  the surface, which is where the `cos(theta)` weight comes from.

---

## 5. The alternative we are *not* taking (worth knowing)

Because stage 1 already gives us a **point cloud with normals**, we could skip
TSDF and mesh it directly with:

- **Poisson surface reconstruction** - Kazhdan, Bolitho & Hoppe, *Poisson Surface
  Reconstruction*, SGP 2006 (and the *Screened* variant, 2013). Open3D exposes it
  as `create_from_point_cloud_poisson`.

It produces beautifully smooth watertight meshes and needs no camera poses. We
still prefer TSDF because **Poisson throws away the sensor model**: it cannot
distinguish "nothing was observed here" from "free space", so depth noise and
unobserved regions both become invented geometry. TSDF's occlusion sign encodes
that. For a *scanned* rather than *modelled* room, that matters.

---

## 6. How this maps onto our code

| Step | Technique | Where |
|---|---|---|
| unproject + quantise + running mean | voxel point accumulation | `bench_voxel.py` (to become `app/fusion.py`) |
| metric scale | affine fit `Z = a/d + b` to ARCore planes | `RECONSTRUCTION.md` section 6 |
| volumetric fusion | **TSDF**, hashed | `open3d` `ScalableTSDFVolume` |
| mesh | **marching cubes** (`extract_triangle_mesh`) | `open3d` |
| fallback mesh from points | Poisson (not used) | `open3d` |

### One practical gotcha, flagged now

Open3D's `ScalableTSDFVolume` constructor is
`ScalableTSDFVolume(voxel_length, sdf_trunc, color_type)`, and `integrate()`
takes `(rgbd, intrinsic, extrinsic)`. **Check the direction of `extrinsic` in the
docs** (world->camera vs camera->world) before trusting it - it is the single
most common bug in every TSDF codebase, including ours if we are careless. Our
phone already sends `worldFromCamera`; the integration call needs whichever
inverse it asks for.

Also: `sdf_trunc` should be around **2-3x `voxel_length`** (e.g. 2 cm voxels,
4-6 cm truncation). Too small and surfaces never accumulate; too large and you
smooth through thin geometry.
