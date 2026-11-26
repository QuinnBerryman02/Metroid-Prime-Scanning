"""Replay a recorded keyframe run offline and compare fusion settings.

    python scripts/replay_recon.py <recon_log/session> [options]

The service records every keyframe (RECON_RECORD=1) as NNNN.jpg + NNNN.json.
This script re-fuses the same frames under different scales / voxel sizes and
compares them, so a single walk-around becomes a dataset we can tune against
instead of a slow on-device loop.

**Important caveat on the objective.** It is tempting to score a candidate scale
by how compact the fused map is (`meanW`). That metric is DEGENERATE: shrinking
the scale always packs points into fewer voxels, so it monotonically prefers
small `a` - see the sweep table. It cannot be trusted to pick a global scale.

The reason is fundamental rather than a bug: with a purely geometric objective
the global scale of a monocular reconstruction is *unobservable* (the same reason
monocular SLAM has scale ambiguity). Only parallax would break that, and a
hand-held walk-around barely has any at room scale. **A global metric scale must
come from a metric reference** - ARCore's planes or hit-test - not from
self-consistency.

So what this tool IS good for:
  * comparing *consistency* of per-frame scale sources (planes vs hit-test),
  * comparing voxel sizes / strides / methods on the same real data,
  * dumping coloured PLYs (`--ply`) of each variant so you can LOOK at them,
which is the judgement that actually matters.

Options:
    --strategy planes|hit|hitmed|sweep   (default: hitmed)
    --voxel 0.05                        fusion voxel size (metres)
    --stride 2                          unproject every Nth pixel
    --sweep-lo 0.2 --sweep-hi 3.0 --sweep-n 15
    --stride-frames 1                   use every Nth keyframe
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from app import config  # noqa: E402
from app.depth import DepthEstimator  # noqa: E402
from app.fusion import (  # noqa: E402
    ScaleFit,
    VoxelMap,
    fit_metric_scale,
    pose_to_matrix,
    unproject_depth,
)
from app.lifting import intrinsics_from_fov  # noqa: E402


def load_frames(folder: Path):
    frames = []
    for meta_path in sorted(folder.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        frames.append((meta_path.with_suffix(".jpg"), meta_path.with_suffix(".npy"), meta))
    return frames


def rel_map(est, jpg: Path, npy: Path) -> np.ndarray:
    if npy.exists():
        return np.load(npy)
    from PIL import Image

    pil = Image.open(jpg).convert("RGB")
    rel = cv2.medianBlur(est.predict_depth(pil), 5)
    np.save(npy, rel)
    return rel


def prepare(folder: Path, stride_frames: int, verbose: bool, want_rgb: bool = False):
    frames = load_frames(folder)
    if not frames:
        sys.exit(f"no frames in {folder}")
    frames = frames[::stride_frames]
    est = None
    out = []
    for jpg, npy, meta in frames:
        if not npy.exists():
            if est is None:
                est = DepthEstimator(
                    config.DEPTH_MODEL, config.DEPTH_DEVICE, config.DEPTH_IR,
                    config.EXPORT_OPENVINO,
                )
            rel = rel_map(est, jpg, npy)
        else:
            rel = rel_map(None, jpg, npy)  # type: ignore[arg-type]
        h, w = rel.shape
        intr = intrinsics_from_fov(w, h, meta.get("fovDeg") or config.FOV_DEG)
        wfm = pose_to_matrix(meta["pose"])
        diag: dict = {}
        fit = fit_metric_scale(meta.get("planes") or [], wfm, intr, rel, diag)
        if fit is not None and fit.rms > config.RECON_SCALE_RMS_MAX:
            fit = None
        hit = None
        hd = meta.get("hitDistance")
        if hd:
            rc = float(rel[h // 2, w // 2])
            if rc > 1e-6:
                hit = hd / rc
        if verbose:
            print(
                f"  {jpg.name}: planes a={fit.a:.3f} rms={fit.rms:.3f} ({fit.method})"
                if fit
                else f"  {jpg.name}: planes REJECTED rms={diag.get('spread', 0):.2f} spread "
                f"{diag.get('samples', 0)} samples",
                f"| hit a={hit:.3f}" if hit else "| hit none",
            )
        rgb = None
        if want_rgb:
            from PIL import Image

            rgb = np.array(Image.open(jpg).convert("RGB"))
        out.append(
            {"rel": rel, "intr": intr, "wfm": wfm, "fit": fit, "hit": hit, "rgb": rgb}
        )
    return out


def fuse(frames, scales, voxel: float, stride: int) -> dict | None:
    vm = VoxelMap(voxel=voxel)
    used = 0
    for f, sc in zip(frames, scales):
        if sc is None:
            continue
        pts, ys, xs = unproject_depth(
            f["rel"], f["wfm"], f["intr"], sc, stride=stride,
            min_depth=config.RECON_MIN_DEPTH, max_depth=config.RECON_MAX_DEPTH,
        )
        if not pts.shape[0]:
            continue
        used += 1
        rgb = f.get("rgb")
        colours = (
            rgb[ys, xs].astype(np.float64)
            if rgb is not None
            else np.full((pts.shape[0], 3), 128.0)
        )
        vm.integrate(pts, colours)
    if vm.size == 0:
        return None
    w = vm.wsum
    return {
        "frames": used,
        "voxels": vm.size,
        "meanW": float(w.mean()),
        "fracW1": float((w <= 1).mean()),
        "vm": vm,
    }


def write_ply(path: Path, vm: VoxelMap) -> None:
    """Coloured ASCII PLY of the fused map, for eyeballing in any viewer."""
    mu = vm.spos / vm.wsum[:, None]
    col = np.clip(vm.scol / vm.wsum[:, None], 0, 255).astype(np.uint8)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {mu.shape[0]}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        for p, c in zip(mu, col):
            fh.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
    print(f"wrote {path} ({mu.shape[0]:,} pts)")


def carry(values):
    """Per-frame scale, holding the previous value when a frame has none."""
    out, last = [], None
    for v in values:
        if v is not None:
            last = v
        out.append(last)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--strategy", default="hitmed",
                    choices=["planes", "hit", "hitmed", "sweep"])
    ap.add_argument("--voxel", type=float, default=config.RECON_VOXEL)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--stride-frames", type=int, default=1)
    ap.add_argument("--sweep-lo", type=float, default=0.2)
    ap.add_argument("--sweep-hi", type=float, default=3.0)
    ap.add_argument("--sweep-n", type=int, default=15)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--ply", type=Path, default=None,
                    help="write the fused cloud as a coloured PLY")
    args = ap.parse_args()

    print(f"run      {args.folder}")
    frames = prepare(args.folder, args.stride_frames, verbose=not args.quiet,
                     want_rgb=bool(args.ply))
    print(f"frames   {len(frames)}")
    hits = [f["hit"] for f in frames if f["hit"]]
    fits = [f["fit"] for f in frames if f["fit"]]
    if hits:
        print(f"hit a    median {np.median(hits):.3f}  ({len(hits)}/{len(frames)} frames)")
    if fits:
        print(f"plane a  median {np.median([f.a for f in fits]):.3f}  ({len(fits)}/{len(frames)})")

    print(f"\nvoxel {args.voxel} m, stride {args.stride}")
    header = f"{'strategy':<22} {'frames':>6} {'voxels':>9} {'meanW':>7} {'fracW1':>7}"
    chosen: dict | None = None

    if args.strategy == "sweep":
        print(header)
        for a in np.linspace(args.sweep_lo, args.sweep_hi, args.sweep_n):
            r = fuse(frames, [ScaleFit(float(a), 0.0, 0, 0.0, "sweep")] * len(frames),
                     args.voxel, args.stride)
            if not r:
                continue
            print(f"a={a:<20.3f} {r['frames']:>6} {r['voxels']:>9} "
                  f"{r['meanW']:>7.2f} {r['fracW1']:>7.3f}")
            if chosen is None or r["meanW"] > chosen["meanW"]:
                chosen = r
        print("\n(meanW is degenerate - see the caveat at the top of this file.)")

    if args.strategy in ("hitmed", "sweep") and hits:
        med = float(np.median(hits))
        r = fuse(frames, [ScaleFit(med, 0.0, 0, 0.0, "hitmed")] * len(frames),
                 args.voxel, args.stride)
        if r:
            print(f"{'hit median':<22} {r['frames']:>6} {r['voxels']:>9} "
                  f"{r['meanW']:>7.2f} {r['fracW1']:>7.3f}")
            if args.strategy != "sweep":
                chosen = r
    if args.strategy == "hit" and hits:
        r = fuse(frames, carry([f["hit"] for f in frames]), args.voxel, args.stride)
        if r:
            print(f"{'hit per-frame':<22} {r['frames']:>6} {r['voxels']:>9} "
                  f"{r['meanW']:>7.2f} {r['fracW1']:>7.3f}")
            chosen = r
    if args.strategy == "planes":
        r = fuse(frames, carry([f["fit"] for f in frames]), args.voxel, args.stride)
        if r:
            print(f"{'planes':<22} {r['frames']:>6} {r['voxels']:>9} "
                  f"{r['meanW']:>7.2f} {r['fracW1']:>7.3f}")
            chosen = r

    if args.ply and chosen:
        write_ply(args.ply, chosen["vm"])


if __name__ == "__main__":
    main()
