"""Is `Z = a * rel` even the right model?

For every recorded frame we have two things: Depth Anything V2's inverse
disparity `rel = 1/d` at the reticle, and ARCore's metric hit-test distance to
the surface under the reticle. The ratio `hit / rel` is the scale that frame
implies.

  * constant ratio  -> the form is right, and a is that constant
  * drifting ratio  -> the form is wrong (or the reticle is not the image
                       centre), and no single a can fix it

Prints the ratio per frame plus a regression against `rel`, because a drift
that is linear in `rel` points at the affine form Z = rel / (m + c*rel) rather
than a pure scale.

    python scripts/analyse_scale.py <session>
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.depth import DepthEstimator  # noqa: E402

LOG_DIR = ROOT / "recon_log"


def main() -> int:
    session = sys.argv[1] if len(sys.argv) > 1 else ""
    d = LOG_DIR / session
    if not d.is_dir():
        print(f"no such recording: {d}")
        return 1
    frames = sorted(d.glob("*.json"))
    model = DepthEstimator()

    rows: list[tuple[str, float, float, float]] = []
    for f in frames:
        meta = json.loads(f.read_text(encoding="utf-8"))
        hit = meta.get("hitDistance")
        if not hit or hit <= 0:
            continue
        pil = Image.open(io.BytesIO(f.with_suffix(".jpg").read_bytes())).convert("RGB")
        w, h = pil.size
        rel = model.predict_depth(pil)
        c = float(rel[h // 2, w // 2])
        # Also sample a small patch: a single pixel is noisy.
        patch = float(np.median(rel[h // 2 - 4 : h // 2 + 5, w // 2 - 4 : w // 2 + 5]))
        rows.append((f.stem, hit, c, patch))

    if not rows:
        print("no frames carried a hitDistance - tap Key with the reticle on a surface")
        return 1

    print(f"{len(rows)} frames with a hit-test reading\n")
    print(f"{'frame':<22} {'hit m':>7} {'rel ctr':>9} {'rel patch':>10} {'a=hit/rel':>10}")
    ratios = []
    for name, hit, c, patch in rows:
        a = hit / patch if patch > 1e-6 else float("nan")
        ratios.append(a)
        print(f"{name:<22} {hit:7.3f} {c:9.4f} {patch:10.4f} {a:10.3f}")

    r = np.array(ratios, dtype=float)
    r = r[np.isfinite(r)]
    print(
        f"\nratio  median {np.median(r):.3f}  mean {r.mean():.3f} "
        f"sd {r.std():.3f}  min {r.min():.3f}  max {r.max():.3f}"
    )
    spread = (r.max() - r.min()) / max(np.median(r), 1e-9)
    print(f"spread {(r.max() - r.min()):.3f} = {spread * 100:.0f}% of the median")

    # Drift with rel: fit hit = m * rel + b (affine in rel) and hit = a * rel.
    rel_patch = np.array([p for _, _, _, p in rows], dtype=float)
    hit = np.array([h for _, h, _, _ in rows], dtype=float)
    pure = float(np.sum(rel_patch * hit) / max(np.sum(rel_patch**2), 1e-12))
    A = np.stack([rel_patch, np.ones_like(rel_patch)], axis=1)
    (m, b), *_ = np.linalg.lstsq(A, hit, rcond=None)
    res_pure = hit - pure * rel_patch
    res_aff = hit - (m * rel_patch + b)
    print(f"\npure    hit = {pure:.4f} * rel        rms {np.sqrt((res_pure**2).mean()):.4f} m")
    print(f"affine  hit = {m:.4f} * rel + {b:+.4f}  rms {np.sqrt((res_aff**2).mean()):.4f} m")

    if b != 0 and abs(b) > 0.15 * abs(m * np.median(rel_patch)):
        inv_m, inv_c = 1.0 / m, b / m
        print(
            f"offset is significant -> the form Z = rel / ({inv_m:.3f} - {inv_c:.3f}*rel)"
            " fits better than a pure scale"
        )
    print(
        f"\nZ = a*rel with a = {pure:.3f} is the best pure-scale fit to these "
        f"{len(rows)} frames"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
