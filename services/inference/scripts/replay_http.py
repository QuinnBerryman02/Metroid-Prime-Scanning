"""Replay a recorded run against a *running* inference service, one method at a
time, and report what each one built.

This drives the same HTTP path the phone and the viewer use, so it exercises the
real wiring - not a reimplementation. Use it to compare methods on one recording:

    python scripts/replay_http.py c381962f-442f-4678-8c75-6c9cbe440362
    python scripts/replay_http.py <session> --methods voxel,tsdf --params '{"voxel":0.03}'

Needs RECON_CLOUD_EVERY=1 on the service to see a cloud on every frame.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "recon_log"


def post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:200]!r}"}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def decode_cloud(cloud: dict | None) -> tuple[np.ndarray, np.ndarray] | None:
    if not cloud:
        return None
    pos = np.frombuffer(base64.b64decode(cloud["positions"]), dtype="<f4").reshape(-1, 3)
    col = np.frombuffer(base64.b64decode(cloud["colors"]), dtype=np.uint8).reshape(-1, 3)
    return pos, col


def describe(pos: np.ndarray) -> str:
    if pos.shape[0] == 0:
        return "empty"
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    ext = hi - lo
    return (
        f"x[{lo[0]:+.2f},{hi[0]:+.2f}] y[{lo[1]:+.2f},{hi[1]:+.2f}] "
        f"z[{lo[2]:+.2f},{hi[2]:+.2f}]  extent {ext[0]:.2f}x{ext[1]:.2f}x{ext[2]:.2f} m"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--methods", default="voxel,points,tsdf,dbscan")
    ap.add_argument("--params", default="{}", help="JSON overriding every method")
    ap.add_argument("--stride", type=int, default=-1, help="override pixel stride")
    ap.add_argument("--scale-mode", default="", help="override scaleMode")
    ap.add_argument("--scale-a", type=float, default=None, help="with --scale-mode fixed")
    ap.add_argument("--fov", type=float, default=None, help="override fovDeg")
    ap.add_argument("--session-suffix", default="-replay")
    ap.add_argument("--from", dest="frame_from", type=int, default=1,
                    help="first frame to replay, 1-based (default 1)")
    ap.add_argument("--to", dest="frame_to", type=int, default=0,
                    help="last frame to replay, 1-based and inclusive "
                         "(0 = to the end). `--from 7 --to 7` plays frame 7 alone")
    ap.add_argument("--max", type=int, default=0,
                    help="alias for --to; kept for older invocations")
    args = ap.parse_args()

    d = LOG_DIR / args.session
    if not d.is_dir():
        print(f"no such recording: {d}")
        return 1
    frames = sorted(d.glob("*.json"))
    if not frames:
        print(f"no frames in {d}")
        return 1
    total = len(frames)
    start = max(1, args.frame_from) - 1
    end = args.frame_to or args.max or total
    if start >= total:
        print(f"--from {args.frame_from} is past the end ({total} frames)")
        return 1
    end = min(end, total)
    frames = frames[start : max(end, start + 1)]
    if len(frames) != total:
        print(f"frames {start + 1}-{start + len(frames)} of {total}")
    print(f"{len(frames)} frames from {d.name}\n")

    base = json.loads(frames[0].read_text(encoding="utf-8"))
    pics = [f.with_suffix(".jpg").read_bytes() for f in frames]

    for method in args.methods.split(","):
        method = method.strip()
        if not method:
            continue
        params = json.loads(args.params)
        if args.stride >= 0:
            params["stride"] = args.stride
        if args.scale_mode:
            params["scaleMode"] = args.scale_mode
        if args.scale_a is not None:
            params["scaleA"] = args.scale_a
        if args.fov is not None:
            params["fovDeg"] = args.fov

        session = f"{args.session}{args.session_suffix}"
        last: dict = {}
        pos = None
        for i, f in enumerate(frames):
            meta = json.loads(f.read_text(encoding="utf-8"))
            payload = dict(meta)
            payload.update(
                {
                    "sessionId": session,
                    "image": base64.b64encode(pics[i]).decode(),
                    "record": False,  # never re-record a replay
                    "method": method,
                    "params": params,
                }
            )
            payload.pop("planes", None)  # keep the recorded planes, if any
            last = post(f"{args.url}/recon/frame", payload)
            if last.get("error"):
                print(f"{method}: {last['error']}")
                break
            got = decode_cloud(last.get("cloud"))
            if got:
                pos = got[0]
        if last.get("error"):
            continue

        pts = pos.shape[0] if pos is not None else 0
        print(f"--- {method}")
        print(
            f"    fused {last.get('fused', 0)} skipped {last.get('skipped', 0)} "
            f"| map {last.get('voxels', 0)} | snapshot {pts} pts | "
            f"a={last.get('scaleA')} src={last.get('scaleMethod')} "
            f"hit={last.get('scaleHit')}"
        )
        if last.get("note"):
            print(f"    note: {last['note']}")
        t = last.get("timing") or {}
        if t:
            print(
                "    ms: "
                + " ".join(f"{k}={v:.0f}" for k, v in t.items())
            )
        print(f"    cloud: {describe(pos) if pos is not None else 'no cloud'}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
