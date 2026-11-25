# Metroid Prime Environment Scanning

A WebXR prototype that turns a phone into a Metroid Prime style **scan visor**.

Point the phone at your environment, scan objects, and the system:

1. **Segments** the object from the camera feed (YOLO-seg / FastSAM / MobileSAM).
2. **Classifies / explains** it with a vision model (CLIP, Florence-2, Moondream).
3. **Lifts** the 2D detection into **3D anchored overlays** using monocular depth + the
   device 6DoF pose (WebXR), fusing everything into a live world map.
4. Renders **colored outlines anchored in space** on the phone (mobile AR view), plus a
   centre reticule and Metroid-style scan info cards.
5. Renders the whole reconstructed environment on a **desktop viewer** where scanned
   objects are highlighted and selectable (coarse box/plane reconstruction is enough).

> Scope is deliberately small: the goal is *interesting, demonstrable behaviour*, not a
> production SLAM/NeRF system.

## Pages

| Page | URL | Purpose |
|---|---|---|
| Landing | `/` | support check + links |
| **AR scan visor** | `/ar.html` | phone: hit-test reticle, anchored placement, live pose (MS1) |
| **Desktop viewer** | `/viewer.html` | shows the phone's live pose in a shared world (MS1) |
| Detection test | `/scan.html` | Stage A: camera still -> server -> YOLO boxes |
| Capability test | `/capabilities.html` | what the device exposes (depth, planes, anchors, ...) |

## Status

- **MS0** repo & tooling - done
- **MS1** WebXR AR skeleton + pose/anchor relay - done (hit-test reticle, WebXR anchors,
  device pose + placed anchors -> gateway -> viewer)
- **MS2** model exploration - done. `services/inference/notebooks/` verifies YOLO11n,
  FastSAM, MobileSAM, SAM 2.1, Depth Anything V2 and CLIP; boxed detection is the
  default, smooth masks are the planned "scan action".
- **MS2** OpenVINO check - done. `services/inference/samples/openvino_check.py`; iGPU is
  2.3x faster than CPU for YOLO11n (NPU available too).
- **MS2** detection service (Stage A) - done. `POST /scan` runs YOLO11n via OpenVINO
  on the iGPU (~20 ms), returns boxes + labels and the aimed object; the gateway and
  `/scan.html` are wired end-to-end.
- **MS2** segmentation + lifting (Stage B) - next (schema already allows it)

Verify the pose relay without a phone:

```bash
npm run build --workspace @metroid-scan/server
npm run start --workspace @metroid-scan/server &   # gateway on :3000
npm run smoke:ws                                    # pose + anchor relay
```

## Docs

- [`docs/PLAN.md`](docs/PLAN.md) - implementation plan, milestones, task breakdown
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - system + data-flow design
- [`docs/LIFTING.md`](docs/LIFTING.md) - how a 2D mask becomes an anchored 3D object
- [`docs/MODELS.md`](docs/MODELS.md) - model choices (iGPU-friendly) and why

## Layout

```
apps/
  client/        Vite + TypeScript + Babylon.js  (mobile AR view + desktop viewer)
  server/        Fastify + TypeScript            (HTTP/WS gateway, session + world state)
services/
  inference/     Python FastAPI + OpenVINO       (YOLO / SAM / depth / CLIP)
packages/
  shared/        shared TypeScript protocol types
docs/            plans and design notes
```

## Quick start

```bash
npm install
npm run dev
```

`npm run dev` starts the whole stack with colour-coded, prefixed output per service
and stops everything on `Ctrl+C`:

| service | url | prefixed as |
|---|---|---|
| inference (FastAPI + OpenVINO) | http://127.0.0.1:8000 | `[inference]` |
| gateway (Fastify + WS) | http://localhost:3000 | `[gateway]` |
| client (Vite, HTTPS) | https://localhost:5173 | `[client]` |

It auto-detects the Python venv at `services/inference/.venv` (Windows or POSIX
layout). If that venv doesn't exist yet, it warns and runs just the gateway + client:

```bash
cd services/inference
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # .venv/bin/pip on macOS/Linux
```

The first inference startup takes ~20-30 s (OpenVINO compiles the model; it warms
up at load so the first `/scan` is fast).

Run services individually when you want to:

```bash
npm run dev:inference   # python model service on :8000
npm run dev:server      # gateway on :3000
npm run dev:client      # web app on :5173 (HTTPS, see below)
```

See `docs/PLAN.md` for the staged build order.

## Running on the phone (HTTPS made easy)

WebXR and the camera only work in a **secure context**. `http://localhost` counts, a LAN
IP does not - so the dev server serves HTTPS using a **self-signed certificate** that Vite
generates automatically (`@vitejs/plugin-basic-ssl`). No cloudflared, no mkcert, no cert
files to manage.

```bash
npm run dev:client
```

Vite prints something like:

```
  ➜  Local:   https://localhost:5173/
  ➜  Network: https://192.168.1.13:5173/
```

On the phone (same Wi-Fi) open the **Network** URL. The first time you'll get a
"Not private / certificate not trusted" warning - tap **Advanced -> Proceed**. That's
expected for a self-signed cert; the connection is still encrypted and now a secure
context, so WebXR works.

### Alternative: no warning at all (Android + USB)

If you'd rather not click through the cert warning, forward the port over USB so the phone
sees it as `localhost` (which is trusted):

```bash
adb reverse tcp:5173 tcp:5173
# then on the phone browser: http://localhost:5173/capabilities.html
```

`adb` ships with Android Studio / the Android Platform Tools. This also avoids any
firewall issues.

## Manual test pages

The client ships small standalone diagnostics so you can sanity-check real hardware
without touching the main app. Open them on the phone using either method above:

| Page | URL | Checks |
|---|---|---|
| Capability test | `/capabilities.html` | secure context, WebXR support, which AR features are enabled (hit-test, anchors, plane + **depth sensing**, camera-access, dom-overlay, ...), plus a live depth preview and hit-test reticle |

These pages deliberately avoid Babylon so they still work if app code regresses.

### Sending a report back (no copy/paste)

The capability page has a **Send report** button. It POSTs the result to a small dev-only
endpoint (`/__diag/report`, see `apps/client/vite.config.ts`) which writes it to disk:

```
apps/client/reports/capabilities-<timestamp>.json
apps/client/reports/latest-capabilities.json   <- most recent, easy to grab
```

A step-by-step **in-AR panel** shows the feature list, live hit-test/depth state, and the
Send/End buttons, so everything is usable while inside the AR session. If the send fails
(e.g. the sink is unavailable) the page falls back to copying to the clipboard.

## Current priorities (per project brief)

1. **Mobile AR scan HUD** - the core deliverable.
2. Classification / segmentation pipeline + anchored outlines.
3. Desktop 3D viewer - *coarse is fine*: room walls + scanned objects as boxes.
4. Occlusion of AR overlays is **out of scope** for now.

## Hardware assumptions

Developed against an **Intel integrated GPU** (no CUDA). Inference uses **OpenVINO**
(Intel GPU / CPU) and quantised ONNX models. See `docs/MODELS.md`.

## Target devices

- **Mobile AR:** Android Chrome (best WebXR: hit-test, anchors, plane + depth sensing,
  raw camera access). iOS Safari WebXR/AR support is limited - see `docs/ARCHITECTURE.md`.
  Use `/capabilities.html` to confirm what your specific handset exposes.
- **Desktop viewer:** any modern browser.
