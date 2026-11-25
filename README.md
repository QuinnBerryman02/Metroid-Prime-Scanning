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

## Docs

- [`docs/PLAN.md`](docs/PLAN.md) - implementation plan, milestones, task breakdown
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - system + data-flow design
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
npm run dev:inference   # python model service on :8000
npm run dev:server      # gateway on :3000
npm run dev:client      # web app on :5173
```

See `docs/PLAN.md` for the staged build order.

## Manual test pages

The client ships small standalone diagnostics so you can sanity-check real hardware
without touching the main app. Open them on the phone over **HTTPS**:

| Page | URL | Checks |
|---|---|---|
| Capability test | `/capabilities.html` | secure context, WebXR support, which AR features are enabled (hit-test, anchors, plane + **depth sensing**, camera-access, dom-overlay, ...), plus a live depth preview and hit-test reticle |

Run the client bound to your LAN and open `https://<your-lan-ip>:5173/capabilities.html`.
These pages deliberately avoid Babylon so they still work if app code regresses.

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
