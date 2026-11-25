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
   objects are highlighted and selectable.

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

## Quick start (once scaffolded)

```bash
npm install
npm run dev:inference   # python model service on :8000
npm run dev:server      # gateway on :3000
npm run dev:client      # web app on :5173
```

See `docs/PLAN.md` for the staged build order.

## Hardware assumptions

Developed against an **Intel integrated GPU** (no CUDA). Inference uses **OpenVINO**
(Intel GPU / CPU) and quantised ONNX models. See `docs/MODELS.md`.

## Target devices

- **Mobile AR:** Android Chrome (best WebXR: hit-test, anchors, plane + depth sensing,
  raw camera access). iOS Safari WebXR/AR support is limited - see `docs/ARCHITECTURE.md`.
- **Desktop viewer:** any modern browser.
