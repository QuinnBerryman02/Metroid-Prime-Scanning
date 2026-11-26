# Running the stack

```bash
npm run dev
```

One command, three processes, all logged to the same terminal with a colour and
a name per line:

| name | what | where |
| --- | --- | --- |
| `inference` | FastAPI + OpenVINO models | http://127.0.0.1:8000 |
| `gateway` | Fastify: WebSocket hub + API proxies | http://127.0.0.1:3000 |
| `client` | Vite dev server (HTTPS, self-signed) | https://localhost:5173 |

The client never talks to the inference directly. It uses **relative URLs** -
`fetch("/api/...")` and `new WebSocket("/ws")` - and Vite proxies those to the
gateway (`/api` → `:3000`, `/ws` → `ws://:3000`). So there is exactly one origin
in play and no CORS or mixed-content problem. If you add a call, use a relative
URL; an absolute `http://localhost:3000/...` will break on the phone.

Pages: `/` (landing), `/ar.html`, `/viewer.html`, `/scan.html`,
`/capabilities.html`.

## When a request fails

The failure mode tells you which piece is broken. These are worth reading
literally, because they are easy to misdiagnose.

| what you see | what it means |
| --- | --- |
| `TypeError: Failed to fetch` | **Network, not application.** The browser never got a response, so there is no status code to interpret. Either the dev server is not running, or the gateway behind its proxy is not. A 500 from the gateway would say `500`, so this is never a logic bug. |
| `gateway unreachable on :3000` | The viewer's own health probe, which separates the two cases above. The gateway process is gone or was never up. |
| `inference unreachable` | The gateway is fine and could not reach `:8000`. Models take ~25 s to load, so right after `npm run dev` this is expected. |
| `no recorded runs` | The gateway is fine; `RECON_LOG_DIR` is empty. Record one with the phone. |
| `log dir unreadable` | `RECON_LOG_DIR` is set but not a directory. `dev.mjs` sets it; a hand-started gateway may not. |

## Stale processes holding a port - the recurring one

**A killed parent does not kill its children.** The inference runs `uvicorn
--reload`, which spawns a reloader plus OpenVINO `multiprocessing.spawn`
workers. Killing the visible process leaves the children holding `:8000`, and
the next `npm run dev` fails to bind with `EADDRINUSE` while the *old* code
keeps serving. This has caused more confusion on this project than anything
else: the symptom is that a change appears to do nothing, because the process
answering the request is not the one you edited.

The same applies to the gateway on `:3000` and Vite on `:5173`.

Check what is actually listening, and who owns it:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen |
  Select-Object OwningProcess -Unique
```

Then kill by port (repeat until it reports free - Windows needs a moment to
release the socket):

```powershell
foreach ($p in 8000,3000,5173) {
  Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
}
```

Do **not** blanket-kill `python.exe`: Jupyter kernels run from the same
interpreter, and killing them loses work.

## "It is still broken" after a code change

Two separate caches, both worth ruling out before debugging logic:

1. **A stale server process** (above). `tsx watch` and `uvicorn --reload` are
   usually reliable, but only if the process that is serving is the one holding
   the terminal.
2. **A stale browser page.** Vite's HMR cannot always repair a module graph
   after a failed reload, and the page keeps running the old modules. Hard
   reload - Ctrl+Shift+R - before believing a front-end fix did nothing.

## Tests

```bash
npm run typecheck                       # all workspaces
npm run build
cd apps/server && node scripts/ws-smoke.mjs ws://127.0.0.1:3000/ws
cd services/inference && .venv/Scripts/python.exe scripts/test_fusion.py
cd services/inference && .venv/Scripts/python.exe scripts/test_recon_methods.py
```

`ws-smoke` defaults to `ws://127.0.0.1:3000/ws`; pass a URL if you started the
gateway elsewhere. It needs the gateway up, and will exit with an unhelpful
`ErrorEvent` if nothing is listening.

Offline reconstruction analysis (see `RECONSTRUCTION.md`):

```bash
cd services/inference
.venv/Scripts/python.exe scripts/analyse_scale.py <session>
.venv/Scripts/python.exe scripts/replay_http.py <session> --methods voxel,tsdf
```

## Environment

`.env.example` lists the knobs. The ones that matter in practice:

- `OV_DEVICE=auto` - GPU if present, else CPU. `GPU`, `CPU`, `NPU`, `AUTO`.
- `INFERENCE_URL` - where the gateway finds the inference (default `:8000`).
- `RECON_LOG_DIR` - where recordings live. Set by `dev.mjs`.
- `RECON_RECORD=1` - record keyframes to `RECON_LOG_DIR`.
- `RECON_CLOUD_EVERY` - send a cloud snapshot every N fused frames; set to 1
  when replaying offline so every frame returns one.
