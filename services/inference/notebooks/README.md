# Model exploration notebooks

Notebooks for verifying each model works on this machine *before* it becomes a
service. See the repo's `docs/MODELS.md` for the reasoning.

`01_model_exploration.ipynb` has been run end-to-end on this machine (CPU):
YOLO11n detects `bus` + 4 `person`; FastSAM-s returns ~90 masks; MobileSAM
returns a single mask for a point prompt; Depth Anything V2 produces a full-size
depth map; CLIP classifies the bus image as `bus` at 0.997.

## Setup (once)

Create an isolated virtual environment for the model work. Use **Python 3.11**
(what `python` resolves to here). From this directory:

```bash
# create + activate a venv at services/inference/.venv
python -m venv ../.venv
# Windows Git Bash:
source ../.venv/Scripts/activate
# (Windows PowerShell: ../.venv/Scripts/Activate.ps1)
# (macOS/Linux: source ../.venv/bin/activate)

python -m pip install --upgrade pip
pip install -r requirements.txt
```

First install is large (PyTorch + friends, ~1-2 GB). Subsequent runs are fast.

If you have an NVIDIA GPU you can install a CUDA build of torch instead; the
notebook auto-detects `cuda`. On this Intel-iGPU machine we stay on CPU here and
move to **OpenVINO** in the service.

## Run

```bash
jupyter lab            # or: jupyter notebook
```

Open **`01_model_exploration.ipynb`** and run all cells top to bottom.

- The first run **downloads model weights** (YOLO11n ~5 MB, FastSAM-s ~23 MB,
  MobileSAM ~40 MB, Depth-Anything-V2-Small ~100 MB, CLIP ~600 MB). They are
  cached under `~/.cache` (HF) and the current dir (Ultralytics).
- Sample images are downloaded to `samples/` (gitignored).
- Section 7 (Florence-2) and section 9 (OpenVINO export) are optional/heavier.

## What good looks like

- **YOLO11n**: a handful of sensible boxes, tens of ms on CPU.
- **FastSAM-s**: many small masks covering the scene, ~0.5-2 s on CPU.
- **MobileSAM**: one clean mask around the point prompt.
- **Depth Anything V2**: near objects bright, far objects dark.
- **CLIP**: the top label for the bus image is `bus`; for zidane, `person`.
- **Section 8**: a recognisable 3D blob for the detected object.

Jot down which models felt fast enough; that feeds the `/scan` endpoint design.

## Troubleshooting

- `torch` wheel fails to install: make sure the venv is active and Python is 3.11.
- Out of memory / very slow: CPU inference at the notebook's default sizes is
  expected to be slow-ish for FastSAM/Florence-2; that's fine for exploration.
- Ultralytics download blocked: it fetches `*.pt` from GitHub releases; set
  `YOLO_CONFIG_DIR` if you need a custom cache.
