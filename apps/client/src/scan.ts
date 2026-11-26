/**
 * Scan HUD with a long-press radial wheel.
 *
 * Hold near the reticle -> a wheel of methods (SAM1 / SAM2 / FastSAM / YOLO /
 * DEPTH) appears. Drag toward one to highlight it, release to run it. Depth
 * replaces the feed; a tap returns to the camera.
 */
import type { MethodInfo, MethodsResponse, ScanResult } from "@metroid-scan/shared";
import { installDebugOverlay, log } from "./lib/debug";

installDebugOverlay();

const video = document.getElementById("video") as HTMLVideoElement;
const overlay = document.getElementById("overlay") as HTMLCanvasElement;
const depthImg = document.getElementById("depthImg") as HTMLImageElement;
const reticle = document.getElementById("reticle")!;
const wheel = document.getElementById("wheel")!;
const statusEl = document.getElementById("status")!;
const hintEl = document.getElementById("hint")!;

const MAX_EDGE = 640;
const LONG_PRESS_MS = 320;
const WHEEL_RADIUS = 118;
const MOVED_THRESHOLD = 14;
const DEAD_ZONE = 34;
const TAU = Math.PI * 2;

let methods: MethodInfo[] = [];
let stream: MediaStream | null = null;
let facing: "environment" | "user" = "environment";
let busy = false;
let depthShown = false;

let pressTimer: number | null = null;
let pressStart = { x: 0, y: 0 };
let moved = false;
let wheelOpen = false;
let selectedIndex = -1;

// --- camera ---------------------------------------------------------------
async function startCamera() {
  stream?.getTracks().forEach((t) => t.stop());
  setStatus("starting camera…");
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: facing },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    setStatus("Hold to scan");
    layoutOverlay();
  } catch (e) {
    setStatus("camera failed");
    log(`getUserMedia failed: ${String(e)}`);
  }
}

function snapshot(): string {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const scale = Math.min(1, MAX_EDGE / Math.max(vw, vh));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(vw * scale);
  canvas.height = Math.round(vh * scale);
  canvas.getContext("2d")!.drawImage(video, 0, 0, canvas.width, canvas.height);
  const url = canvas.toDataURL("image/jpeg", 0.8);
  return url.slice(url.indexOf(",") + 1);
}

function videoRect() {
  const vw = video.videoWidth || 1;
  const vh = video.videoHeight || 1;
  const ew = video.clientWidth;
  const eh = video.clientHeight;
  const scale = Math.min(ew / vw, eh / vh);
  const w = vw * scale;
  const h = vh * scale;
  return { x: (ew - w) / 2, y: (eh - h) / 2, w, h };
}

function layoutOverlay() {
  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.round(video.clientWidth * dpr);
  overlay.height = Math.round(video.clientHeight * dpr);
}

// --- methods + wheel ------------------------------------------------------
async function loadMethods() {
  try {
    const res = await fetch("/api/methods");
    const data = (await res.json()) as MethodsResponse;
    methods = data.methods;
    log(`methods: ${methods.map((m) => m.label).join(", ")} (device ${data.device})`);
  } catch (e) {
    log(`methods load failed: ${String(e)}`);
  }
}

const optionAngle = (i: number) =>
  -Math.PI / 2 + (i * TAU) / Math.max(methods.length, 1);

function angleDelta(a: number, b: number) {
  let d = (a - b) % TAU;
  if (d > Math.PI) d -= TAU;
  if (d < -Math.PI) d += TAU;
  return d;
}

function buildWheel() {
  const cx = window.innerWidth / 2;
  const cy = window.innerHeight / 2;
  wheel.innerHTML = "";
  const hub = document.createElement("div");
  hub.className = "wheel-hub";
  hub.textContent = "scan";
  hub.style.left = `${cx}px`;
  hub.style.top = `${cy}px`;
  wheel.appendChild(hub);

  methods.forEach((m, i) => {
    const a = optionAngle(i);
    const el = document.createElement("div");
    el.className = "wheel-opt" + (m.available ? "" : " unavailable");
    el.innerHTML = `${m.label}<span class="dev">${m.available ? m.device : "n/a"}</span>`;
    el.style.left = `${cx + WHEEL_RADIUS * Math.cos(a)}px`;
    el.style.top = `${cy + WHEEL_RADIUS * Math.sin(a)}px`;
    wheel.appendChild(el);
  });
}

function openWheel() {
  if (!methods.length) return;
  buildWheel();
  selectedIndex = -1;
  wheelOpen = true;
  wheel.classList.add("show");
}

function closeWheel() {
  wheelOpen = false;
  wheel.classList.remove("show");
}

function updateSelection(px: number, py: number) {
  const dx = px - window.innerWidth / 2;
  const dy = py - window.innerHeight / 2;
  let idx = -1;
  if (Math.hypot(dx, dy) > DEAD_ZONE) {
    const angle = Math.atan2(dy, dx);
    let best = Infinity;
    methods.forEach((_, i) => {
      const d = Math.abs(angleDelta(angle, optionAngle(i)));
      if (d < best) {
        best = d;
        idx = i;
      }
    });
  }
  if (idx === selectedIndex) return;
  selectedIndex = idx;
  wheel
    .querySelectorAll<HTMLElement>(".wheel-opt")
    .forEach((el, i) => el.classList.toggle("selected", i === idx));
}

// --- pointer handling -----------------------------------------------------
function onPointerDown(e: PointerEvent) {
  if (busy || (e.target as HTMLElement).closest("#topbar")) return;
  pressStart = { x: e.clientX, y: e.clientY };
  moved = false;
  pressTimer = window.setTimeout(() => {
    pressTimer = null;
    openWheel();
  }, LONG_PRESS_MS);
}

function onPointerMove(e: PointerEvent) {
  if (wheelOpen) {
    updateSelection(e.clientX, e.clientY);
    return;
  }
  if (!pressTimer) return;
  if (Math.hypot(e.clientX - pressStart.x, e.clientY - pressStart.y) > MOVED_THRESHOLD) {
    moved = true;
    window.clearTimeout(pressTimer);
    pressTimer = null;
  }
}

function onPointerUp() {
  if (pressTimer) {
    window.clearTimeout(pressTimer);
    pressTimer = null;
  }
  if (wheelOpen) {
    const method = methods[selectedIndex];
    closeWheel();
    if (method?.available) void executeMethod(method);
    return;
  }
  if (!moved && depthShown) hideDepth();
}

// --- actions --------------------------------------------------------------
async function executeMethod(method: MethodInfo) {
  setBusy(true);
  setStatus(`${method.label}…`, true);
  const image = snapshot();
  const t0 = performance.now();
  try {
    const res = await fetch("/api/scan", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        image,
        frameId: String(Date.now()),
        aimX: 0.5,
        aimY: 0.5,
        method: method.id,
      }),
    });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    const result = (await res.json()) as ScanResult;
    const rtt = performance.now() - t0;

    if (result.kind === "depth") {
      showDepth(result);
      setStatus(`Depth ${ms(result.timing.depth)} · ${rtt.toFixed(0)}ms`);
    } else {
      draw(result);
      const n = result.detections.length;
      setStatus(`${method.label} · ${n} mask${n === 1 ? "" : "s"} · ${rtt.toFixed(0)}ms`);
      log(`${method.id}: ${n} masks in ${rtt.toFixed(0)} ms`);
    }
  } catch (e) {
    setStatus("Error");
    log(`scan failed: ${String(e)}`);
  } finally {
    setBusy(false);
  }
}

function showDepth(result: ScanResult) {
  if (!result.depth) return;
  depthImg.src = `data:image/png;base64,${result.depth.png}`;
  depthImg.classList.add("show");
  overlay.classList.add("hidden");
  depthShown = true;
  hintEl.textContent = "Tap to go back";
}

function hideDepth() {
  depthImg.classList.remove("show");
  depthImg.removeAttribute("src");
  overlay.classList.remove("hidden");
  depthShown = false;
  hintEl.textContent = "Hold to scan";
}

function draw(result: ScanResult) {
  const ctx = overlay.getContext("2d")!;
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const rect = videoRect();
  const sx = rect.w / result.width;
  const sy = rect.h / result.height;
  const aimedIndex = result.aimed?.detectionIndex ?? -1;

  ctx.font = "13px ui-monospace, monospace";
  result.detections.forEach((d, i) => {
    const isAimed = i === aimedIndex;
    const colour = isAimed ? "#ffd54f" : "#7cf7c4";
    ctx.strokeStyle = colour;
    ctx.fillStyle = colour;

    if (d.contour2d && d.contour2d.length > 2) {
      ctx.beginPath();
      d.contour2d.forEach(([u, v], k) => {
        const px = rect.x + u * sx;
        const py = rect.y + v * sy;
        if (k === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.closePath();
      ctx.fillStyle = isAimed ? "rgba(255, 213, 79, 0.18)" : "rgba(124, 247, 196, 0.10)";
      ctx.fill();
      ctx.strokeStyle = colour;
      ctx.lineWidth = isAimed ? 4 : 2;
      ctx.stroke();
    } else {
      const [x1, y1, x2, y2] = d.box;
      ctx.lineWidth = 3;
      ctx.strokeRect(rect.x + x1 * sx, rect.y + y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
    }

    const [x1, y1] = d.box;
    ctx.fillStyle = colour;
    ctx.fillText(d.label, rect.x + x1 * sx + 2, Math.max(rect.y + y1 * sy - 4, 14));
  });
}

// --- helpers --------------------------------------------------------------
function setStatus(text: string, scanning = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("busy", scanning);
}

function setBusy(value: boolean) {
  busy = value;
  reticle.classList.toggle("scanning", value);
}

const ms = (v?: number) => (v == null ? "-" : `${v.toFixed(0)}ms`);

async function init() {
  // Kill the browser's long-press menu ("copy video frame") so a hold only
  // opens our wheel.
  window.addEventListener("contextmenu", (e) => e.preventDefault());

  await loadMethods();
  await startCamera();

  window.addEventListener("pointerdown", onPointerDown);
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);
  window.addEventListener("pointercancel", onPointerUp);
  window.addEventListener("resize", () => {
    layoutOverlay();
    if (wheelOpen) buildWheel();
  });
  video.addEventListener("loadedmetadata", layoutOverlay);
}

void init();
