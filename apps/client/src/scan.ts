/**
 * Detection / depth test page.
 *
 * Live camera -> capture a downscaled still -> POST /api/scan -> draw the
 * result. Two modes:
 *   * Scan  - YOLO boxes + the aimed object's segmentation silhouette.
 *   * Depth - the depth model's output, shown in place of the camera feed.
 */
import type { ScanResult } from "@metroid-scan/shared";
import { installDebugOverlay, log } from "./lib/debug";

installDebugOverlay();

const video = document.getElementById("video") as HTMLVideoElement;
const overlay = document.getElementById("overlay") as HTMLCanvasElement;
const depthImg = document.getElementById("depthImg") as HTMLImageElement;
const reticle = document.getElementById("reticle")!;
const scanBtn = document.getElementById("scanBtn") as HTMLButtonElement;
const depthBtn = document.getElementById("depthBtn") as HTMLButtonElement;
const fastToggle = document.getElementById("fastToggle") as HTMLButtonElement;
const backBtn = document.getElementById("backBtn") as HTMLButtonElement;
const statusEl = document.getElementById("status")!;
const cardEl = document.getElementById("card")!;

const MAX_EDGE = 640;
let stream: MediaStream | null = null;
let facing: "environment" | "user" = "environment";
let busy = false;
let fastMask = false;

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
    scanBtn.disabled = false;
    depthBtn.disabled = false;
    setStatus("Ready");
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
  const width = Math.round(vw * scale);
  const height = Math.round(vh * scale);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d")!.drawImage(video, 0, 0, width, height);
  const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
  return dataUrl.slice(dataUrl.indexOf(",") + 1);
}

/** The rect the video is actually drawn in, given object-fit: contain. */
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

function draw(result: ScanResult) {
  const ctx = overlay.getContext("2d")!;
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  const rect = videoRect();
  const sx = rect.w / result.width;
  const sy = rect.h / result.height;
  const aimed = result.aimed;
  const aimedIndex = aimed?.detectionIndex ?? -1;

  ctx.font = "13px ui-monospace, monospace";

  result.detections.forEach((d, i) => {
    const isAimed = i === aimedIndex;
    const colour = isAimed ? "#ffd54f" : "#7cf7c4";
    ctx.strokeStyle = colour;
    ctx.fillStyle = colour;
    ctx.lineWidth = 3;

    const [x1, y1, x2, y2] = d.box;
    const x = rect.x + x1 * sx;
    const y = rect.y + y1 * sy;

    if (isAimed && aimed?.contour2d && aimed.contour2d.length > 2) {
      // Smooth silhouette from the segmentation mask.
      ctx.beginPath();
      aimed.contour2d.forEach(([u, v], k) => {
        const px = rect.x + u * sx;
        const py = rect.y + v * sy;
        if (k === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.closePath();
      ctx.fillStyle = "rgba(255, 213, 79, 0.18)";
      ctx.fill();
      ctx.strokeStyle = colour;
      ctx.lineWidth = 4;
      ctx.stroke();
    } else {
      ctx.strokeRect(x, y, (x2 - x1) * sx, (y2 - y1) * sy);
    }

    ctx.fillStyle = colour;
    ctx.fillText(`${d.label} ${d.score.toFixed(2)}`, x + 2, Math.max(y - 4, 14));
  });
}

function setStatus(text: string, scanning = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("busy", scanning);
  reticle.classList.toggle("scanning", scanning);
}

function setBusy(value: boolean) {
  busy = value;
  scanBtn.disabled = value;
  depthBtn.disabled = value;
}

const ms = (v?: number) => (v == null ? "-" : `${v.toFixed(0)}ms`);

function showDepth(result: ScanResult) {
  if (!result.depth) return;
  depthImg.src = `data:image/png;base64,${result.depth.png}`;
  depthImg.classList.add("show");
  overlay.classList.add("hidden");
  backBtn.classList.add("show");
  cardEl.textContent = `Depth · range ${result.depth.min.toFixed(2)}–${result.depth.max.toFixed(2)}`;
}

function hideDepth() {
  depthImg.classList.remove("show");
  depthImg.removeAttribute("src");
  overlay.classList.remove("hidden");
  backBtn.classList.remove("show");
  setStatus("Ready");
}

async function capture(mode: "scan" | "depth") {
  if (busy) return;
  setBusy(true);
  setStatus(mode === "depth" ? "Measuring depth…" : "Scanning…", true);
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
        mode,
        segmenter: fastMask ? "yolo11" : undefined,
      }),
    });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    const result = (await res.json()) as ScanResult;
    const rtt = performance.now() - t0;

    if (mode === "depth") {
      showDepth(result);
      setStatus(`Depth ${ms(result.timing.depth)} · ${rtt.toFixed(0)}ms round trip`);
    } else {
      draw(result);
      const aimed = result.aimed;
      cardEl.textContent = aimed
        ? `SCANNED: ${aimed.name}${aimed.pointCount ? ` · ${aimed.pointCount} pts` : ""}`
        : "No object under the reticle";
      setStatus(
        `Scan ${ms(result.timing.detect)} det · ${ms(result.timing.segment)} seg · ` +
          `${ms(result.timing.depth)} depth · ${rtt.toFixed(0)}ms round trip`,
      );
      log(`scan ok: ${result.detections.length} dets · ${rtt.toFixed(0)} ms rtt`);
    }
  } catch (e) {
    setStatus("Error");
    cardEl.textContent = "failed — is the gateway + inference running?";
    log(`scan failed: ${String(e)}`);
  } finally {
    setBusy(false);
  }
}

scanBtn.addEventListener("click", () => void capture("scan"));
depthBtn.addEventListener("click", () => void capture("depth"));
backBtn.addEventListener("click", hideDepth);
fastToggle.addEventListener("click", () => {
  fastMask = !fastMask;
  fastToggle.classList.toggle("active", fastMask);
  fastToggle.textContent = `Fast mask: ${fastMask ? "on" : "off"}`;
});
window.addEventListener("resize", layoutOverlay);
video.addEventListener("loadedmetadata", layoutOverlay);

void startCamera();
