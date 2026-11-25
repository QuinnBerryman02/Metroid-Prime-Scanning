/**
 * Stage A verification page.
 *
 * Live camera -> capture a downscaled still -> POST /api/scan -> draw the
 * returned YOLO boxes, highlighting the one under the reticle. This exercises
 * the whole phone -> gateway -> inference -> phone loop without any AR.
 */
import type { ScanResult } from "@metroid-scan/shared";
import { installDebugOverlay, log } from "./lib/debug";

installDebugOverlay();

const video = document.getElementById("video") as HTMLVideoElement;
const overlay = document.getElementById("overlay") as HTMLCanvasElement;
const captureBtn = document.getElementById("captureBtn") as HTMLButtonElement;
const switchBtn = document.getElementById("switchBtn") as HTMLButtonElement;
const statusEl = document.getElementById("status")!;
const cardEl = document.getElementById("card")!;

const MAX_EDGE = 640;
let stream: MediaStream | null = null;
let facing: "environment" | "user" = "environment";

async function startCamera() {
  stream?.getTracks().forEach((t) => t.stop());
  statusEl.textContent = "starting camera…";
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
    captureBtn.disabled = false;
    statusEl.textContent = `${video.videoWidth}×${video.videoHeight} · ready`;
    layoutOverlay();
  } catch (e) {
    statusEl.textContent = `camera failed: ${String(e)}`;
    log(`getUserMedia failed: ${String(e)}`);
  }
}

function snapshot(): { base64: string; width: number; height: number } {
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
  return { base64: dataUrl.slice(dataUrl.indexOf(",") + 1), width, height };
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
  const aimedIndex = result.aimed?.detectionIndex ?? -1;

  ctx.lineWidth = 3;
  ctx.font = "13px ui-monospace, monospace";
  result.detections.forEach((d, i) => {
    const [x1, y1, x2, y2] = d.box;
    const aimed = i === aimedIndex;
    const colour = aimed ? "#ffd54f" : "#7cf7c4";
    ctx.strokeStyle = colour;
    ctx.fillStyle = colour;
    const x = rect.x + x1 * sx;
    const y = rect.y + y1 * sy;
    ctx.strokeRect(x, y, (x2 - x1) * sx, (y2 - y1) * sy);
    const label = `${d.label} ${d.score.toFixed(2)}`;
    ctx.fillText(label, x + 2, Math.max(y - 4, 14));
  });
}

async function capture() {
  captureBtn.disabled = true;
  const { base64 } = snapshot();
  const t0 = performance.now();
  try {
    const res = await fetch("/api/scan", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        image: base64,
        frameId: String(Date.now()),
        aimX: 0.5,
        aimY: 0.5,
      }),
    });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    const result = (await res.json()) as ScanResult;
    const rtt = performance.now() - t0;
    draw(result);
    const server = result.timing.detect?.toFixed(0) ?? "?";
    log(`scan ok: ${result.detections.length} dets, server ${server} ms, rtt ${rtt.toFixed(0)} ms`);
    cardEl.textContent = result.aimed
      ? `SCANNED: ${result.aimed.name} (class ${result.aimed.classId})`
      : "No object under the reticle";
  } catch (e) {
    log(`scan failed: ${String(e)}`);
    cardEl.textContent = "scan failed — is the gateway + inference running?";
  } finally {
    captureBtn.disabled = false;
  }
}

captureBtn.addEventListener("click", () => void capture());
switchBtn.addEventListener("click", () => {
  facing = facing === "environment" ? "user" : "environment";
  void startCamera();
});
window.addEventListener("resize", layoutOverlay);
video.addEventListener("loadedmetadata", layoutOverlay);

void startCamera();
