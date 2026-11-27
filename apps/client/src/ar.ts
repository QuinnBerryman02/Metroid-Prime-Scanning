/**
 * AR scan visor with anchoring (Stage B).
 *
 * Tap Scan with the reticle on an object:
 *   1. grab the camera frame (WebXR `camera-access`)
 *   2. hit-test at the reticle -> metric distance to the surface
 *   3. POST /scan -> camera-space contour + OBB + relative depth
 *   4. scale = hitTestDistance / depthAtCentre  (relative depth -> metres)
 *   5. unproject the contour into world space, anchor it to an XRAnchor
 *
 * The camera-space convention is OpenCV's (+x right, +y down, +z forward); the
 * phone owns the transform (see docs/LIFTING.md sections 4-5).
 */
import { Engine } from "@babylonjs/core/Engines/engine";
import { Scene } from "@babylonjs/core/scene";
import { FreeCamera } from "@babylonjs/core/Cameras/freeCamera";
import { Quaternion, Vector3 } from "@babylonjs/core/Maths/math.vector";
import { Color3, Color4 } from "@babylonjs/core/Maths/math.color";
import { MeshBuilder } from "@babylonjs/core/Meshes/meshBuilder";
import { TransformNode } from "@babylonjs/core/Meshes/transformNode";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { DynamicTexture } from "@babylonjs/core/Materials/Textures/dynamicTexture";
import { Mesh } from "@babylonjs/core/Meshes/mesh";
import { WebXRDefaultExperience } from "@babylonjs/core/XR/webXRDefaultExperience";
import { WebXRFeatureName } from "@babylonjs/core/XR/webXRFeaturesManager";
import "@babylonjs/core/XR/features/WebXRHitTest";
import "@babylonjs/core/XR/features/WebXRAnchorSystem";
import "@babylonjs/core/XR/features/WebXRDOMOverlay";
import "@babylonjs/core/XR/features/WebXRPlaneDetector";
import type { WebXRHitTest } from "@babylonjs/core/XR/features/WebXRHitTest";
import type { WebXRAnchorSystem } from "@babylonjs/core/XR/features/WebXRAnchorSystem";
import type {
  IWebXRPlane,
  WebXRPlaneDetector,
} from "@babylonjs/core/XR/features/WebXRPlaneDetector";
import type { Camera } from "@babylonjs/core/Cameras/camera";
import type { Obb, Quat, ReconObjectInfo, ScanResult, Vec3 } from "@metroid-scan/shared";
import { installDebugOverlay, log } from "./lib/debug";
import { createDiagnostics } from "./lib/report";
import { GatewayClient } from "./lib/gateway";

installDebugOverlay();
const diag = createDiagnostics("ar");
diag.installAutoSend();
diag.info("ua", navigator.userAgent);
log("ar.ts loaded (anchoring)");

const sessionId = crypto.randomUUID();
const RATE = 66;
const SCAN_METHOD = "yolo"; // fast, iGPU; swap to "sam2" for the best mask
const MAX_EDGE = 640;
// If the hit-test finds no surface we still allow scanning, assuming the object
// is this far away (docs/LIFTING.md section 4 option 4: fixed fallback).
const DEFAULT_DISTANCE = 1.5;

// --- DOM --------------------------------------------------------------------
const canvas = document.getElementById("app") as HTMLCanvasElement;
const startOverlay = document.getElementById("start")!;
const startBtn = document.getElementById("startBtn") as HTMLButtonElement;
const hud = document.getElementById("hud")!;
const keyBtn = document.getElementById("keyBtn") as HTMLButtonElement;
const scanBtn = document.getElementById("scanBtn") as HTMLButtonElement;
const clearBtn = document.getElementById("clearBtn") as HTMLButtonElement;
const reportBtn = document.getElementById("reportBtn") as HTMLButtonElement;
const exitBtn = document.getElementById("exitBtn") as HTMLButtonElement;
const statusEl = document.getElementById("status")!;

function setStatus(html: string) {
  statusEl.innerHTML = html;
}

// --- gateway ----------------------------------------------------------------
const gateway = new GatewayClient("device", sessionId, {
  onStatus: (s) => log(`ws ${s}`),
  onReconObjects: (msg) => updateObjectMarkers(msg.objects),
});
gateway.connect();

// --- scannable objects ------------------------------------------------------
// The server's objects, drawn as circles at their anchors. Found = orange.
// Holding the reticle on one for OBJECT_HOLD_MS turns it green and pops its
// info. Scanned ids are remembered for the session, so a re-found object
// stays green.
const OBJECT_HOLD_MS = 2000;
const OBJECT_SIZE = 0.22; // m, circle diameter
const ORANGE = "#ff8c1a";
const GREEN = "#3bff7a";
const objectInfo = document.getElementById("objectInfo")!;
objectInfo.addEventListener("click", () => {
  objectInfo.hidden = true;
});

interface ObjectMarker {
  id: number;
  mesh: Mesh;
  tex: DynamicTexture;
  info: ReconObjectInfo;
  scanned: boolean;
  hold: number; // ms the reticle has been on it
  pop: number; // 0..1 after a scan completes
}
const markers = new Map<number, ObjectMarker>();
const scannedIds = new Set<number>();

function paintCircle(m: ObjectMarker, colour: string) {
  const ctx = m.tex.getContext() as unknown as CanvasRenderingContext2D;
  ctx.clearRect(0, 0, 256, 256);
  ctx.beginPath();
  ctx.arc(128, 128, 112, 0, Math.PI * 2);
  ctx.fillStyle = colour;
  ctx.fill();
  ctx.lineWidth = 10;
  ctx.strokeStyle = "#0b1014";
  ctx.stroke();
  m.tex.update();
}

function updateObjectMarkers(objects: ReconObjectInfo[]) {
  const live = new Set(objects.map((o) => o.id));
  for (const [id, m] of markers) {
    if (!live.has(id)) {
      m.mesh.material?.dispose();
      m.tex.dispose();
      m.mesh.dispose();
      markers.delete(id);
    }
  }
  for (const o of objects) {
    let m = markers.get(o.id);
    if (!m) {
      const mesh = MeshBuilder.CreatePlane(`object-circle-${o.id}`, { size: OBJECT_SIZE }, scene);
      mesh.billboardMode = Mesh.BILLBOARDMODE_ALL;
      const tex = new DynamicTexture(`object-tex-${o.id}`, { width: 256, height: 256 }, scene, false);
      tex.hasAlpha = true;
      const mat = new StandardMaterial(`object-mat-${o.id}`, scene);
      mat.diffuseTexture = tex;
      mat.emissiveTexture = tex;
      mat.useAlphaFromDiffuseTexture = true;
      mat.disableLighting = true;
      mat.backFaceCulling = false;
      mesh.material = mat;
      m = { id: o.id, mesh, tex, info: o, scanned: scannedIds.has(o.id), hold: 0, pop: 1 };
      markers.set(o.id, m);
      paintCircle(m, m.scanned ? GREEN : ORANGE);
    }
    m.info = o;
    m.mesh.position = new Vector3(o.anchor[0], o.anchor[1], o.anchor[2]);
  }
}

function completeScan(m: ObjectMarker) {
  m.scanned = true;
  m.hold = 0;
  m.pop = 0;
  scannedIds.add(m.id);
  paintCircle(m, GREEN);
  objectInfo.innerHTML =
    `<strong>${m.info.label}</strong> &middot; ${m.info.views} views` +
    `<p>Description not generated yet.</p>`;
  objectInfo.hidden = false;
  log(`scanned: ${m.info.label} #${m.id}`);
}

function updateObjectHold() {
  const dtMs = engine.getDeltaTime();
  const pick = scene.pick(engine.getRenderWidth() / 2, engine.getRenderHeight() / 2, (mesh) =>
    mesh.name.startsWith("object-circle-"),
  );
  const onId =
    pick?.hit && pick.pickedMesh
      ? Number(pick.pickedMesh.name.slice("object-circle-".length))
      : null;
  for (const m of markers.values()) {
    if (m.scanned) {
      // pop: swell to 1.4x then settle back to 1x
      m.pop = Math.min(1, m.pop + dtMs / 400);
      m.mesh.scaling.setAll(1 + 0.4 * (1 - m.pop));
      continue;
    }
    if (m.id === onId) m.hold = Math.min(OBJECT_HOLD_MS, m.hold + dtMs);
    else m.hold = Math.max(0, m.hold - dtMs * 2);
    m.mesh.scaling.setAll(1 + 0.4 * (m.hold / OBJECT_HOLD_MS));
    if (m.hold >= OBJECT_HOLD_MS) completeScan(m);
  }
}

// --- ARCore planes ----------------------------------------------------------
// Plane detection is ARCore's free environment model: metric polygons for the
// floor / walls / tables. We stream them to the viewer (and, later, to the
// server as the structure set for object segmentation). Updates can fire many
// times per frame, so throttle them.
const PLANE_UPDATE_MS = 250;
const planeSentAt = new Map<number, number>();

/** Centroid + robust unit normal (Newell's method) of a planar polygon. */
function planeCenterNormal(poly: Vector3[]): { center: Vector3; normal: Vector3 } {
  const center = new Vector3(0, 0, 0);
  for (const p of poly) center.addInPlace(p);
  center.scaleInPlace(1 / Math.max(1, poly.length));
  const n = new Vector3(0, 0, 0);
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i]!;
    const b = poly[(i + 1) % poly.length]!;
    n.x += (a.y - b.y) * (a.z + b.z);
    n.y += (a.z - b.z) * (a.x + b.x);
    n.z += (a.x - b.x) * (a.y + b.y);
  }
  if (n.lengthSquared() > 1e-9) n.normalize();
  else n.set(0, 1, 0);
  return { center, normal: n };
}

function sendPlane(action: "added" | "updated" | "removed", plane: IWebXRPlane) {
  const local = plane.polygonDefinition;
  if (!local || local.length < 3) return;
  // `polygonDefinition` is in the plane's LOCAL space; the plane's world pose
  // lives in `transformationMatrix` (= planeSpace -> referenceSpace). Applying
  // it is mandatory: without it every plane collapses to its own local frame
  // (which is why walls were appearing flat on the floor).
  const mat = plane.transformationMatrix;
  const poly = mat
    ? local.map((p) => Vector3.TransformCoordinates(p, mat))
    : local;
  const now = performance.now();
  if (action === "updated" && now - (planeSentAt.get(plane.id) ?? 0) < PLANE_UPDATE_MS) {
    return;
  }
  planeSentAt.set(plane.id, now);
  if (action === "removed") planeSentAt.delete(plane.id);
  const { center, normal } = planeCenterNormal(poly);
  const orientation =
    ((plane.xrPlane as unknown as { orientation?: string })?.orientation as
      | "horizontal"
      | "vertical"
      | undefined) ?? "unknown";
  gateway.sendPlane({
    id: plane.id,
    action,
    orientation,
    polygon: poly.map((p) => [p.x, p.y, p.z] as Vec3),
    center: [center.x, center.y, center.z],
    normal: [normal.x, normal.y, normal.z],
  });
  if (action !== "updated") {
    log(
      `plane ${action} #${plane.id} ${orientation} (${poly.length}pts) ` +
        `c=[${center.x.toFixed(2)}, ${center.y.toFixed(2)}, ${center.z.toFixed(2)}]`,
    );
  }
}

// --- Babylon ----------------------------------------------------------------
const engine = new Engine(canvas, true, { stencil: true });
const scene = new Scene(engine);
scene.clearColor = new Color4(0, 0, 0, 0);

// Babylon needs a scene camera *before* the XR rig takes over. Without one, the
// first scene.render() throws "No camera defined", and because the engine
// queues the next frame *after* the render callback, that single throw kills
// the render loop for good (no hit-test -> Scan stays disabled). This fallback
// is also saved as the "non-VR" camera the XR helper restores on exit.
const fallbackCamera = new FreeCamera("fallback", new Vector3(0, 1.5, 0), scene);
fallbackCamera.setTarget(new Vector3(0, 1.5, -1));

const reticleMat = new StandardMaterial("reticleMat", scene);
reticleMat.emissiveColor = new Color3(0.49, 0.97, 0.77);
reticleMat.disableLighting = true;
reticleMat.alpha = 0.9;

const reticle = MeshBuilder.CreateTorus(
  "reticle",
  { diameter: 0.16, thickness: 0.02, tessellation: 32 },
  scene,
);
reticle.rotationQuaternion = Quaternion.Identity();
reticle.material = reticleMat;
reticle.isVisible = false;

let xr: WebXRDefaultExperience | null = null;
let hitTest: WebXRHitTest | null = null;
let anchorSystem: WebXRAnchorSystem | null = null;
let planeDetector: WebXRPlaneDetector | null = null;
let currentHit: { position: Vector3; quaternion: Quaternion } | null = null;
let lastHitCount = -1;
// Independent hit-test probe so the report can tell us whether Babylon's
// feature is broken or the room simply has no trackable surface.
let rawHitSource: XRHitTestSource | null = null;
let rawHitCount = 0;
let frameCount = 0;
let lastPoseSend = 0;
let anchorCount = 0;

const placedAnchors: { remove(): void }[] = [];
const placedNodes: TransformNode[] = [];

// --- camera-access capture --------------------------------------------------
interface CaptureFrame {
  base64: string;
  width: number;
  height: number;
  fovDeg: number;
  /** Laplacian variance of the greyscale image; higher = sharper. */
  sharpness: number;
}

let pendingCapture: ((f: CaptureFrame | null) => void) | null = null;

function grabFrame(): Promise<CaptureFrame | null> {
  return new Promise((resolve) => {
    // Only one capture can be in flight (there is a single `pendingCapture`
    // slot); the loser gets null rather than hijacking the other's promise.
    if (pendingCapture) {
      resolve(null);
      return;
    }
    pendingCapture = resolve;
    window.setTimeout(() => {
      if (pendingCapture === resolve) {
        pendingCapture = null;
        resolve(null);
      }
    }, 1500);
  });
}

/** Laplacian variance - the standard "is this frame blurry" score. */
function laplacianVariance(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
): number {
  const { data } = ctx.getImageData(0, 0, w, h);
  const n = w * h;
  const grey = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const p = i * 4;
    grey[i] = 0.299 * data[p]! + 0.587 * data[p + 1]! + 0.114 * data[p + 2]!;
  }
  let sum = 0;
  let sumSq = 0;
  let count = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const lap =
        -4 * grey[i]! + grey[i - 1]! + grey[i + 1]! + grey[i - w]! + grey[i + w]!;
      sum += lap;
      sumSq += lap * lap;
      count++;
    }
  }
  if (count === 0) return 0;
  const mean = sum / count;
  return sumSq / count - mean * mean;
}

/** Horizontal FOV from the XR view's projection matrix. */
function viewFov(view: XRView): number {
  const fx = view.projectionMatrix[0];
  if (!fx || !Number.isFinite(fx) || fx <= 0) return 60;
  return (2 * Math.atan(1 / fx) * 180) / Math.PI;
}

function captureNow(): CaptureFrame | null {
  const sm = xr?.baseExperience?.sessionManager;
  const gl = engine._gl;
  diag.info("capture.hasSessionManager", Boolean(sm));
  diag.info("capture.hasGl", Boolean(gl));
  if (!sm || !gl) {
    diag.error("capture", new Error(`missing ${sm ? "" : "session "}${gl ? "" : "gl"}`));
    return null;
  }
  diag.info(
    "capture.enabledFeatures",
    (sm.session as unknown as { enabledFeatures?: string[] }).enabledFeatures ?? null,
  );
  diag.info("capture.glClass", gl.constructor?.name ?? "?");

  const frame = sm.currentFrame;
  const refSpace = sm.referenceSpace;
  const view = frame?.getViewerPose(refSpace)?.views?.[0];
  if (!view) {
    diag.error("capture", new Error("no XR view on this frame"));
    return null;
  }
  diag.info(
    "capture.viewCameraAttr",
    Boolean((view as unknown as { camera?: unknown }).camera),
  );

  if (typeof XRWebGLBinding === "undefined") {
    diag.error("capture", new Error("XRWebGLBinding is undefined"));
    return null;
  }

  let texture: WebGLTexture | null = null;
  try {
    // `getCameraImage` is the camera-access spec; not in this @types/webxr yet.
    const binding = new XRWebGLBinding(sm.session, gl) as unknown as {
      getCameraImage?: (arg: unknown) => WebGLTexture | null;
    };
    diag.info("capture.getCameraImageType", typeof binding.getCameraImage);
    if (typeof binding.getCameraImage !== "function") {
      diag.error("capture", new TypeError("getCameraImage is not a function"));
      return null;
    }
    // Chrome validates parameter 1 as an XRCamera (the spec text says XRView),
    // so try `view.camera` first, then the view itself.
    const cameraArg = (view as unknown as { camera?: unknown }).camera;
    const args: unknown[] = cameraArg ? [cameraArg, view] : [view];
    let lastErr: unknown = null;
    for (const arg of args) {
      try {
        texture = binding.getCameraImage(arg);
        if (texture) break;
      } catch (e) {
        lastErr = e;
      }
    }
    if (!texture && lastErr) throw lastErr;
  } catch (e) {
    diag.error("getCameraImage", e);
    return null;
  }
  if (!texture) {
    diag.error("capture", new Error("getCameraImage returned null"));
    return null;
  }
  diag.info("capture.textureClass", texture.constructor?.name ?? "?");

  const cam = (view as unknown as { camera?: { width: number; height: number } })
    .camera;
  const w = cam?.width ?? gl.drawingBufferWidth;
  const h = cam?.height ?? gl.drawingBufferHeight;
  diag.info("capture.cameraWH", [w, h]);

  let pixels: Uint8Array;
  try {
    const prevFb = gl.getParameter(gl.FRAMEBUFFER_BINDING) as WebGLFramebuffer | null;
    const fb = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    diag.info(
      "capture.framebufferStatus",
      status === gl.FRAMEBUFFER_COMPLETE ? "complete" : status,
    );
    pixels = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    const glError = gl.getError();
    if (glError) diag.info("capture.glError", glError);
    gl.bindFramebuffer(gl.FRAMEBUFFER, prevFb);
    gl.deleteFramebuffer(fb);
  } catch (e) {
    diag.error("readPixels", e);
    return null;
  }

  try {
    // WebGL is bottom-up; flip vertically and downscale into a JPEG.
    const src = document.createElement("canvas");
    src.width = w;
    src.height = h;
    src
      .getContext("2d")!
      .putImageData(new ImageData(new Uint8ClampedArray(pixels), w, h), 0, 0);

    const scale = Math.min(1, MAX_EDGE / Math.max(w, h));
    const outW = Math.max(1, Math.round(w * scale));
    const outH = Math.max(1, Math.round(h * scale));
    const dst = document.createElement("canvas");
    dst.width = outW;
    dst.height = outH;
    const ctx = dst.getContext("2d")!;
    ctx.translate(0, outH);
    ctx.scale(1, -1);
    ctx.drawImage(src, 0, 0, outW, outH);
    const sharpness = laplacianVariance(ctx, outW, outH);
    const url = dst.toDataURL("image/jpeg", 0.8);

    return {
      base64: url.slice(url.indexOf(",") + 1),
      width: outW,
      height: outH,
      fovDeg: viewFov(view),
      sharpness,
    };
  } catch (e) {
    diag.error("capture.encode", e);
    return null;
  }
}

// --- keyframes (MS3.0) ------------------------------------------------------
// A keyframe is a frame worth *fusing* later: new enough, moved enough, sharp
// enough, and not a duplicate. Captures are expensive (readPixels + JPEG + a
// blur score), so gate them hard - ~2-3 Hz is plenty for reconstruction.
// Auto-streaming: a keyframe every ~300 ms while the phone is moving, but only
// when it has actually moved or turned (the minimums below). A still phone
// sends nothing, and the Key button still forces one.
const AUTO_KEYFRAMES = true;
const KF_MIN_INTERVAL_MS = 300;
const KF_MIN_MOVE = 0.02; // metres
const KF_MIN_ROT = 0.026; // radians (~1.5 deg)
// Conservative floor: reject only truly ruined frames until device reports give
// us real sharpness numbers to calibrate against.
const KF_BLUR_MIN = 3;

let inAr = false;
let kfCount = 0;
let kfSending = false;
let kfLastMs = 0;
let kfLastPos: Vector3 | null = null;
let kfLastQuat: Quaternion | null = null;

/**
 * Row-major 4x4 `worldFromCamera` in the OpenCV convention (+x right, +y down,
 * +z forward) - the frame the server unprojects and fuses in. Built by hand so
 * we never have to reason about Babylon's left-handed, +y-up camera basis.
 */
function cameraWorldMatrix(camera: Camera): number[] {
  const p = camera.globalPosition;
  const right = camera.getDirection(Vector3.Right()).normalize();
  const up = camera.getDirection(Vector3.Up()).normalize();
  const forward = camera.getDirection(Vector3.Forward()).normalize();
  const x = [right.x, right.y, right.z];
  const y = [-up.x, -up.y, -up.z]; // OpenCV +y is down
  const z = [forward.x, forward.y, forward.z];
  // Columns are the axes: world = M * [x, y, z, 1]^T
  return [
    x[0]!, y[0]!, z[0]!, p.x,
    x[1]!, y[1]!, z[1]!, p.y,
    x[2]!, y[2]!, z[2]!, p.z,
    0, 0, 0, 1,
  ];
}

/** Angle between two rotations, in radians. */
function quatAngle(a: Quaternion, b: Quaternion): number {
  const dot = Math.min(1, Math.abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w));
  return 2 * Math.acos(dot);
}

function resetKeyframes() {
  kfLastPos = null;
  kfLastQuat = null;
  kfLastMs = 0;
}

async function captureKeyframe(manual = false) {
  if (!inAr || kfSending) return;
  if (!manual && !AUTO_KEYFRAMES) return;
  const cam = scene.activeCamera;
  if (!cam) return;
  const now = performance.now();
  if (!manual && now - kfLastMs < KF_MIN_INTERVAL_MS) return;

  const scale = new Vector3();
  const quat = new Quaternion();
  const pos = new Vector3();
  cam.getWorldMatrix().decompose(scale, quat, pos);

  let reason: "first" | "move" | "rotate" | "both" | "manual" = manual
    ? "manual"
    : "first";
  if (!manual && kfLastPos && kfLastQuat) {
    const moved = Vector3.Distance(pos, kfLastPos) > KF_MIN_MOVE;
    const rotated = quatAngle(quat, kfLastQuat) > KF_MIN_ROT;
    if (!moved && !rotated) return;
    reason = moved && rotated ? "both" : moved ? "move" : "rotate";
  }

  kfSending = true;
  try {
    const frame = await grabFrame();
    if (!frame) return;
    if (frame.sharpness < KF_BLUR_MIN) {
      log(`keyframe skipped: blurry (lap=${frame.sharpness.toFixed(1)})`);
      return;
    }
    kfCount++;
    kfLastMs = now;
    kfLastPos = pos.clone();
    kfLastQuat = quat.clone();
    gateway.sendReconFrame({
      frameId: `${sessionId}-kf${kfCount}`,
      position: [pos.x, pos.y, pos.z],
      quaternion: [quat.x, quat.y, quat.z, quat.w],
      pose: cameraWorldMatrix(cam),
      fovDeg: frame.fovDeg,
      width: frame.width,
      height: frame.height,
      image: frame.base64,
      reason,
      sharpness: frame.sharpness,
      hitDistance: currentHit ? Vector3.Distance(currentHit.position, pos) : null,
    });
    log(
      `keyframe #${kfCount} (${reason}) ${frame.width}x${frame.height} ` +
        `lap=${frame.sharpness.toFixed(0)}`,
    );
  } catch (e) {
    diag.error("keyframe", e);
  } finally {
    kfSending = false;
  }
}

// --- XR setup ---------------------------------------------------------------
async function setupXr() {
  xr = await WebXRDefaultExperience.CreateAsync(scene, {
    disableDefaultUI: true,
    disableTeleportation: true,
    disablePointerSelection: true,
    disableNearInteraction: true,
    disableHandTracking: true,
  });
  if (!xr?.baseExperience) throw new Error("WebXR base experience unavailable");
  const fm = xr.baseExperience.featuresManager;

  fm.enableFeature(WebXRFeatureName.DOM_OVERLAY, "latest", {
    element: hud,
    supressXRSelectEvents: true,
  });
  hitTest = fm.enableFeature(WebXRFeatureName.HIT_TEST, "latest", {
    disablePermanentHitTest: false,
    enableTransientHitTest: true,
  }) as WebXRHitTest | null;
  anchorSystem = fm.enableFeature(WebXRFeatureName.ANCHOR_SYSTEM, "latest", {
    doNotRemoveAnchorsOnSessionEnded: false,
  }) as WebXRAnchorSystem | null;
  planeDetector = fm.enableFeature(WebXRFeatureName.PLANE_DETECTION, "latest", {}) as
    | WebXRPlaneDetector
    | null;
  if (planeDetector) {
    planeDetector.onPlaneAddedObservable.add((p) => sendPlane("added", p));
    planeDetector.onPlaneUpdatedObservable.add((p) => sendPlane("updated", p));
    planeDetector.onPlaneRemovedObservable.add((p) => sendPlane("removed", p));
  }

  log(
    `hit-test: ${hitTest ? "on" : "OFF"} | anchors: ${anchorSystem ? "on" : "OFF"} | planes: ${
      planeDetector ? "on" : "OFF"
    }`,
  );

  hitTest?.onHitTestResultObservable.add((results) => {
    const n = results.length;
    if (n !== lastHitCount) {
      lastHitCount = n;
      log(`hit-test: ${n} result(s)`);
    }
    const r = results[0];
    if (r) {
      reticle.isVisible = true;
      reticle.position.copyFrom(r.position);
      reticle.rotationQuaternion!.copyFrom(r.rotationQuaternion);
      currentHit = {
        position: r.position.clone(),
        quaternion: r.rotationQuaternion.clone(),
      };
    } else {
      reticle.isVisible = false;
      currentHit = null;
    }
  });
}

async function enterAr() {
  if (!xr?.baseExperience) return;
  try {
    await xr.baseExperience.enterXRAsync("immersive-ar", "local-floor", xr.renderTarget, {
      requiredFeatures: ["hit-test", "anchors"],
      optionalFeatures: ["camera-access", "plane-detection", "light-estimation", "dom-overlay"],
    });
    startOverlay.style.display = "none";
    hud.style.display = "block";
    inAr = true;
    resetKeyframes();
    const session = xr.baseExperience.sessionManager.session;
    keyBtn.disabled = false;
    scanBtn.disabled = false; // scanning works without a hit (default depth)
    try {
      const viewerSpace = await session.requestReferenceSpace("viewer");
      const request = session.requestHitTestSource?.bind(session);
      rawHitSource = request ? ((await request({ space: viewerSpace })) ?? null) : null;
      diag.info("rawHitSource", rawHitSource ? "created" : "null");
      log(`raw hit-test source: ${rawHitSource ? "created" : "null"}`);
    } catch (e) {
      diag.error("requestHitTestSource", e);
      log(`raw hit-test source failed: ${String(e)}`);
    }
    const enabled =
      (session as unknown as { enabledFeatures?: string[] }).enabledFeatures ?? [];
    diag.info("session.enabledFeatures", enabled);
    log(`features: ${enabled.join(", ") || "(none reported)"}`);
    diag.info("session.blendMode", session.environmentBlendMode ?? null);
    session.addEventListener("end", () => {
      hud.style.display = "none";
      startOverlay.style.display = "flex";
      inAr = false;
      resetKeyframes();
      log(`session ended after ${kfCount} keyframe(s)`);
      diag.info("keyframes", kfCount);
      void diag.send("session-end");
    });
    log("entered immersive-ar");
    window.setTimeout(
      () => log(`activeCamera: ${scene.activeCamera?.name ?? "NONE"}`),
      600,
    );
  } catch (e) {
    log(`enterXRAsync FAILED: ${String(e)}`);
    setStatus(`<b style="color:#ff6b6b">AR failed</b><br>${String(e)}`);
  }
}

// --- camera-space -> world --------------------------------------------------
interface Basis {
  right: Vector3;
  up: Vector3;
  forward: Vector3;
}

function cameraBasis(camera: Camera): Basis {
  return {
    right: camera.getDirection(Vector3.Right()).normalize(),
    up: camera.getDirection(Vector3.Up()).normalize(),
    forward: camera.getDirection(Vector3.Forward()).normalize(),
  };
}

function quatToMat3(q: Quat): number[][] {
  const [x, y, z, w] = q;
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
}

function mat3Apply(m: number[][], v: Vec3): Vec3 {
  return [
    m[0]![0]! * v[0] + m[0]![1]! * v[1] + m[0]![2]! * v[2],
    m[1]![0]! * v[0] + m[1]![1]! * v[1] + m[1]![2]! * v[2],
    m[2]![0]! * v[0] + m[2]![1]! * v[1] + m[2]![2]! * v[2],
  ];
}

function obbEdges(): [number, number][] {
  const edges: [number, number][] = [];
  for (let i = 0; i < 8; i++) {
    for (const bit of [1, 2, 4]) {
      const j = i ^ bit;
      if (j > i) edges.push([i, j]);
    }
  }
  return edges;
}

// --- scan + anchor ----------------------------------------------------------
async function scanAndAnchor() {
  const sm = xr?.baseExperience?.sessionManager;
  if (!sm || !anchorSystem) return;
  const camera = scene.activeCamera;
  if (!camera) return;

  const camPos = camera.globalPosition.clone();
  const basis = cameraBasis(camera);
  const hitPos = currentHit?.position.clone() ?? null;
  const distance = hitPos ? Vector3.Distance(hitPos, camPos) : DEFAULT_DISTANCE;
  diag.info("scan.hitTest", hitPos ? "hit" : "miss");
  diag.info("scan.distance", distance);
  diag.info("scan.hitTestResults", lastHitCount);

  setStatus("capturing…");
  const frame = await grabFrame();
  if (!frame) {
    setStatus("capture failed — report sent");
    void diag.send("capture-failed");
    return;
  }
  log(
    `captured ${frame.width}×${frame.height} fov=${frame.fovDeg.toFixed(1)}° | ` +
      `hit ${distance.toFixed(2)}m from ${camPos.x.toFixed(2)},${camPos.y.toFixed(2)},${camPos.z.toFixed(2)}`,
  );
  setStatus(`scanning (${SCAN_METHOD})…`);

  let data: ScanResult;
  try {
    const res = await fetch("/api/scan", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        image: frame.base64,
        frameId: String(Date.now()),
        aimX: 0.5,
        aimY: 0.5,
        method: SCAN_METHOD,
        fovDeg: frame.fovDeg,
      }),
    });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    data = (await res.json()) as ScanResult;
  } catch (e) {
    log(`scan failed: ${String(e)}`);
    setStatus("scan failed");
    return;
  }

  const aimed = data.aimed;
  if (!aimed?.contourCam?.length || !aimed.obbCam || !aimed.depthAtCentre) {
    setStatus("nothing scanned — aim at an object");
    return;
  }

  const s = distance / aimed.depthAtCentre;
  const toWorld = (p: Vec3) =>
    camPos
      .add(basis.right.scale(s * p[0]))
      .add(basis.up.scale(-s * p[1])) // server +y is down
      .add(basis.forward.scale(s * p[2]));
  const centroid = toWorld(aimed.obbCam.center);

  log(
    `SCAN ${aimed.name}: hitDist=${distance.toFixed(2)}m depthAtCentre=${aimed.depthAtCentre.toFixed(2)} ` +
      `scale=${s.toFixed(3)} contour=${aimed.contourCam.length}pts centroid=${centroid.x.toFixed(2)},${centroid.y.toFixed(2)},${centroid.z.toFixed(2)}`,
  );

  try {
    await anchorObject(aimed, centroid, toWorld, s);
    setStatus(
      `anchored <b>${aimed.name}</b><br>${data.detections.length} mask(s) · ${s.toFixed(2)}× scale`,
    );
  } catch (e) {
    log(`anchor failed: ${String(e)}`);
    setStatus("anchor failed");
  }
}

async function anchorObject(
  aimed: NonNullable<ScanResult["aimed"]>,
  centroid: Vector3,
  toWorld: (p: Vec3) => Vector3,
  scale: number,
) {
  if (!anchorSystem) return;
  const id = `scan-${anchorCount}`;
  const node = new TransformNode(id, scene);
  node.position.copyFrom(centroid);

  // The smooth outline, relative to the centroid.
  const outline = aimed.contourCam!.map((p) => toWorld(p).subtract(centroid));
  const line = MeshBuilder.CreateLines(`${id}-outline`, { points: outline }, scene);
  line.color = new Color3(1, 0.83, 0.31);
  line.isPickable = false;
  line.parent = node;

  // OBB wireframe.
  const obb = aimed.obbCam!;
  const R = quatToMat3(obb.quat);
  const half = obb.size.map((v) => v / 2) as Vec3;
  const corners: Vec3[] = [];
  for (const sx of [-1, 1])
    for (const sy of [-1, 1])
      for (const sz of [-1, 1])
        corners.push([sx * half[0], sy * half[1], sz * half[2]]);
  const worldCorners = corners.map((c) => {
    const r = mat3Apply(R, c);
    const camPoint: Vec3 = [
      obb.center[0] + r[0],
      obb.center[1] + r[1],
      obb.center[2] + r[2],
    ];
    return toWorld(camPoint).subtract(centroid);
  });
  const box = MeshBuilder.CreateLineSystem(
    `${id}-obb`,
    { lines: obbEdges().map(([a, b]) => [worldCorners[a]!, worldCorners[b]!]) },
    scene,
  );
  box.color = new Color3(0.49, 0.97, 0.77);
  box.isPickable = false;
  box.parent = node;

  const anchor = await anchorSystem.addAnchorAtPositionAndRotationAsync(
    centroid,
    Quaternion.Identity(),
  );
  anchor.attachedNode = node;
  placedAnchors.push(anchor);
  placedNodes.push(node);
  anchorCount++;

  gateway.sendAnchor(
    id,
    [centroid.x, centroid.y, centroid.z],
    [0, 0, 0, 1],
    aimed.obbCam!.size.map((v) => v * scale) as Vec3,
    outline.map((p) => [p.x, p.y, p.z] as Vec3),
    aimed.name,
  );
}

function clearAnchors() {
  for (const node of placedNodes) node.dispose(false, true);
  placedNodes.length = 0;
  for (const anchor of placedAnchors) anchor.remove();
  placedAnchors.length = 0;
  anchorCount = 0;
  gateway.sendAnchorClear();
  setStatus("cleared");
}

// --- pose streaming ---------------------------------------------------------
async function streamPose() {
  const cam = scene.activeCamera;
  if (!cam || !xr?.baseExperience) return;
  const now = performance.now();
  if (now - lastPoseSend < RATE) return;
  lastPoseSend = now;
  const scale = new Vector3();
  const quat = new Quaternion();
  const pos = new Vector3();
  cam.getWorldMatrix().decompose(scale, quat, pos);
  const forward = cam.getDirection(Vector3.Forward()).normalize();
  gateway.setPose(
    [pos.x, pos.y, pos.z],
    [quat.x, quat.y, quat.z, quat.w],
    [forward.x, forward.y, forward.z],
    "tracked",
  );
  frameCount++;
  if (rawHitSource) {
    const frame = xr?.baseExperience?.sessionManager.currentFrame;
    if (frame) {
      try {
        rawHitCount = frame.getHitTestResults?.(rawHitSource)?.length ?? 0;
      } catch {
        /* ignore */
      }
    }
  }
  if (frameCount % 120 === 0) {
    const ht = hitTest as unknown as {
      _xrHitTestSource?: unknown;
      _transientXrHitTestSource?: unknown;
    } | null;
    diag.info("rawHitCount", rawHitCount);
    log(
      `hb f=${frameCount} perm=${ht?._xrHitTestSource ? "y" : "n"} ` +
        `trans=${ht?._transientXrHitTestSource ? "y" : "n"} ` +
        `babylonHits=${lastHitCount} rawHits=${rawHitCount}`,
    );
  }
  if (frameCount % 30 === 0 && placedNodes.length === 0) {
    setStatus(
      `tracking · pos ${pos.x.toFixed(1)}, ${pos.y.toFixed(1)}, ${pos.z.toFixed(1)}` +
        `<br>hit-test: ${lastHitCount > 0 ? "LOCK (" + lastHitCount + ")" : "searching…"}` +
        `<br>keyframes: ${kfCount}${AUTO_KEYFRAMES ? "" : " (Key button)"}`,
    );
  }
}

scene.onBeforeRenderObservable.add(() => {
  updateObjectHold();
  if (pendingCapture) {
    const resolve = pendingCapture;
    pendingCapture = null;
    let frame: CaptureFrame | null = null;
    try {
      frame = captureNow();
    } catch (e) {
      log(`capture error: ${String(e)}`);
    }
    resolve(frame);
  }
  void streamPose();
  void captureKeyframe();
});

startBtn.addEventListener("click", () => void enterAr());
keyBtn.addEventListener("click", () => void captureKeyframe(true));
scanBtn.addEventListener("click", () => void scanAndAnchor());
clearBtn.addEventListener("click", () => clearAnchors());
reportBtn.addEventListener("click", () => {
  setStatus("sending report…");
  void diag
    .send("manual")
    .then((ok) => setStatus(ok ? "report sent ✓" : "report failed — check log"));
});
exitBtn.addEventListener("click", () => void xr?.baseExperience?.exitXRAsync());

let renderErrors = 0;
engine.runRenderLoop(() => {
  if (!scene.activeCamera) return; // XR rig not ready yet
  try {
    scene.render();
  } catch (e) {
    // A throw here would stop the engine scheduling frames, so swallow it.
    if (renderErrors++ < 3) diag.error("render", e);
  }
});
window.addEventListener("resize", () => engine.resize());

// --- boot -------------------------------------------------------------------
(async () => {
  try {
    const supported = await navigator.xr?.isSessionSupported("immersive-ar");
    if (!supported) {
      startBtn.disabled = true;
      setStatus("immersive-ar not supported on this browser/device.");
      return;
    }
    await setupXr();
    startBtn.disabled = false;
    setStatus("Ready. Tap Start.");
  } catch (e) {
    log(`boot error: ${String(e)}`);
    setStatus(`boot error: ${String(e)}`);
  }
})();
