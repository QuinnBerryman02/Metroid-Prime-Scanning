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
import { Quaternion, Vector3 } from "@babylonjs/core/Maths/math.vector";
import { Color3, Color4 } from "@babylonjs/core/Maths/math.color";
import { MeshBuilder } from "@babylonjs/core/Meshes/meshBuilder";
import { TransformNode } from "@babylonjs/core/Meshes/transformNode";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { WebXRDefaultExperience } from "@babylonjs/core/XR/webXRDefaultExperience";
import { WebXRFeatureName } from "@babylonjs/core/XR/webXRFeaturesManager";
import "@babylonjs/core/XR/features/WebXRHitTest";
import "@babylonjs/core/XR/features/WebXRAnchorSystem";
import "@babylonjs/core/XR/features/WebXRDOMOverlay";
import type { WebXRHitTest } from "@babylonjs/core/XR/features/WebXRHitTest";
import type { WebXRAnchorSystem } from "@babylonjs/core/XR/features/WebXRAnchorSystem";
import type { Camera } from "@babylonjs/core/Cameras/camera";
import type { Obb, Quat, ScanResult, Vec3 } from "@metroid-scan/shared";
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

// --- DOM --------------------------------------------------------------------
const canvas = document.getElementById("app") as HTMLCanvasElement;
const startOverlay = document.getElementById("start")!;
const startBtn = document.getElementById("startBtn") as HTMLButtonElement;
const hud = document.getElementById("hud")!;
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
});
gateway.connect();

// --- Babylon ----------------------------------------------------------------
const engine = new Engine(canvas, true, { stencil: true });
const scene = new Scene(engine);
scene.clearColor = new Color4(0, 0, 0, 0);

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
let currentHit: { position: Vector3; quaternion: Quaternion } | null = null;
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
}

let pendingCapture: ((f: CaptureFrame | null) => void) | null = null;

function grabFrame(): Promise<CaptureFrame | null> {
  return new Promise((resolve) => {
    pendingCapture = resolve;
    window.setTimeout(() => {
      if (pendingCapture === resolve) {
        pendingCapture = null;
        resolve(null);
      }
    }, 1500);
  });
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
      getCameraImage?: (view: XRView) => WebGLTexture | null;
    };
    diag.info("capture.getCameraImageType", typeof binding.getCameraImage);
    if (typeof binding.getCameraImage !== "function") {
      diag.error("capture", new TypeError("getCameraImage is not a function"));
      return null;
    }
    texture = binding.getCameraImage(view);
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
    const url = dst.toDataURL("image/jpeg", 0.8);

    return {
      base64: url.slice(url.indexOf(",") + 1),
      width: outW,
      height: outH,
      fovDeg: viewFov(view),
    };
  } catch (e) {
    diag.error("capture.encode", e);
    return null;
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

  log(`hit-test: ${hitTest ? "on" : "OFF"} | anchors: ${anchorSystem ? "on" : "OFF"}`);

  hitTest?.onHitTestResultObservable.add((results) => {
    const r = results[0];
    if (r) {
      reticle.isVisible = true;
      reticle.position.copyFrom(r.position);
      reticle.rotationQuaternion!.copyFrom(r.rotationQuaternion);
      currentHit = {
        position: r.position.clone(),
        quaternion: r.rotationQuaternion.clone(),
      };
      scanBtn.disabled = false;
    } else {
      reticle.isVisible = false;
      currentHit = null;
      scanBtn.disabled = true;
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
    const session = xr.baseExperience.sessionManager.session;
    const enabled =
      (session as unknown as { enabledFeatures?: string[] }).enabledFeatures ?? [];
    diag.info("session.enabledFeatures", enabled);
    log(`features: ${enabled.join(", ") || "(none reported)"}`);
    diag.info("session.blendMode", session.environmentBlendMode ?? null);
    session.addEventListener("end", () => {
      hud.style.display = "none";
      startOverlay.style.display = "flex";
      log("session ended");
      void diag.send("session-end");
    });
    log("entered immersive-ar");
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
  if (!currentHit) {
    setStatus("no surface under the reticle");
    return;
  }
  const camera = scene.activeCamera;
  if (!camera) return;

  const hitPos = currentHit.position.clone();
  const camPos = camera.globalPosition.clone();
  const distance = Vector3.Distance(hitPos, camPos);
  const basis = cameraBasis(camera);

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
  if (frameCount % 30 === 0 && placedNodes.length === 0) {
    setStatus(
      `tracking · pos ${pos.x.toFixed(1)}, ${pos.y.toFixed(1)}, ${pos.z.toFixed(1)}`,
    );
  }
}

scene.onBeforeRenderObservable.add(() => {
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
});

startBtn.addEventListener("click", () => void enterAr());
scanBtn.addEventListener("click", () => void scanAndAnchor());
clearBtn.addEventListener("click", () => clearAnchors());
reportBtn.addEventListener("click", () => {
  setStatus("sending report…");
  void diag
    .send("manual")
    .then((ok) => setStatus(ok ? "report sent ✓" : "report failed — check log"));
});
exitBtn.addEventListener("click", () => void xr?.baseExperience?.exitXRAsync());

engine.runRenderLoop(() => scene.render());
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
