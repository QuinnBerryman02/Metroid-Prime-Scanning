/**
 * MS1 - AR scan visor skeleton.
 *
 * - Babylon WebXR `immersive-ar` session
 * - hit-test reticle that tracks real surfaces
 * - "Place" creates a WebXR anchor holding a box (proves anchoring)
 * - streams the camera pose to the gateway (visible in the desktop viewer)
 *
 * Deliberately minimal: no inference, no HUD cards yet (those are MS2-MS4).
 */
import { Engine } from "@babylonjs/core/Engines/engine";
import { Scene } from "@babylonjs/core/scene";
import { Vector3, Quaternion } from "@babylonjs/core/Maths/math.vector";
import { Color3, Color4 } from "@babylonjs/core/Maths/math.color";
import { MeshBuilder } from "@babylonjs/core/Meshes/meshBuilder";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { WebXRDefaultExperience } from "@babylonjs/core/XR/webXRDefaultExperience";
import { WebXRFeatureName } from "@babylonjs/core/XR/webXRFeaturesManager";
// Side-effect imports register each feature with the features manager.
import "@babylonjs/core/XR/features/WebXRHitTest";
import "@babylonjs/core/XR/features/WebXRAnchorSystem";
import "@babylonjs/core/XR/features/WebXRDOMOverlay";
import type { WebXRHitTest } from "@babylonjs/core/XR/features/WebXRHitTest";
import type { WebXRAnchorSystem } from "@babylonjs/core/XR/features/WebXRAnchorSystem";
import { installDebugOverlay, log } from "./lib/debug";
import { GatewayClient } from "./lib/gateway";

installDebugOverlay();
log("ar.ts loaded");

const sessionId = crypto.randomUUID();
const RATE = 66; // ms between streamed poses

// --- DOM --------------------------------------------------------------------

const canvas = document.getElementById("app") as HTMLCanvasElement;
const startOverlay = document.getElementById("start")!;
const startBtn = document.getElementById("startBtn") as HTMLButtonElement;
const hud = document.getElementById("hud")!;
const placeBtn = document.getElementById("placeBtn") as HTMLButtonElement;
const clearBtn = document.getElementById("clearBtn") as HTMLButtonElement;
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
scene.clearColor = new Color4(0, 0, 0, 0); // transparent -> camera passthrough

const reticleMat = new StandardMaterial("reticleMat", scene);
reticleMat.emissiveColor = new Color3(0.49, 0.97, 0.77);
reticleMat.disableLighting = true;
reticleMat.alpha = 0.9;

const boxMat = new StandardMaterial("boxMat", scene);
boxMat.emissiveColor = new Color3(0.2, 0.7, 1);
boxMat.alpha = 0.85;
boxMat.disableLighting = true;

const reticle = MeshBuilder.CreateTorus(
  "reticle",
  { diameter: 0.16, thickness: 0.02, tessellation: 32 },
  scene,
);
reticle.rotationQuaternion = Quaternion.Identity();
reticle.material = reticleMat;
reticle.isVisible = false;

const placed: import("@babylonjs/core/Meshes/mesh").Mesh[] = [];

let xr: WebXRDefaultExperience | null = null;
let hitTest: WebXRHitTest | null = null;
let anchorSystem: WebXRAnchorSystem | null = null;
let currentHit: { position: Vector3; quaternion: Quaternion } | null = null;
let frameCount = 0;
let lastPoseSend = 0;

async function setupXr() {
  log("creating default XR experience…");
  xr = await WebXRDefaultExperience.CreateAsync(scene, {
    disableDefaultUI: true,
    disableTeleportation: true,
    disablePointerSelection: true,
    disableNearInteraction: true,
    disableHandTracking: true,
  });
  if (!xr || !xr.baseExperience) {
    throw new Error("WebXR base experience unavailable");
  }
  const fm = xr.baseExperience.featuresManager;

  // HUD shown over the camera passthrough.
  fm.enableFeature(WebXRFeatureName.DOM_OVERLAY, "latest", {
    element: hud,
    supressXRSelectEvents: true,
  });
  hitTest = fm.enableFeature(WebXRFeatureName.HIT_TEST, "latest", {
    disablePermanentHitTest: false,
    enableTransientHitTest: true,
  }) as WebXRHitTest | null;
  anchorSystem = fm.enableFeature(
    WebXRFeatureName.ANCHOR_SYSTEM,
    "latest",
    { doNotRemoveAnchorsOnSessionEnded: false },
  ) as WebXRAnchorSystem | null;

  log(`hit-test feature: ${hitTest ? "on" : "OFF"}`);
  log(`anchor feature: ${anchorSystem ? "on" : "OFF"}`);

  hitTest?.onHitTestResultObservable.add((results) => {
    if (results.length > 0) {
      const r = results[0]!;
      reticle.isVisible = true;
      reticle.position.copyFrom(r.position);
      reticle.rotationQuaternion!.copyFrom(r.rotationQuaternion);
      currentHit = {
        position: r.position.clone(),
        quaternion: r.rotationQuaternion.clone(),
      };
      placeBtn.disabled = false;
    } else {
      reticle.isVisible = false;
      currentHit = null;
      placeBtn.disabled = true;
    }
  });
}

async function enterAr() {
  if (!xr?.baseExperience) return;
  try {
    await xr.baseExperience.enterXRAsync(
      "immersive-ar",
      "local-floor",
      xr.renderTarget,
      {
        requiredFeatures: ["hit-test", "anchors"],
        optionalFeatures: ["plane-detection", "light-estimation", "dom-overlay"],
      },
    );
    startOverlay.style.display = "none";
    hud.style.display = "block";
    const session = xr.baseExperience.sessionManager.session;
    session.addEventListener("select", () => void place());
    session.addEventListener("end", () => {
      hud.style.display = "none";
      startOverlay.style.display = "flex";
      log("session ended");
    });
    log("entered immersive-ar");
  } catch (e) {
    log(`enterXRAsync FAILED: ${String(e)}`);
    setStatus(`<b style="color:#ff6b6b">AR failed</b><br>${String(e)}`);
  }
}

async function place() {
  if (!currentHit || !anchorSystem) {
    log("place: no hit");
    return;
  }
  const pos = currentHit.position.clone();
  const quat = currentHit.quaternion.clone();
  const SIZE = 0.12;
  const id = `box-${placed.length}`;
  try {
    const box = MeshBuilder.CreateBox(id, { size: SIZE }, scene);
    box.material = boxMat;
    box.position.copyFrom(pos);
    const anchor = await anchorSystem.addAnchorAtPositionAndRotationAsync(pos, quat);
    anchor.attachedNode = box;
    placed.push(box);
    gateway.sendAnchor(
      id,
      [pos.x, pos.y, pos.z],
      [quat.x, quat.y, quat.z, quat.w],
      [SIZE, SIZE, SIZE],
    );
    log(`anchor #${placed.length} placed @ ${pos.x.toFixed(2)},${pos.y.toFixed(2)},${pos.z.toFixed(2)}`);
  } catch (e) {
    log(`anchor failed: ${String(e)}`);
  }
}

function clearBoxes() {
  for (const b of placed) b.dispose();
  placed.length = 0;
  anchorSystem?.anchors.forEach((a) => a.remove());
  gateway.sendAnchorClear();
  log("cleared");
}

async function streamPose() {
  const cam = scene.activeCamera;
  if (!cam) return;
  const now = performance.now();
  if (now - lastPoseSend < RATE) return;
  lastPoseSend = now;
  const world = cam.getWorldMatrix();
  const scale = new Vector3();
  const quat = new Quaternion();
  const pos = new Vector3();
  world.decompose(scale, quat, pos);
  const forward = new Vector3(0, 0, 1).applyRotationQuaternion(quat).normalize();
  gateway.setPose(
    [pos.x, pos.y, pos.z],
    [quat.x, quat.y, quat.z, quat.w],
    [forward.x, forward.y, forward.z],
    "tracked",
  );
  frameCount++;
  if (frameCount % 15 === 0) {
    setStatus(
      `tracking · ${frameCount} frames<br>` +
        `pos ${pos.x.toFixed(1)}, ${pos.y.toFixed(1)}, ${pos.z.toFixed(1)}<br>` +
        `${placed.length} anchor(s)`,
    );
  }
}

scene.onBeforeRenderObservable.add(() => {
  void streamPose();
});

startBtn.addEventListener("click", () => void enterAr());
placeBtn.addEventListener("click", () => void place());
clearBtn.addEventListener("click", () => clearBoxes());
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
    setStatus("Ready. Tap Start to scan.");
  } catch (e) {
    log(`boot error: ${String(e)}`);
    setStatus(`boot error: ${String(e)}`);
  }
})();
