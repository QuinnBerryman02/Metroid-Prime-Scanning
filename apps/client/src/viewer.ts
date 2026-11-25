/**
 * MS1 - desktop viewer.
 *
 * Connects to the gateway as a `viewer`, subscribes to relayed device poses and
 * renders the phone as a tracked marker in a shared world space. Later
 * milestones add the reconstructed mesh + scanned objects here.
 */
import { Engine } from "@babylonjs/core/Engines/engine";
import { Scene } from "@babylonjs/core/scene";
import { ArcRotateCamera } from "@babylonjs/core/Cameras/arcRotateCamera";
import { HemisphericLight } from "@babylonjs/core/Lights/hemisphericLight";
import { Vector3, Quaternion } from "@babylonjs/core/Maths/math.vector";
import { Color3, Color4 } from "@babylonjs/core/Maths/math.color";
import { MeshBuilder } from "@babylonjs/core/Meshes/meshBuilder";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { installDebugOverlay, log } from "./lib/debug";
import { GatewayClient } from "./lib/gateway";

installDebugOverlay();
log("viewer.ts loaded");

const sessionId = "viewer-" + Math.random().toString(36).slice(2, 7);

const canvas = document.getElementById("app") as HTMLCanvasElement;
const statusEl = document.getElementById("status")!;

const engine = new Engine(canvas, true);
const scene = new Scene(engine);
scene.clearColor = new Color4(0.02, 0.03, 0.04, 1);

const camera = new ArcRotateCamera(
  "cam",
  -Math.PI / 3,
  Math.PI / 3,
  4,
  new Vector3(0, 1, 0),
  scene,
);
camera.attachControl(canvas, true);
camera.wheelDeltaPercentage = 0.02;
new HemisphericLight("light", new Vector3(0.3, 1, 0.2), scene);

// Ground grid (metres).
const grid = MeshBuilder.CreateGround("grid", { width: 20, height: 20 }, scene);
grid.material = (() => {
  const m = new StandardMaterial("gridMat", scene);
  m.diffuseColor = new Color3(0.06, 0.1, 0.12);
  m.emissiveColor = new Color3(0.02, 0.05, 0.06);
  return m;
})();
grid.position.y = 0;

// Phone marker: a body box + a forward wedge, positioned by relayed pose.
const phone = MeshBuilder.CreateBox("phone", { width: 0.08, height: 0.16, depth: 0.02 }, scene);
phone.material = (() => {
  const m = new StandardMaterial("phoneMat", scene);
  m.emissiveColor = new Color3(0.49, 0.97, 0.77);
  m.disableLighting = true;
  return m;
})();

const forward = MeshBuilder.CreateCylinder(
  "fwd",
  { diameterTop: 0, diameterBottom: 0.06, height: 0.2, tessellation: 4 },
  scene,
);
forward.rotation.x = Math.PI / 2;
forward.position.z = 0.11;
forward.parent = phone;

const frustum = MeshBuilder.CreateLines(
  "frustum",
  {
    points: [
      new Vector3(-0.15, -0.1, 0.3),
      new Vector3(0.15, -0.1, 0.3),
      new Vector3(0.15, 0.1, 0.3),
      new Vector3(-0.15, 0.1, 0.3),
      new Vector3(-0.15, -0.1, 0.3),
      new Vector3(0, 0, 0),
      new Vector3(0.15, -0.1, 0.3),
      new Vector3(0, 0, 0),
      new Vector3(0.15, 0.1, 0.3),
      new Vector3(0, 0, 0),
      new Vector3(-0.15, 0.1, 0.3),
    ],
  },
  scene,
);
frustum.parent = phone;
frustum.color = new Color3(0.49, 0.97, 0.77);

phone.isVisible = false;
let lastSeen = 0;
let poseCount = 0;

const gateway = new GatewayClient("viewer", sessionId, {
  onStatus: (s) => {
    log(`ws ${s}`);
    if (s === "open") statusEl.textContent = "connected · waiting for device…";
    if (s === "closed") statusEl.textContent = "disconnected";
  },
  onPose: (msg) => {
    phone.isVisible = true;
    if (!phone.rotationQuaternion) phone.rotationQuaternion = Quaternion.Identity();
    phone.position.set(msg.position[0], msg.position[1], msg.position[2]);
    phone.rotationQuaternion.set(
      msg.quaternion[0],
      msg.quaternion[1],
      msg.quaternion[2],
      msg.quaternion[3],
    );
    lastSeen = performance.now();
    poseCount++;
  },
  onSession: (msg) => log(`session ${msg.status}: ${msg.sessionId}`),
  onWorld: (msg) => log(`world: ${msg.added.length} objects`),
});
gateway.connect();

setInterval(() => {
  if (phone.isVisible && performance.now() - lastSeen > 1500) {
    phone.isVisible = false;
    statusEl.textContent = "device offline";
  }
  if (phone.isVisible) {
    const p = phone.position;
    statusEl.textContent =
      `tracking · ${poseCount} poses\n` +
      `pos ${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)}`;
  }
}, 250);

engine.runRenderLoop(() => scene.render());
window.addEventListener("resize", () => engine.resize());
