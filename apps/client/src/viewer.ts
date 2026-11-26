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
import { Mesh } from "@babylonjs/core/Meshes/mesh";
import { VertexData } from "@babylonjs/core/Meshes/mesh.vertexData";
import { TransformNode } from "@babylonjs/core/Meshes/transformNode";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import type { PlaneMsg, ReconCloudMsg, ReconKeyframeMsg } from "@metroid-scan/shared";
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

// Anchored boxes streamed from the phone.
const anchorMat = new StandardMaterial("anchorMat", scene);
anchorMat.emissiveColor = new Color3(0.2, 0.7, 1);
anchorMat.alpha = 0.85;
anchorMat.disableLighting = true;
const anchorMeshes = new Map<string, ReturnType<typeof MeshBuilder.CreateBox>>();
const contourMeshes = new Map<string, ReturnType<typeof MeshBuilder.CreateLines>>();

// --- ARCore planes ----------------------------------------------------------
// ARCore's free environment model: metric polygons for floor / walls / tables,
// coloured by orientation. Press "P" to toggle.
const PLANE_COLORS: Record<string, Color3> = {
  horizontal: new Color3(0.25, 0.72, 0.68), // teal - floor / table
  vertical: new Color3(1.0, 0.72, 0.3), // amber - wall
  unknown: new Color3(0.6, 0.62, 0.66),
};

interface PlaneVis {
  fill: Mesh;
  outline: ReturnType<typeof MeshBuilder.CreateLines>;
  normal: ReturnType<typeof MeshBuilder.CreateLines>;
}
const planeMeshes = new Map<number, PlaneVis>();
let planesVisible = true;

function applyPlaneVisibility() {
  for (const v of planeMeshes.values()) {
    v.fill.isVisible = planesVisible;
    v.outline.isVisible = planesVisible;
    v.normal.isVisible = planesVisible;
  }
}

function disposePlane(id: number) {
  const v = planeMeshes.get(id);
  if (!v) return;
  v.fill.dispose();
  v.outline.dispose();
  v.normal.dispose();
  planeMeshes.delete(id);
}

function renderPlane(msg: PlaneMsg) {
  if (msg.action === "removed" || msg.polygon.length < 3) {
    disposePlane(msg.id);
    return;
  }
  disposePlane(msg.id); // rebuild: planes grow/merge in place

  const color = PLANE_COLORS[msg.orientation] ?? PLANE_COLORS.unknown!;
  const pts = msg.polygon.map((p) => new Vector3(p[0], p[1], p[2]));
  const center = new Vector3(msg.center[0], msg.center[1], msg.center[2]);

  // Filled polygon: fan from the centroid (fine for the near-convex ARCore
  // polygons, and avoids pulling in an earcut triangulator).
  const positions: number[] = [...msg.center];
  for (const p of msg.polygon) positions.push(p[0], p[1], p[2]);
  const indices: number[] = [];
  for (let i = 0; i < pts.length; i++) indices.push(0, 1 + i, 1 + ((i + 1) % pts.length));
  const normals: number[] = [];
  VertexData.ComputeNormals(positions, indices, normals);
  const vd = new VertexData();
  vd.positions = positions;
  vd.indices = indices;
  vd.normals = normals;

  const fill = new Mesh(`plane-${msg.id}`, scene);
  vd.applyToMesh(fill);
  const mat = new StandardMaterial(`planeMat-${msg.id}`, scene);
  mat.emissiveColor = color.scale(0.5);
  mat.diffuseColor = color.scale(0.2);
  mat.alpha = 0.22;
  mat.backFaceCulling = false;
  mat.disableLighting = true;
  fill.material = mat;
  fill.isPickable = false;

  const outline = MeshBuilder.CreateLines(
    `plane-outline-${msg.id}`,
    { points: [...pts, pts[0]!] },
    scene,
  );
  outline.color = color;
  outline.isPickable = false;

  const n = new Vector3(msg.normal[0], msg.normal[1], msg.normal[2]);
  const normal = MeshBuilder.CreateLines(
    `plane-normal-${msg.id}`,
    { points: [center, center.add(n.scale(0.25))] },
    scene,
  );
  normal.color = color;
  normal.isPickable = false;

  planeMeshes.set(msg.id, { fill, outline, normal });
  applyPlaneVisibility();
}

window.addEventListener("keydown", (e) => {
  if (e.key === "p" || e.key === "P") {
    planesVisible = !planesVisible;
    applyPlaneVisibility();
    log(`planes ${planesVisible ? "on" : "off"}`);
  }
});

// --- recon keyframes (MS3.0) ------------------------------------------------
// One small wireframe frustum per streamed keyframe: a breadcrumb trail of
// where the phone has been, so reconstruction coverage is visible at a glance.
const KEYFRAME_MAX = 120;
const keyframeMarkers: TransformNode[] = [];
let reconKeyframes = 0;
let reconFused = 0;
let reconSkipped = 0;
let reconNote: string | null = null;
let lastLoggedNote = "";

/** A tiny line-list frustum, in the camera's local frame (+z forward). */
function keyframeFrustum(fovDeg: number, aspect: number, depth = 0.22): Vector3[][] {
  const halfH = depth * Math.tan(((fovDeg * Math.PI) / 180) / 2);
  const halfW = halfH * aspect;
  const o = new Vector3(0, 0, 0);
  const c = [
    new Vector3(-halfW, -halfH, depth),
    new Vector3(halfW, -halfH, depth),
    new Vector3(halfW, halfH, depth),
    new Vector3(-halfW, halfH, depth),
  ];
  return [
    [o, c[0]!],
    [o, c[1]!],
    [o, c[2]!],
    [o, c[3]!],
    [c[0]!, c[1]!],
    [c[1]!, c[2]!],
    [c[2]!, c[3]!],
    [c[3]!, c[0]!],
  ];
}

function addKeyframe(msg: ReconKeyframeMsg) {
  const node = new TransformNode(`kf-${msg.frameId}`, scene);
  node.position.set(msg.position[0], msg.position[1], msg.position[2]);
  node.rotationQuaternion = new Quaternion(
    msg.quaternion[0],
    msg.quaternion[1],
    msg.quaternion[2],
    msg.quaternion[3],
  );
  const lines = MeshBuilder.CreateLineSystem(
    `kf-${msg.frameId}-lines`,
    { lines: keyframeFrustum(msg.fovDeg, msg.width / Math.max(1, msg.height)) },
    scene,
  );
  lines.color = new Color3(0.62, 0.78, 0.86);
  lines.alpha = 0.55;
  lines.isPickable = false;
  lines.parent = node;
  keyframeMarkers.push(node);
  while (keyframeMarkers.length > KEYFRAME_MAX) {
    keyframeMarkers.shift()!.dispose(false, true);
  }
}

function clearKeyframes() {
  for (const n of keyframeMarkers) n.dispose(false, true);
  keyframeMarkers.length = 0;
  reconKeyframes = 0;
}

// --- fused point cloud (MS3.2) ----------------------------------------------
// One mesh, rebuilt per snapshot. `pointsCloud` draws the vertices as discs,
// the cheapest way to show ~30k points in Babylon. The material is created
// once so we do not recompile its shader on every snapshot.
const cloudMaterial = (() => {
  const m = new StandardMaterial("cloudMat", scene);
  m.pointsCloud = true;
  m.pointSize = 3;
  // Unlit: the cloud carries its own colour, so lighting would only darken it.
  m.disableLighting = true;
  m.diffuseColor = new Color3(1, 1, 1);
  m.emissiveColor = new Color3(0, 0, 0);
  return m;
})();

let cloudMesh: Mesh | null = null;
let cloudPoints = 0;
let cloudVoxels = 0;
let cloudScale = "";

function decodeBase64(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function disposeCloud() {
  cloudMesh?.dispose();
  cloudMesh = null;
  cloudPoints = 0;
  cloudVoxels = 0;
  cloudScale = "";
}

function renderCloud(msg: ReconCloudMsg) {
  if (!msg.count) {
    disposeCloud();
    return;
  }
  const posBytes = decodeBase64(msg.positions);
  const colBytes = decodeBase64(msg.colors);
  // little-endian float32 xyz triples, per the protocol
  const positions = new Float32Array(posBytes.buffer, posBytes.byteOffset, msg.count * 3);
  const colors = new Float32Array(msg.count * 4);
  for (let i = 0; i < msg.count; i++) {
    colors[i * 4] = (colBytes[i * 3] ?? 0) / 255;
    colors[i * 4 + 1] = (colBytes[i * 3 + 1] ?? 0) / 255;
    colors[i * 4 + 2] = (colBytes[i * 3 + 2] ?? 0) / 255;
    colors[i * 4 + 3] = 1;
  }

  const mesh = new Mesh("cloud", scene);
  const vd = new VertexData();
  vd.positions = new Float32Array(positions); // copy: the source buffer is reused
  vd.colors = colors;
  vd.applyToMesh(mesh);
  mesh.material = cloudMaterial;
  mesh.useVertexColors = true;
  mesh.isPickable = false;

  cloudMesh?.dispose();
  cloudMesh = mesh;
  cloudPoints = msg.count;
  cloudVoxels = msg.voxels;
  cloudScale = `${msg.scaleMethod} a=${msg.scaleA.toFixed(3)} b=${msg.scaleB.toFixed(3)}`;
}

const gateway = new GatewayClient("viewer", sessionId, {
  onStatus: (s) => {
    log(`ws ${s}`);
    if (s === "open") statusEl.textContent = "connected · waiting for device…";
    if (s === "open") log("viewer connected");
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
  onPlane: (msg) => renderPlane(msg),
  onReconKeyframe: (msg) => addKeyframe(msg),
  onReconStatus: (msg) => {
    reconKeyframes = msg.keyframes;
    reconFused = msg.fused ?? reconFused;
    reconSkipped = msg.skipped ?? reconSkipped;
    reconNote = msg.note ?? null;
    // Only log when it changes - this fires on every keyframe.
    if ((msg.note ?? "") !== lastLoggedNote) {
      lastLoggedNote = msg.note ?? "";
      if (msg.note) log(`recon NOT FUSING: ${msg.note}`);
      else log("recon fusing");
    }
  },
  onReconCloud: (msg) => renderCloud(msg),
  onReconClear: () => {
    clearKeyframes();
    disposeCloud();
  },
  onAnchor: (msg) => {
    const [sx, sy, sz] = msg.size;
    let mesh = anchorMeshes.get(msg.id);
    if (!mesh) {
      mesh = MeshBuilder.CreateBox(msg.id, { width: sx, height: sy, depth: sz }, scene);
      mesh.material = anchorMat;
      anchorMeshes.set(msg.id, mesh);
      log(`anchor +${msg.id}`);
    }
    mesh.position.set(...msg.position);
    if (!mesh.rotationQuaternion) mesh.rotationQuaternion = Quaternion.Identity();
    mesh.rotationQuaternion.set(...msg.quaternion);

    // World-space outline (relative to the anchor position).
    contourMeshes.get(msg.id)?.dispose();
    if (msg.contour && msg.contour.length > 1) {
      const points = msg.contour.map(
        (p) =>
          new Vector3(
            msg.position[0] + p[0],
            msg.position[1] + p[1],
            msg.position[2] + p[2],
          ),
      );
      const line = MeshBuilder.CreateLines(`${msg.id}-outline`, { points }, scene);
      line.color = new Color3(1, 0.83, 0.31);
      contourMeshes.set(msg.id, line);
      log(`  outline ${msg.contour.length}pts (${msg.label ?? "?"})`);
    }
  },
  onAnchorClear: (msg) => {
    if (msg.sessionId) {
      for (const m of anchorMeshes.values()) m.dispose();
      anchorMeshes.clear();
      for (const m of contourMeshes.values()) m.dispose();
      contourMeshes.clear();
      log("anchors cleared");
    }
  },
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
      (reconNote ? `\u26a0 NOT FUSING: ${reconNote}\n` : "") +
      `tracking \u00b7 ${poseCount} poses\n` +
      `pos ${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)}\n` +
      `${planeMeshes.size} plane(s)${planesVisible ? "" : " (hidden, P)"}\n` +
      `${reconKeyframes} keyframe(s) \u00b7 ${reconFused} fused \u00b7 ${reconSkipped} skipped\n` +
      `${cloudVoxels} voxels -> ${cloudPoints} pts\n` +
      `${cloudScale}\n` +
      `${anchorMeshes.size} anchor(s)`;
  }
}, 250);

engine.runRenderLoop(() => scene.render());
window.addEventListener("resize", () => engine.resize());
