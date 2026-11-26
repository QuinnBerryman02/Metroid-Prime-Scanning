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
const statusEl = document.getElementById("status") as HTMLDetailsElement;
const statusHead = document.getElementById("statusHead")!;
const statusBody = document.getElementById("statusBody")!;

/**
 * The status box is a <details>: one line of the numbers you actually watch,
 * and the full breakdown behind a click. It used to be ten lines of always-on
 * text covering the scene.
 */
function setStatus(head: string, body = "") {
  statusHead.textContent = head;
  statusBody.textContent = body;
  statusBody.hidden = !body;
}

const engine = new Engine(canvas, true);
const scene = new Scene(engine);
scene.clearColor = new Color4(0.02, 0.03, 0.04, 1);

/**
 * Background themes. The cloud is painted with the camera's own colours, so a
 * dark scene hides near-black points and a light one hides near-white ones -
 * hence a mid grey in the middle, and `B` to cycle. viewer.html owns the
 * panel colours via CSS variables under the matching body class.
 */
const BACKGROUNDS = {
  dark: [0.02, 0.03, 0.04],
  grey: [0.235, 0.259, 0.294],
  light: [0.91, 0.925, 0.941],
} as const;
type BackgroundId = keyof typeof BACKGROUNDS;
const BACKGROUND_ORDER: BackgroundId[] = ["dark", "grey", "light"];

function setBackground(id: BackgroundId): BackgroundId {
  const c = BACKGROUNDS[id];
  scene.clearColor = new Color4(c[0], c[1], c[2], 1);
  // The panel palette hangs off this class (see viewer.html).
  document.body.className = `theme-${id}`;
  try {
    localStorage.setItem("viewer.background", id);
  } catch {
    /* ignore */
  }
  return id;
}

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

// Navigation: the mouse wheel already zooms, but be explicit about keys too.
window.addEventListener("keydown", (e) => {
  const step = 1.25;
  switch (e.key) {
    case "+":
    case "=":
      camera.radius = Math.max(0.15, camera.radius / step);
      break;
    case "-":
    case "_":
      camera.radius = Math.min(500, camera.radius * step);
      break;
    case "ArrowLeft":
      camera.alpha -= 0.12;
      break;
    case "ArrowRight":
      camera.alpha += 0.12;
      break;
    case "ArrowUp":
      camera.beta = Math.max(0.02, camera.beta - 0.08);
      break;
    case "ArrowDown":
      camera.beta = Math.min(Math.PI - 0.02, camera.beta + 0.08);
      break;
    default:
      return;
  }
  e.preventDefault();
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
let reconTiming: Record<string, number> | null = null;
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
  cloudScale =
    `${msg.method ?? "?"}${msg.note ? ` (${msg.note})` : ""} · ` +
    `${msg.scaleMethod} a=${msg.scaleA.toFixed(3)} b=${msg.scaleB.toFixed(3)}`;
}

const gateway = new GatewayClient("viewer", sessionId, {
  onStatus: (s) => {
    log(`ws ${s}`);
    if (s === "open") setStatus("connected · waiting for device…");
    if (s === "open") log("viewer connected");
    if (s === "closed") setStatus("disconnected");
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
    reconTiming = msg.timing ?? null;
    // Only log when it changes - this fires on every keyframe.
    if ((msg.note ?? "") !== lastLoggedNote) {
      lastLoggedNote = msg.note ?? "";
      if (msg.note) log(`recon NOT FUSING: ${msg.note}`);
      else log("recon fusing");
    }
  },
  onReconCloud: (msg) => {
    replayClouds += 1;
    renderCloud(msg);
  },
  onReconReplay: (msg) => {
    replayProgress.hidden = false;
    replayBar.max = Math.max(1, msg.total);
    replayBar.value = msg.index;
    if (msg.state === "running") {
      if (msg.index === 0) replayClouds = 0;
      // A single-frame run says which frame, since that is the whole point of
      // picking a range - otherwise "1/1" tells you nothing about frame 7.
      const where =
        msg.total === 1
          ? `frame ${msg.label ?? msg.from}`
          : `${msg.index}/${msg.total}`;
      replayProgressText.textContent = `replaying ${msg.source.slice(0, 8)}\u2026 ${where}`;
      window.clearTimeout(replayHideTimer);
      return;
    }

    // Finished. Say so out loud if no geometry came back - a replay that ran
    // to completion and drew nothing is exactly the confusing case, and the
    // reason is almost always in the fusion note.
    replayProgressText.textContent =
      `${msg.state} \u00b7 ${msg.total} frame${msg.total === 1 ? "" : "s"}` +
      (replayClouds === 0 ? " \u00b7 no cloud returned" : ` \u00b7 ${replayClouds} cloud(s)`);
    if (replayClouds === 0) {
      log(
        `replay ${msg.state} with NO cloud returned for ${msg.total} frame(s) - ` +
          (reconNote ? `fusion says: ${reconNote}` : "check the log for NOT FUSING"),
      );
    }
    window.clearTimeout(replayHideTimer);
    replayHideTimer = window.setTimeout(() => {
      replayProgress.hidden = true;
    }, 6000);
  },
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
    setStatus("device offline", statusBody.textContent ?? "");
  }
  if (phone.isVisible) {
    const p = phone.position;
    setStatus(
      // The headline is what you watch while a replay runs.
      `${reconKeyframes} kf \u00b7 ${reconFused} fused \u00b7 ${cloudPoints} pts \u00b7 ` +
        `${cloudScale}` +
        (reconNote ? " \u00b7 \u26a0" : ""),
      [
        reconNote ? `\u26a0 ${reconNote}` : "",
        `tracking \u00b7 ${poseCount} poses`,
        `pos ${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)}`,
        `${planeMeshes.size} plane(s)${planesVisible ? "" : " (hidden, P)"}`,
        `${reconKeyframes} keyframe(s) \u00b7 ${reconFused} fused \u00b7 ${reconSkipped} skipped`,
        `${cloudVoxels} voxels -> ${cloudPoints} pts`,
        cloudScale,
        reconTiming
          ? `depth ${(reconTiming.depth ?? 0).toFixed(0)}ms \u00b7 ` +
            `scale ${(reconTiming.scale ?? 0).toFixed(0)}ms \u00b7 ` +
            `fuse ${(reconTiming.fuse ?? 0).toFixed(0)}ms \u00b7 ` +
            `total ${(reconTiming.total ?? 0).toFixed(0)}ms`
          : "",
        `${anchorMeshes.size} anchor(s)`,
        `wheel / +/- zoom \u00b7 arrows orbit \u00b7 P planes \u00b7 B background \u00b7 A algorithms`,
      ]
        .filter(Boolean)
        .join("\n"),
    );
  }
}, 250);

// --- offline replay ---------------------------------------------------------
// Recorded keyframe runs (written by the inference service) can be replayed
// through the live gateway, so the viewer sees them exactly as if a phone were
// walking the room. This is the fast tuning loop: one recording, many settings.
const logSelect = document.getElementById("logSelect") as HTMLSelectElement;
const replayBtn = document.getElementById("replayBtn") as HTMLButtonElement;
const stopBtn = document.getElementById("stopBtn") as HTMLButtonElement;
const replayHint = document.getElementById("replayHint")!;
const fromFrameInput = document.getElementById("fromFrame") as HTMLInputElement;
const toFrameInput = document.getElementById("toFrame") as HTMLInputElement;
const replayProgress = document.getElementById("replayProgress")!;
const replayProgressText = document.getElementById("replayProgressText")!;
const replayBar = document.getElementById("replayBar") as HTMLProgressElement;
let replayHideTimer = 0;
/** Clouds received during the current replay, so "nothing appeared" can say so. */
let replayClouds = 0;
const methodSelect = document.getElementById("methodSelect") as HTMLSelectElement;
const paramsBox = document.getElementById("params")!;
const methodDesc = document.getElementById("methodDesc")!;
const resetParams = document.getElementById("resetParams") as HTMLButtonElement;
const algoToggle = document.getElementById("algoToggle") as HTMLButtonElement;
const algoBox = document.getElementById("algoBox")!;
const algoSummary = document.getElementById("algoSummary")!;
const ALGO_KEY = "viewer.algoOpen";

interface ParamSpec {
  key: string;
  label: string;
  kind: "float" | "int" | "bool" | "enum";
  default: number | string | boolean;
  min: number | null;
  max: number | null;
  step: number | null;
  choices: string[] | null;
  help: string;
}
interface MethodSpec {
  id: string;
  label: string;
  description: string;
  params: ParamSpec[];
}
let methods: MethodSpec[] = [];

function currentMethod(): MethodSpec | undefined {
  return methods.find((m) => m.id === methodSelect.value);
}

/**
 * The parameter grid is tall and the scene is the point, so it starts folded
 * away. The one-line summary still says what a Replay would actually run.
 */
function setAlgoOpen(open: boolean) {
  algoBox.hidden = !open;
  algoToggle.innerHTML = open ? "&#9662; algorithms" : "&#9656; algorithms";
  try {
    localStorage.setItem(ALGO_KEY, open ? "1" : "0");
  } catch {
    /* private mode - not worth caring about */
  }
}

function updateAlgoSummary() {
  const m = currentMethod();
  if (!m) {
    algoSummary.textContent = "";
    return;
  }
  const p = collectParams();
  const scale =
    p.scaleMode === "fixed" ? `a=${Number(p.scaleA).toFixed(2)}` : String(p.scaleMode);
  algoSummary.textContent = `${m.id} \u00b7 ${scale}`;
}

/**
 * Rebuild the parameter controls from the selected method's declaration. The
 * server owns the schema (app/recon_methods.py), so nothing here needs to know
 * what a parameter means - or to be edited when a new one is added.
 */
function renderParams() {
  const m = currentMethod();
  paramsBox.innerHTML = "";
  methodDesc.textContent = m?.description ?? "";
  if (!m) return;
  for (const p of m.params) {
    const label = document.createElement("label");
    label.textContent = p.label;
    paramsBox.appendChild(label);

    let input: HTMLInputElement | HTMLSelectElement;
    if (p.kind === "enum") {
      input = document.createElement("select");
      for (const c of p.choices ?? []) {
        const o = document.createElement("option");
        o.value = c;
        o.textContent = c;
        input.appendChild(o);
      }
      (input as HTMLSelectElement).value = String(p.default);
    } else if (p.kind === "bool") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.checked = Boolean(p.default);
    } else {
      input = document.createElement("input");
      input.type = "number";
      input.value = String(p.default);
      if (p.min !== null) input.min = String(p.min);
      if (p.max !== null) input.max = String(p.max);
      if (p.step !== null) input.step = String(p.step);
    }
    input.dataset.key = p.key;
    input.title = p.help;
    paramsBox.appendChild(input);

    if (p.help) {
      const h = document.createElement("div");
      h.className = "phelp";
      h.textContent = p.help;
      paramsBox.appendChild(h);
    }
  }
}

/** Read the controls back out. The server clamps, so no validation here. */
function collectParams(): Record<string, number | string | boolean> {
  const out: Record<string, number | string | boolean> = {};
  for (const el of paramsBox.querySelectorAll<HTMLElement>("[data-key]")) {
    const key = el.dataset.key!;
    if (el instanceof HTMLInputElement && el.type === "checkbox") {
      out[key] = el.checked;
    } else if (el instanceof HTMLSelectElement) {
      out[key] = el.value;
    } else {
      const n = Number((el as HTMLInputElement).value);
      if (Number.isFinite(n)) out[key] = n;
    }
  }
  return out;
}

async function loadMethods() {
  try {
    const res = await fetch("/api/recon/methods");
    const data = (await res.json()) as { methods?: MethodSpec[]; default?: string };
    methods = data.methods ?? [];
    methodSelect.innerHTML = "";
    for (const m of methods) {
      const o = document.createElement("option");
      o.value = m.id;
      o.textContent = m.label;
      methodSelect.appendChild(o);
    }
    methodSelect.value = data.default ?? methods[0]?.id ?? "voxel";
    renderParams();
    updateAlgoSummary();
    log(`methods: ${methods.map((m) => m.id).join(", ")}`);
  } catch (e) {
    methodDesc.textContent = "could not load methods (inference up?)";
    log(`methods fetch failed: ${String(e)}`);
  }
}

methodSelect.addEventListener("change", () => {
  renderParams();
  updateAlgoSummary();
});
resetParams.addEventListener("click", () => {
  renderParams();
  updateAlgoSummary();
});
paramsBox.addEventListener("input", updateAlgoSummary);
paramsBox.addEventListener("change", updateAlgoSummary);

algoToggle.addEventListener("click", () => setAlgoOpen(algoBox.hidden));
// `A`, matching the other single-key viewer shortcuts (P toggles planes).
window.addEventListener("keydown", (e) => {
  if (e.key !== "a" && e.key !== "A") return;
  const t = e.target as HTMLElement | null;
  // Don't steal the key while a control has focus.
  if (t && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName)) return;
  setAlgoOpen(algoBox.hidden);
});

let algoOpen = false;
try {
  algoOpen = localStorage.getItem(ALGO_KEY) === "1";
} catch {
  /* ignore */
}
setAlgoOpen(algoOpen);

async function loadLogs() {
  try {
    const res = await fetch("/api/recon/logs");
    const data = (await res.json()) as {
      logs?: { session: string; frames: number }[];
      error?: string;
    };
    logSelect.innerHTML = "";
    const logs = data.logs ?? [];
    if (!logs.length) {
      replayHint.textContent = data.error ? "log dir unreadable" : "no recorded runs";
      replayBtn.disabled = true;
      return;
    }
    for (const l of logs) {
      const opt = document.createElement("option");
      opt.value = l.session;
      opt.textContent = `${l.session.slice(0, 12)}… (${l.frames})`;
      logSelect.appendChild(opt);
    }
    replayHint.textContent = `${logs.length} run(s):`;
    replayBtn.disabled = false;
    log(
      `logs: ${logs.map((l) => `${l.session}(${l.frames})`).join(" ")}`,
    );
  } catch (e) {
    replayHint.textContent = "log list failed";
    log(`logs fetch failed: ${String(e)}`);
  }
}

function replayClear() {
  // Drop whatever is on screen (live session or previous replay) before the
  // new run starts, so the two never overlap.
  clearKeyframes();
  disposeCloud();
}

/**
 * A fetch rejection is always network-level - "Failed to fetch" never comes with
 * a status code - so it means the browser could not reach the dev server's
 * proxy at all. Probe the gateway on its own so the message says *which* piece
 * is down instead of leaving a bare TypeError on screen.
 */
async function explainFetchFailure(e: unknown): Promise<string> {
  const reason = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
  try {
    const r = await fetch("/api/recon/logs");
    return r.ok
      ? `replay failed (${reason}) - gateway is up, so it rejected the request`
      : `replay failed (${reason}) - gateway answered ${r.status}`;
  } catch {
    return (
      `replay failed (${reason}) - gateway unreachable on :3000. ` +
      "Is `npm run dev` still running? See docs/RUNNING.md."
    );
  }
}

replayBtn.addEventListener("click", () => {
  const session = logSelect.value;
  if (!session) return;
  replayClear();
  const method = methodSelect.value;
  const params = collectParams();
  updateAlgoSummary();
  // 1-based and inclusive. Both blank = the whole run; 7..7 = frame 7 alone.
  const from = Math.max(1, Math.floor(Number(fromFrameInput.value) || 1));
  const to = Math.max(0, Math.floor(Number(toFrameInput.value) || 0));
  const range = from === 1 && to === 0 ? "all frames" : `frames ${from}-${to || "end"}`;
  void fetch("/api/recon/replay", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ session, method, params, from, to }),
  })
    .then(async (r) => {
      if (!r.ok) {
        log(`replay failed ${r.status} for ${session}`);
        return;
      }
      const j = (await r.json()) as { frames?: number };
      log(
        `replaying ${session} (${range}, ${j.frames ?? "?"} frame(s)) ` +
          `with ${method} ${JSON.stringify(params)}`,
      );
    })
    .catch((e) => void explainFetchFailure(e).then(log));
});
stopBtn.addEventListener("click", () => {
  void fetch("/api/recon/replay/stop", { method: "POST" })
    .then(() => log("replay stopped"))
    .catch((e) => void explainFetchFailure(e).then(log));
});
void loadLogs();
void loadMethods();

// --- background ---------------------------------------------------------
// `grey` by default: it is the only theme where neither the darkest nor the
// brightest points of a vertex-coloured cloud disappear.
function savedBackground(): BackgroundId {
  try {
    const saved = localStorage.getItem("viewer.background");
    if (saved && (BACKGROUND_ORDER as string[]).includes(saved)) {
      return saved as BackgroundId;
    }
  } catch {
    /* ignore */
  }
  return "grey";
}

let backgroundId = setBackground(savedBackground());
log(`background: ${backgroundId} (B to change)`);

window.addEventListener("keydown", (e) => {
  if (e.key !== "b" && e.key !== "B") return;
  const t = e.target as HTMLElement | null;
  if (t && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName)) return;
  const next = BACKGROUND_ORDER.indexOf(backgroundId) + 1;
  backgroundId = setBackground(
    BACKGROUND_ORDER[next % BACKGROUND_ORDER.length] ?? "grey",
  );
  log(`background: ${backgroundId} (B to change)`);
});

engine.runRenderLoop(() => scene.render());
window.addEventListener("resize", () => engine.resize());
