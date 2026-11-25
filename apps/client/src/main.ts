/**
 * Metroid-scan client (skeleton).
 *
 * Two modes share one Babylon engine:
 *   - AR view  (/ar, default on touch devices): camera passthrough + HUD overlays
 *   - Viewer   (/viewer): orbit the reconstructed world, select objects
 *
 * See docs/PLAN.md milestones 1, 4 and 5.
 */
import { Engine } from "@babylonjs/core/Engines/engine";
import { Scene } from "@babylonjs/core/scene";
import { ArcRotateCamera } from "@babylonjs/core/Cameras/arcRotateCamera";
import { HemisphericLight } from "@babylonjs/core/Lights/hemisphericLight";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import { Color4 } from "@babylonjs/core/Maths/math.color";

const canvas = document.getElementById("app") as HTMLCanvasElement;
const engine = new Engine(canvas, true, { stencil: true });
const scene = new Scene(engine);
scene.clearColor = new Color4(0.02, 0.03, 0.04, 1);

const camera = new ArcRotateCamera(
  "cam",
  Math.PI / 2,
  Math.PI / 3,
  4,
  Vector3.Zero(),
  scene,
);
camera.attachControl(canvas, true);
new HemisphericLight("light", new Vector3(0, 1, 0), scene);

// TODO(M1): if navigator.xr supports immersive-ar, create an AR session with
//   features: ["hit-test", "anchors", "plane-detection", "depth-sensing", "camera-access"]
// TODO(M4): WebSocket to gateway, project WorldObject OBBs onto the overlay canvas,
//           reticule raycast, scan action.
// TODO(M5): viewer mode renders point cloud + selectable OBBs.

engine.runRenderLoop(() => scene.render());
window.addEventListener("resize", () => engine.resize());
