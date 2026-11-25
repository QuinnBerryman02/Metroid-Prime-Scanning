/**
 * MS1 smoke test: device sends a pose -> gateway relays it to a viewer.
 *
 *   node apps/server/scripts/ws-smoke.mjs [ws://127.0.0.1:3000/ws]
 *
 * Uses the Node 22 global WebSocket, so no dependencies.
 */
const url = process.argv[2] ?? "ws://127.0.0.1:3000/ws";

const open = (role) =>
  new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    ws.onopen = () => {
      ws.send(JSON.stringify({ type: "hello", sessionId: "smoke-" + role, role }));
      resolve(ws);
    };
    ws.onerror = reject;
  });

const waitForPose = (ws, timeoutMs = 2000) =>
  new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error("timed out waiting for pose")), timeoutMs);
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "pose") {
        clearTimeout(t);
        resolve(msg);
      }
    };
  });

const device = await open("device");
const viewer = await open("viewer");

const pose = {
  type: "pose",
  sessionId: "smoke-device",
  position: [1.5, 1.6, -0.5],
  quaternion: [0, 0, 0, 1],
  forward: [0, 0, -1],
  tracking: "tracked",
};

const received = waitForPose(viewer);
device.send(JSON.stringify(pose));
const got = await received;

const ok =
  got.position[0] === 1.5 && got.forward[2] === -1 && got.sessionId === "smoke-device";
console.log(ok ? "PASS" : "FAIL", JSON.stringify(got));

device.close();
viewer.close();
process.exit(ok ? 0 : 1);
