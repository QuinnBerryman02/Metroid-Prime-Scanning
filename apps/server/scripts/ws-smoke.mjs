/**
 * MS1 smoke test: device sends a pose + anchor -> gateway relays to a viewer.
 *
 *   node apps/server/scripts/ws-smoke.mjs [wss://host:port/ws]
 *
 * Uses the Node 22 global WebSocket, so no dependencies. For the HTTPS dev
 * proxy use:  NODE_TLS_REJECT_UNAUTHORIZED=0 node ... wss://localhost:5173/ws
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

const waitFor = (ws, type, timeoutMs = 2000) =>
  new Promise((resolve, reject) => {
    const t = setTimeout(
      () => reject(new Error(`timed out waiting for "${type}"`)),
      timeoutMs,
    );
    ws.addEventListener("message", function onMsg(ev) {
      const msg = JSON.parse(ev.data);
      if (msg.type === type) {
        clearTimeout(t);
        ws.removeEventListener("message", onMsg);
        resolve(msg);
      }
    });
  });

const results = [];
const check = (name, cond, detail = "") => {
  results.push([name, cond]);
  console.log(`${cond ? "PASS" : "FAIL"}  ${name}${detail ? "  " + detail : ""}`);
};

const device = await open("device");
const viewer = await open("viewer");

// pose relay
const pose = {
  type: "pose",
  sessionId: "smoke-device",
  position: [1.5, 1.6, -0.5],
  quaternion: [0, 0, 0, 1],
  forward: [0, 0, -1],
  tracking: "tracked",
};
let rx = waitFor(viewer, "pose");
device.send(JSON.stringify(pose));
const gotPose = await rx;
check(
  "pose relay",
  gotPose.position[0] === 1.5 && gotPose.sessionId === "smoke-device",
  JSON.stringify(gotPose.position),
);

// anchor relay
const anchor = {
  type: "anchor",
  sessionId: "smoke-device",
  id: "box-0",
  position: [0.2, 0.0, -0.4],
  quaternion: [0, 0, 0, 1],
  size: [0.12, 0.12, 0.12],
};
rx = waitFor(viewer, "anchor");
device.send(JSON.stringify(anchor));
const gotAnchor = await rx;
check("anchor relay", gotAnchor.id === "box-0" && gotAnchor.size[0] === 0.12);

// anchor replay for a late-joining viewer
const late = await open("viewer");
const replayed = await waitFor(late, "anchor");
check("anchor replay to late viewer", replayed.id === "box-0");

// clear
rx = waitFor(viewer, "anchor_clear");
device.send(JSON.stringify({ type: "anchor_clear", sessionId: "smoke-device" }));
const gotClear = await rx;
check("anchor clear broadcast", gotClear.sessionId === "smoke-device");

device.close();
viewer.close();
late.close();

const ok = results.every(([, c]) => c);
console.log(ok ? "\nALL PASS" : "\nFAILURES");
process.exit(ok ? 0 : 1);
