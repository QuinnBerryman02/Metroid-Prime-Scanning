/**
 * Metroid-scan gateway.
 *
 * Responsibilities (see docs/ARCHITECTURE.md):
 *   - accept device frames + poses over WebSocket
 *   - forward to the Python inference service
 *   - own the authoritative world model (objects, fusion, deltas)
 *   - relay device pose to desktop viewers
 *
 * MS1 scope: role-based WebSocket hub + pose relay. Inference/world model are stubs.
 */
import Fastify from "fastify";
import cors from "@fastify/cors";
import websocket, { type WebSocket } from "@fastify/websocket";
import type {
  AnchorClearMsg,
  AnchorMsg,
  ClientToServer,
  PoseBroadcastMsg,
  ServerToClient,
  WorldDeltaMsg,
  WorldObject,
} from "@metroid-scan/shared";

const PORT = Number(process.env.PORT ?? 3000);
const HOST = process.env.HOST ?? "0.0.0.0";
const INFERENCE_URL = process.env.INFERENCE_URL ?? "http://127.0.0.1:8000";

type Role = "device" | "viewer";

interface ClientState {
  role: Role;
  sessionId: string;
}

// TODO: real authoritative store, keyed by session.
const world = new Map<string, WorldObject>();
const clients = new Map<WebSocket, ClientState>();
const latestPose = new Map<string, PoseBroadcastMsg>();
const anchors = new Map<string, AnchorMsg[]>();

const app = Fastify({ logger: true });
await app.register(cors, { origin: true });
await app.register(websocket);

function send(socket: WebSocket, msg: ServerToClient) {
  if (socket.readyState === 1) socket.send(JSON.stringify(msg));
}

function broadcast(msg: ServerToClient, role: Role) {
  const payload = JSON.stringify(msg);
  for (const [socket, state] of clients) {
    if (state.role === role && socket.readyState === 1) socket.send(payload);
  }
}

app.get("/health", async () => ({
  ok: true,
  objects: world.size,
  clients: clients.size,
  inference: INFERENCE_URL,
}));

app.get("/world", async () => ({ objects: [...world.values()] }));

app.register(async (f) => {
  f.get("/ws", { websocket: true }, (socket) => {
    app.log.info("ws client connected");
    // Default to viewer until a `hello` identifies the role.
    clients.set(socket, { role: "viewer", sessionId: "unknown" });
    send(socket, {
      type: "world_delta",
      added: [...world.values()],
      updated: [],
      removed: [],
    } satisfies WorldDeltaMsg);
    // Replay known device poses + anchors to a newly connected viewer.
    for (const pose of latestPose.values()) send(socket, pose);
    for (const list of anchors.values()) for (const a of list) send(socket, a);

    socket.on("message", (raw: Buffer, isBinary: boolean) => {
      if (isBinary) return; // binary frames handled in a later milestone
      let msg: ClientToServer;
      try {
        msg = JSON.parse(raw.toString()) as ClientToServer;
      } catch {
        send(socket, { type: "error", message: "invalid JSON" });
        return;
      }

      switch (msg.type) {
        case "hello": {
          clients.set(socket, { role: msg.role, sessionId: msg.sessionId });
          app.log.info({ role: msg.role, sessionId: msg.sessionId }, "hello");
          if (msg.role === "device") {
            broadcast(
              { type: "session_event", sessionId: msg.sessionId, status: "connected" },
              "viewer",
            );
          }
          break;
        }
        case "pose": {
          const state = clients.get(socket);
          if (state) state.sessionId = msg.sessionId;
          const pose: PoseBroadcastMsg = { ...msg, type: "pose" };
          latestPose.set(msg.sessionId, pose);
          broadcast(pose, "viewer");
          break;
        }
        case "anchor": {
          const list = anchors.get(msg.sessionId) ?? [];
          list.push(msg);
          anchors.set(msg.sessionId, list);
          broadcast(msg, "viewer");
          break;
        }
        case "anchor_clear": {
          anchors.delete(msg.sessionId);
          const clear: AnchorClearMsg = { type: "anchor_clear", sessionId: msg.sessionId };
          broadcast(clear, "viewer");
          break;
        }
        case "scan_frame":
        case "rescan":
          // TODO(MS2/MS3): forward to inference, fuse results, emit world deltas.
          app.log.debug({ type: msg.type }, "scan message (unhandled in MS1)");
          break;
        default:
          break;
      }
    });

    socket.on("close", () => {
      const state = clients.get(socket);
      clients.delete(socket);
      if (state?.role === "device") {
        latestPose.delete(state.sessionId);
        broadcast(
          {
            type: "session_event",
            sessionId: state.sessionId,
            status: "disconnected",
          },
          "viewer",
        );
      }
      app.log.info("ws client disconnected");
    });
  });
});

try {
  await app.listen({ port: PORT, host: HOST });
  app.log.info(`gateway listening on :${PORT}`);
} catch (err) {
  app.log.error(err);
  process.exit(1);
}
