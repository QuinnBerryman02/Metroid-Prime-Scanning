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
  PlaneMsg,
  PoseBroadcastMsg,
  ReconClearMsg,
  ReconFrameMsg,
  ReconKeyframeMsg,
  ScanRequest,
  ScanResult,
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
// Latest plane per (session, id), replayed to viewers so a late join sees the
// environment ARCore has already mapped.
const planes = new Map<string, Map<number, PlaneMsg>>();
// MS3.0: accepted keyframes (metadata only) per session, replayed to viewers.
// Capped so a long session cannot grow without bound.
const KEYFRAME_REPLAY_MAX = 200;
const reconKeyframes = new Map<string, ReconKeyframeMsg[]>();

function broadcastReconStatus(sessionId: string) {
  const list = reconKeyframes.get(sessionId) ?? [];
  broadcast(
    {
      type: "recon_status",
      sessionId,
      keyframes: list.length,
      lastFrameMs: list.at(-1)?.at ?? null,
    },
    "viewer",
  );
}

/** Hand a keyframe to the inference service. Never block the socket on it. */
async function forwardReconFrame(msg: ReconFrameMsg) {
  try {
    const res = await fetch(`${INFERENCE_URL}/recon/frame`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(msg),
    });
    if (!res.ok) {
      app.log.warn({ status: res.status }, "inference /recon/frame failed");
    }
  } catch (e) {
    app.log.warn({ err: e }, "inference /recon/frame unreachable");
  }
}

const app = Fastify({ logger: true, bodyLimit: 20 * 1024 * 1024 });
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

/**
 * MS3.0 progress: our own keyframe counters, plus whatever the inference
 * service reports (proxied, so a viewer only ever talks to the gateway).
 */
app.get("/recon/status", async (req) => {
  const sessions = [...reconKeyframes.entries()].map(([sessionId, list]) => ({
    sessionId,
    keyframes: list.length,
    lastFrameMs: list.at(-1)?.at ?? null,
  }));
  let inference: unknown = null;
  try {
    const res = await fetch(`${INFERENCE_URL}/recon/status`);
    if (res.ok) inference = await res.json();
  } catch (e) {
    req.log.warn({ err: e }, "inference /recon/status unreachable");
  }
  return {
    total: sessions.reduce((n, s) => n + s.keyframes, 0),
    sessions,
    inference,
  };
});

/** The wheel's method registry, straight from the inference service. */
app.get("/methods", async (req, reply) => {
  try {
    const res = await fetch(`${INFERENCE_URL}/methods`);
    if (!res.ok) {
      reply.code(502);
      return { error: `inference ${res.status}` };
    }
    return await res.json();
  } catch (e) {
    req.log.error({ err: e }, "inference unreachable");
    reply.code(502);
    return { error: `inference unreachable: ${String(e)}` };
  }
});

/**
 * Forward a keyframe to the inference service. The phone posts here (via the
 * Vite `/api` proxy) so it only ever talks to the gateway.
 */
app.post("/scan", async (req, reply) => {
  const body = req.body as ScanRequest | undefined;
  if (!body?.image) {
    reply.code(400);
    return { error: "missing base64 image" };
  }
  try {
    const res = await fetch(`${INFERENCE_URL}/scan`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const detail = await res.text();
      req.log.warn({ status: res.status, detail }, "inference /scan failed");
      reply.code(502);
      return { error: `inference ${res.status}`, detail };
    }
    const result = (await res.json()) as ScanResult;
    req.log.info(
      { detections: result.detections.length, aimed: result.aimed?.name },
      "scan",
    );
    return result;
  } catch (e) {
    req.log.error({ err: e }, "inference unreachable");
    reply.code(502);
    return { error: `inference unreachable: ${String(e)}` };
  }
});

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
    for (const list of planes.values()) for (const a of list.values()) send(socket, a);
    for (const list of reconKeyframes.values()) for (const kf of list) send(socket, kf);

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
        case "recon_frame": {
          const list = reconKeyframes.get(msg.sessionId) ?? [];
          const kf: ReconKeyframeMsg = {
            type: "recon_keyframe",
            sessionId: msg.sessionId,
            frameId: msg.frameId,
            index: list.length,
            position: msg.position,
            quaternion: msg.quaternion,
            fovDeg: msg.fovDeg,
            width: msg.width,
            height: msg.height,
            sharpness: msg.sharpness,
            at: Date.now(),
          };
          list.push(kf);
          if (list.length > KEYFRAME_REPLAY_MAX) list.splice(0, list.length - KEYFRAME_REPLAY_MAX);
          reconKeyframes.set(msg.sessionId, list);
          app.log.info(
            { sessionId: msg.sessionId, index: kf.index, reason: msg.reason },
            "recon keyframe",
          );
          broadcast(kf, "viewer");
          broadcastReconStatus(msg.sessionId);
          void forwardReconFrame(msg);
          break;
        }
        case "plane": {
          const perSession = planes.get(msg.sessionId) ?? new Map<number, PlaneMsg>();
          if (msg.action === "removed") perSession.delete(msg.id);
          else perSession.set(msg.id, msg);
          planes.set(msg.sessionId, perSession);
          broadcast(msg, "viewer");
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
        if (reconKeyframes.delete(state.sessionId)) {
          broadcast(
            { type: "recon_clear", sessionId: state.sessionId } satisfies ReconClearMsg,
            "viewer",
          );
        }
        const perSession = planes.get(state.sessionId);
        if (perSession) {
          for (const p of perSession.values()) {
            broadcast({ ...p, action: "removed" }, "viewer");
          }
          planes.delete(state.sessionId);
        }
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
