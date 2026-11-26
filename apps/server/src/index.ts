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
import { readdir, readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import Fastify from "fastify";
import cors from "@fastify/cors";
import websocket, { type WebSocket } from "@fastify/websocket";
import type {
  AnchorClearMsg,
  AnchorMsg,
  ClientToServer,
  PlaneMsg,
  PoseBroadcastMsg,
  Quat,
  ReconClearMsg,
  ReconCloudMsg,
  ReconFrameMsg,
  ReconKeyframeMsg,
  ScanRequest,
  ScanResult,
  ServerToClient,
  Vec3,
  WorldDeltaMsg,
  WorldObject,
} from "@metroid-scan/shared";

const PORT = Number(process.env.PORT ?? 3000);
const HOST = process.env.HOST ?? "0.0.0.0";
const INFERENCE_URL = process.env.INFERENCE_URL ?? "http://127.0.0.1:8000";
// Recorded keyframe runs, written by the inference service (RECON_RECORD=1).
const RECON_LOG_DIR =
  process.env.RECON_LOG_DIR ??
  resolve(process.cwd(), "..", "..", "services", "inference", "recon_log");
const REPLAY_INTERVAL_MS = Number(process.env.REPLAY_INTERVAL_MS ?? 700);

/** One recorded frame's metadata (the .json sidecar, minus the image). */
interface RecordedFrame {
  position?: Vec3;
  quaternion?: Quat;
  pose?: number[];
  fovDeg?: number;
  width?: number;
  height?: number;
  sharpness?: number;
  hitDistance?: number | null;
  planes?: PlaneMsg[];
}

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
// Latest cloud snapshot per session, so a late viewer gets the map immediately.
const reconClouds = new Map<string, ReconCloudMsg>();

interface ReconStats {
  fused?: number;
  skipped?: number;
  voxels?: number;
  scale?: string;
  timing?: Record<string, number> | null;
  note?: string | null;
}

// Per-session fusion health, so the viewer can explain "no points" instead of
// just showing nothing. Cleared on disconnect.
const reconStats = new Map<string, ReconStats>();

function broadcastReconStatus(sessionId: string) {
  const list = reconKeyframes.get(sessionId) ?? [];
  broadcast(
    {
      type: "recon_status",
      sessionId,
      keyframes: list.length,
      lastFrameMs: list.at(-1)?.at ?? null,
      ...(reconStats.get(sessionId) ?? {}),
    },
    "viewer",
  );
}

function setReconStats(sessionId: string, stats: ReconStats) {
  reconStats.set(sessionId, stats);
  broadcastReconStatus(sessionId);
}

interface InferenceCloud {
  count: number;
  positions: string;
  colors: string;
  voxel: number;
  scaleA: number;
  scaleB: number;
  scaleMethod: string;
  scaleRms: number;
  scaleSamples: number;
}

/**
 * Hand a keyframe to the inference service and relay whatever comes back.
 * Never block the socket on it - the keyframe broadcast has already happened.
 *
 * The session's current metric planes ride along: the inference needs them to
 * recover the depth model's metric scale (MS3.1).
 */
async function forwardReconFrame(msg: ReconFrameMsg) {
  const currentPlanes = [...(planes.get(msg.sessionId)?.values() ?? [])];
  try {
    const res = await fetch(`${INFERENCE_URL}/recon/frame`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ ...msg, planes: currentPlanes }),
    });
    if (!res.ok) {
      app.log.warn({ status: res.status }, "inference /recon/frame failed");
      setReconStats(msg.sessionId, {
        note: `inference /recon/frame -> HTTP ${res.status}`,
      });
      return;
    }
    const data = (await res.json()) as {
      cloud?: InferenceCloud | null;
      voxels?: number;
      fused?: number;
      skipped?: number;
      reason?: string | null;
      timing?: Record<string, number>;
      scaleA?: number | null;
      scaleB?: number | null;
      scaleMethod?: string | null;
      scaleHit?: number | null;
    };

    // Show the plane-fit scale and the independent hit-test estimate side by
    // side: if they disagree, the plane fit is the problem.
    const parts: string[] = [];
    if (typeof data.scaleA === "number") {
      parts.push(
        `${data.scaleMethod ?? "scale"} a=${data.scaleA.toFixed(3)} b=${(data.scaleB ?? 0).toFixed(3)}`,
      );
    }
    if (typeof data.scaleHit === "number") {
      parts.push(`hit a=${data.scaleHit.toFixed(3)}`);
      if (typeof data.scaleA === "number" && data.scaleA > 0) {
        parts.push(`${(data.scaleHit / data.scaleA).toFixed(2)}x`);
      }
    }
    const scale = parts.length ? parts.join(" \u00b7 ") : undefined;
    setReconStats(msg.sessionId, {
      fused: data.fused,
      skipped: data.skipped,
      voxels: data.voxels,
      scale,
      timing: data.timing ?? null,
      note: data.reason ?? null,
    });

    if (!data.cloud) return;
    const cloud: ReconCloudMsg = {
      type: "recon_cloud",
      sessionId: msg.sessionId,
      at: Date.now(),
      voxels: data.voxels ?? 0,
      ...data.cloud,
    };
    reconClouds.set(msg.sessionId, cloud);
    app.log.info(
      { voxels: cloud.voxels, points: cloud.count, timing: data.timing },
      "recon cloud",
    );
    broadcast(cloud, "viewer");
  } catch (e) {
    app.log.warn({ err: e }, "inference /recon/frame unreachable");
    setReconStats(msg.sessionId, {
      note: `inference unreachable: ${e instanceof Error ? e.message : String(e)}`,
    });
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

function handleReconFrame(msg: ReconFrameMsg) {
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
  broadcast(kf, "viewer");
  broadcastReconStatus(msg.sessionId);
  void forwardReconFrame(msg);
}

/** The reconstruction methods + their params, so the viewer can build controls. */
app.get("/recon/methods", async (_req, reply) => {
  try {
    const res = await fetch(`${INFERENCE_URL}/recon/methods`);
    if (!res.ok) throw new Error(`inference ${res.status}`);
    return await res.json();
  } catch (e) {
    reply.code(502);
    return { error: `inference unreachable: ${String(e)}`, methods: [] };
  }
});

// --- offline replay ---------------------------------------------------------
// Replays a recorded run through the *live* pipeline, so the viewer sees it
// exactly as if a phone were walking the room: keyframes, planes, the fused
// cloud, and the pose breadcrumbs.

let stopReplay: (() => void) | null = null;

/** List the recorded runs available to replay. */
app.get("/recon/logs", async () => {
  try {
    const entries = await readdir(RECON_LOG_DIR, { withFileTypes: true });
    const logs: { session: string; frames: number }[] = [];
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const files = await readdir(join(RECON_LOG_DIR, entry.name));
      logs.push({
        session: entry.name,
        frames: files.filter((f) => f.endsWith(".json")).length,
      });
    }
    logs.sort((a, b) => a.session.localeCompare(b.session));
    return { dir: RECON_LOG_DIR, logs };
  } catch (e) {
    app.log.warn({ err: e }, "recon log dir unreadable");
    return { dir: RECON_LOG_DIR, logs: [], error: String(e) };
  }
});

async function startReplay(
  session: string,
  intervalMs: number,
  method: string,
  params: Record<string, number | string | boolean>,
  maxFrames: number,
): Promise<number> {
  stopReplay?.();
  let stopped = false;
  stopReplay = () => {
    stopped = true;
  };

  const dir = join(RECON_LOG_DIR, session);
  const all = (await readdir(dir)).filter((f) => f.endsWith(".json")).sort();
  // maxFrames=1 replays only the first keyframe, which is how you look at one
  // frame's geometry without waiting out the whole walk. 0 means all of them.
  const files = maxFrames > 0 ? all.slice(0, maxFrames) : all;
  const replayId = `replay-${session}`;

  // Fresh slate, so a second replay does not stack on the first.
  reconKeyframes.delete(replayId);
  reconClouds.delete(replayId);
  reconStats.delete(replayId);
  planes.delete(replayId);
  broadcast({ type: "recon_clear", sessionId: replayId } satisfies ReconClearMsg, "viewer");
  app.log.info(
    { session, frames: files.length, of: all.length, intervalMs, method },
    "replay starting",
  );

  void (async () => {
    for (const file of files) {
      if (stopped) break;
      try {
        const meta = JSON.parse(
          await readFile(join(dir, file), "utf8"),
        ) as RecordedFrame;
        const jpg = await readFile(join(dir, file.replace(/\.json$/, ".jpg")));

        // Planes as they were at that moment - the scale fit needs them.
        const seen = planes.get(replayId) ?? new Map<number, PlaneMsg>();
        for (const p of meta.planes ?? []) {
          seen.set(p.id, { ...p, type: "plane", sessionId: replayId, action: "added" });
        }
        planes.set(replayId, seen);

        const msg: ReconFrameMsg = {
          type: "recon_frame",
          sessionId: replayId,
          frameId: `replay-${file.replace(/\.json$/, "")}`,
          position: meta.position ?? [0, 0, 0],
          quaternion: meta.quaternion ?? [0, 0, 0, 1],
          pose: meta.pose ?? [],
          fovDeg: meta.fovDeg ?? 60,
          width: meta.width ?? 0,
          height: meta.height ?? 0,
          image: jpg.toString("base64"),
          reason: "manual",
          sharpness: meta.sharpness ?? 0,
          hitDistance: meta.hitDistance ?? null,
          record: false, // never re-record a replay
          method,
          params,
        };
        broadcast(
          {
            type: "pose",
            sessionId: replayId,
            position: msg.position,
            quaternion: msg.quaternion,
            forward: [0, 0, -1],
            tracking: "tracked",
          },
          "viewer",
        );
        handleReconFrame(msg);
      } catch (e) {
        app.log.warn({ file, err: e }, "replay frame failed");
      }
      await new Promise((r) => setTimeout(r, intervalMs));
    }
    app.log.info({ session, stopped }, "replay finished");
  })();
  return files.length;
}

app.post("/recon/replay", async (req, reply) => {
  const body = req.body as
    | {
        session?: string;
        intervalMs?: number;
        method?: string;
        params?: Record<string, number | string | boolean>;
        /** 0 or absent = every frame. 1 = only the first. */
        maxFrames?: number;
      }
    | undefined;
  const session = body?.session;
  if (!session) {
    reply.code(400);
    return { error: "missing session" };
  }
  if (!/^[A-Za-z0-9._-]+$/.test(session)) {
    reply.code(400);
    return { error: "bad session name" };
  }
  // Clamp: a huge value would just be the whole recording, but a negative or
  // NaN one must not silently become "all".
  const raw = Number(body?.maxFrames ?? 0);
  const maxFrames = Number.isFinite(raw) ? Math.max(0, Math.floor(raw)) : 0;

  let played: number;
  try {
    played = await startReplay(
      session,
      body?.intervalMs ?? REPLAY_INTERVAL_MS,
      body?.method ?? "voxel",
      body?.params ?? {},
      maxFrames,
    );
  } catch (e) {
    reply.code(404);
    return { error: `cannot replay ${session}: ${String(e)}` };
  }
  return {
    ok: true,
    session,
    method: body?.method ?? "voxel",
    frames: played,
  };
});

app.post("/recon/replay/stop", async () => {
  stopReplay?.();
  stopReplay = null;
  return { ok: true };
});

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
    for (const cloud of reconClouds.values()) send(socket, cloud);

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
        case "recon_frame":
          handleReconFrame(msg);
          break;
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
        reconClouds.delete(state.sessionId);
        reconStats.delete(state.sessionId);
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
