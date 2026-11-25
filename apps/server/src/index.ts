/**
 * Metroid-scan gateway.
 *
 * Responsibilities (see docs/ARCHITECTURE.md):
 *   - accept device frames + poses over WebSocket
 *   - forward to the Python inference service
 *   - own the authoritative world model (objects, fusion, deltas)
 *   - broadcast world state to desktop viewers
 *
 * This is a skeleton: the world-model store and inference client are stubbed.
 */
import Fastify from "fastify";
import cors from "@fastify/cors";
import websocket from "@fastify/websocket";
import type { WorldDeltaMsg, WorldObject } from "@metroid-scan/shared";

const PORT = Number(process.env.PORT ?? 3000);
const HOST = process.env.HOST ?? "0.0.0.0";
const INFERENCE_URL = process.env.INFERENCE_URL ?? "http://127.0.0.1:8000";

// TODO: real authoritative store, keyed by session.
const world = new Map<string, WorldObject>();

const app = Fastify({ logger: true });
await app.register(cors, { origin: true });
await app.register(websocket);

app.get("/health", async () => ({
  ok: true,
  objects: world.size,
  inference: INFERENCE_URL,
}));

app.get("/world", async () => ({ objects: [...world.values()] }));

app.register(async (f) => {
  f.get("/ws", { websocket: true }, (socket) => {
    app.log.info("ws client connected");
    socket.send(
      JSON.stringify({
        type: "world_delta",
        added: [...world.values()],
        updated: [],
        removed: [],
      } satisfies WorldDeltaMsg),
    );

    socket.on("message", (raw: Buffer) => {
      // TODO: parse ClientToServer, fetch binary frame, call inference, fuse, broadcast.
      app.log.debug({ bytes: raw.length }, "ws message");
    });

    socket.on("close", () => app.log.info("ws client disconnected"));
  });
});

// TODO: POST /scan for one-shot stills during development.

try {
  await app.listen({ port: PORT, host: HOST });
  app.log.info(`gateway listening on :${PORT}`);
} catch (err) {
  app.log.error(err);
  process.exit(1);
}
