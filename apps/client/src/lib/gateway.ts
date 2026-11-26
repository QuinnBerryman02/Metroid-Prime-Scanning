/**
 * Thin typed WebSocket client for the gateway.
 *
 * Uses a relative `/ws` URL so it goes through the Vite dev proxy (HTTPS page
 * -> wss localhost), avoiding mixed-content problems on the phone.
 */
import type {
  ClientToServer,
  Quat,
  ServerToClient,
  Vec3,
} from "@metroid-scan/shared";

export type GatewayStatus = "connecting" | "open" | "closed" | "error";

export interface GatewayHandlers {
  onPose?: (msg: Extract<ServerToClient, { type: "pose" }>) => void;
  onWorld?: (msg: Extract<ServerToClient, { type: "world_delta" }>) => void;
  onSession?: (
    msg: Extract<ServerToClient, { type: "session_event" }>,
  ) => void;
  onAnchor?: (msg: Extract<ServerToClient, { type: "anchor" }>) => void;
  onAnchorClear?: (
    msg: Extract<ServerToClient, { type: "anchor_clear" }>,
  ) => void;
  onStatus?: (status: GatewayStatus, detail?: string) => void;
}

export class GatewayClient {
  private ws: WebSocket | null = null;
  private poseTimer: number | null = null;
  private latestPose: {
    position: Vec3;
    quaternion: Quat;
    forward: Vec3;
    tracking: "tracked" | "lost";
  } | null = null;

  constructor(
    private readonly role: "device" | "viewer",
    private readonly sessionId: string,
    private readonly handlers: GatewayHandlers = {},
  ) {}

  connect() {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const url = `${scheme}://${location.host}/ws`;
    this.handlers.onStatus?.("connecting");
    const ws = new WebSocket(url);
    this.ws = ws;

    ws.onopen = () => {
      this.handlers.onStatus?.("open");
      this.send({ type: "hello", sessionId: this.sessionId, role: this.role });
    };
    ws.onclose = () => this.handlers.onStatus?.("closed");
    ws.onerror = () => this.handlers.onStatus?.("error");
    ws.onmessage = (ev) => {
      if (typeof ev.data !== "string") return;
      let msg: ServerToClient;
      try {
        msg = JSON.parse(ev.data) as ServerToClient;
      } catch {
        return;
      }
      if (msg.type === "pose") this.handlers.onPose?.(msg);
      else if (msg.type === "world_delta") this.handlers.onWorld?.(msg);
      else if (msg.type === "session_event") this.handlers.onSession?.(msg);
      else if (msg.type === "anchor") this.handlers.onAnchor?.(msg);
      else if (msg.type === "anchor_clear") this.handlers.onAnchorClear?.(msg);
    };
  }

  private send(msg: ClientToServer) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }

  /**
   * Queue a pose; it is flushed to the wire at ~15 Hz so we don't flood the
   * socket from the (up to 90 Hz) XR frame loop.
   */
  setPose(
    position: Vec3,
    quaternion: Quat,
    forward: Vec3,
    tracking: "tracked" | "lost",
  ) {
    this.latestPose = { position, quaternion, forward, tracking };
    if (this.poseTimer !== null) return;
    this.flushPose();
    this.poseTimer = window.setInterval(() => this.flushPose(), 66);
  }

  private flushPose() {
    if (!this.latestPose) return;
    this.send({ type: "pose", sessionId: this.sessionId, ...this.latestPose });
  }

  sendAnchor(
    id: string,
    position: Vec3,
    quaternion: Quat,
    size: Vec3,
    contour?: Vec3[],
    label?: string,
  ) {
    this.send({
      type: "anchor",
      sessionId: this.sessionId,
      id,
      position,
      quaternion,
      size,
      contour,
      label,
    });
  }

  sendAnchorClear() {
    this.send({ type: "anchor_clear", sessionId: this.sessionId });
  }

  close() {
    if (this.poseTimer !== null) window.clearInterval(this.poseTimer);
    this.poseTimer = null;
    this.ws?.close();
    this.ws = null;
  }
}
