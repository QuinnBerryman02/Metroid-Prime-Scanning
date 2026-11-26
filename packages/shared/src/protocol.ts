/**
 * Shared wire protocol + world-model types.
 * Kept runtime-agnostic (no DOM / no Node globals) so it can be imported by
 * the Bun/Node gateway, the browser client and tooling.
 */

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number];
/** Column-major 4x4, matching WebXR / glMatrix conventions. */
export type Mat4 = [
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
];

export interface Intrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
}

/** A 2D detection returned by the inference service. */
export interface Detection {
  /** COCO-ish or open-vocab label id. */
  classId: number;
  label: string;
  score: number;
  /** [x1, y1, x2, y2] in pixels of the submitted (downscaled) image. */
  box: [number, number, number, number];
  /** Mask boundary in image pixels (closed polyline). */
  contour2d?: [number, number][];
  /** Optional per-pixel mask, RLE or base64-PNG, in the same image space. */
  mask?: { encoding: "rle" | "png"; data: string };
}

/** One entry in the radial-wheel method registry (`GET /methods`). */
export interface MethodInfo {
  id: string;
  label: string;
  kind: "prompt" | "detect" | "depth";
  device: string;
  available: boolean;
}

export interface MethodsResponse {
  device: string;
  methods: MethodInfo[];
}

/** Oriented bounding box in camera space (Stage B). */
export interface Obb {
  center: Vec3;
  size: Vec3;
  quat: Quat;
}

/** The object under the reticle. Stage A fills only index/class/label. */
export interface Aimed {
  detectionIndex: number;
  classId: number;
  name?: string;
  /** Mask boundary in image pixels (closed polyline), Stage B. */
  contour2d?: [number, number][];
  /** Camera-space (origin = camera, +z forward) - see docs/LIFTING.md. */
  contourCam?: Vec3[];
  obbCam?: Obb;
  depthAtCentre?: number;
  scaleHint?: string;
  /** Number of masked points unprojected (debug / quality signal). */
  pointCount?: number;
}

/** Colourised depth map returned by `mode: "depth"`. */
export interface DepthResult {
  /** base64 PNG (no data-URI prefix). */
  png: string;
  width: number;
  height: number;
  min: number;
  max: number;
}

/** POST /scan request (base64 image, no data-URI prefix). */
export interface ScanRequest {
  image: string;
  frameId?: string;
  aimX?: number;
  aimY?: number;
  /** Which wheel method to run (see `MethodInfo`). */
  method?: string;
  fovDeg?: number;
}

/** POST /scan response. */
export interface ScanResult {
  frameId: string;
  width: number;
  height: number;
  method: string;
  kind: "prompt" | "detect" | "depth";
  detections: Detection[];
  aimed: Aimed | null;
  depth?: DepthResult | null;
  timing: Record<string, number>;
}

/** Classification / explanation attached to a scanned object. */
export interface Classification {
  label: string;
  confidence: number;
  /** Longer VLM caption / explanation. */
  description?: string;
  tags?: string[];
}

/** An object that has been scanned and anchored in the world. */
export interface WorldObject {
  id: string;
  classification: Classification;
  color: string;

  /** World-space oriented bounding box (the "scan border"). */
  center: Vec3;
  size: Vec3;
  quaternion: Quat;

  /** Optional object point cloud (x,y,z triples, world space). */
  hull?: number[];

  /** Original 2D box in the keyframe, for HUD appearance matching. */
  sourceBox?: [number, number, number, number];
  /** WebXR anchor uuid on the scanning device, if any. */
  anchorUuid?: string;

  firstSeen: number;
  lastSeen: number;
  scanCount: number;
}

// ---------------------------------------------------------------------------
// Client -> Server
// ---------------------------------------------------------------------------

export interface ScanFrameMsg {
  type: "scan_frame";
  frameId: string;
  sessionId: string;
  /** worldFromCamera at capture time. */
  pose: Mat4;
  intrinsics: Intrinsics;
  /** Frames are sent as a separate binary WS message keyed by frameId. */
  jpegBytes: number;
  depthBytes?: number;
  /** Screen-centre in normalised image coords, if the user is aiming. */
  aim?: [number, number];
}

export interface RescanMsg {
  type: "rescan";
  sessionId: string;
  objectId: string;
  frameId: string;
  jpegBytes: number;
}

export interface HelloMsg {
  type: "hello";
  sessionId: string;
  role: "device" | "viewer";
}

/** Device -> server: pose of the phone camera, streamed at a modest rate. */
export interface PoseMsg {
  type: "pose";
  sessionId: string;
  /** Camera position in the (Babylon) reference space, metres. */
  position: Vec3;
  /** Camera orientation in the reference space. */
  quaternion: Quat;
  /** Camera forward unit vector in the reference space. */
  forward: Vec3;
  tracking: "tracked" | "lost";
}

/** Device -> server -> viewer: a world-anchored scanned object. */
export interface AnchorMsg {
  type: "anchor";
  sessionId: string;
  id: string;
  position: Vec3;
  quaternion: Quat;
  size: Vec3;
  /** World-space outline relative to `position` (metres). */
  contour?: Vec3[];
  label?: string;
}

/** Device -> server -> viewer: remove all anchors for a session. */
export interface AnchorClearMsg {
  type: "anchor_clear";
  sessionId: string;
}

// ---------------------------------------------------------------------------
// MS3.0 - reconstruction keyframes (docs/RECONSTRUCTION.md)
// ---------------------------------------------------------------------------

/**
 * Device -> server: one streamed reconstruction keyframe.
 *
 * The camera image rides along as base64 (like `/scan`). At ~3 Hz on a LAN the
 * 33% overhead is irrelevant and every hop stays plain JSON. If the keyframe
 * rate ever climbs, swap the body for a binary WS message keyed by frameId.
 */
export interface ReconFrameMsg {
  type: "recon_frame";
  sessionId: string;
  frameId: string;
  /** Camera world pose at the capture instant (ARCore `local` frame). */
  position: Vec3;
  quaternion: Quat;
  /** Horizontal FOV in degrees, from the XR projection matrix. */
  fovDeg: number;
  width: number;
  height: number;
  /** base64 JPEG (no data-URI prefix). */
  image: string;
  /** Why this frame qualified (debug). */
  reason: "first" | "move" | "rotate" | "both";
  /** Laplacian variance of the greyscale frame (higher = sharper). */
  sharpness: number;
}

/** Server -> viewer: a keyframe was accepted (metadata only - no pixels). */
export interface ReconKeyframeMsg {
  type: "recon_keyframe";
  sessionId: string;
  frameId: string;
  /** 0-based index within the session. */
  index: number;
  position: Vec3;
  quaternion: Quat;
  fovDeg: number;
  width: number;
  height: number;
  sharpness: number;
  at: number;
}

/** Server -> any: reconstruction progress for a session. */
export interface ReconStatusMsg {
  type: "recon_status";
  sessionId: string;
  keyframes: number;
  lastFrameMs: number | null;
}

/** Server -> viewer: drop all keyframe markers for a session. */
export interface ReconClearMsg {
  type: "recon_clear";
  sessionId: string;
}

/**
 * Device -> server -> viewer: one ARCore-detected plane (the free environment
 * model WebXR hands us - polygon is in world/Babylon space, metres).
 */
export interface PlaneMsg {
  type: "plane";
  sessionId: string;
  /** Babylon-assigned plane id, stable within a session. */
  id: number;
  action: "added" | "updated" | "removed";
  orientation: "horizontal" | "vertical" | "unknown";
  /** Closed polygon, world space (metres). */
  polygon: Vec3[];
  /** Polygon centroid, world space. */
  center: Vec3;
  /** World-space unit normal. */
  normal: Vec3;
}

// ---------------------------------------------------------------------------
// Server -> Client
// ---------------------------------------------------------------------------

export interface WorldDeltaMsg {
  type: "world_delta";
  added: WorldObject[];
  updated: WorldObject[];
  removed: string[];
}

export interface ScanResultMsg {
  type: "scan_result";
  frameId: string;
  objects: WorldObject[];
}

/** Server -> viewer: current pose of a connected device. */
export interface PoseBroadcastMsg extends Omit<PoseMsg, "type"> {
  type: "pose";
}

/** Server -> any: a device came online / went offline. */
export interface SessionEventMsg {
  type: "session_event";
  sessionId: string;
  status: "connected" | "disconnected";
}

export interface ErrorMsg {
  type: "error";
  message: string;
}

export type ClientToServer =
  | ScanFrameMsg
  | RescanMsg
  | HelloMsg
  | PoseMsg
  | AnchorMsg
  | AnchorClearMsg
  | PlaneMsg
  | ReconFrameMsg;
export type ServerToClient =
  | WorldDeltaMsg
  | ScanResultMsg
  | PoseBroadcastMsg
  | AnchorMsg
  | AnchorClearMsg
  | PlaneMsg
  | ReconKeyframeMsg
  | ReconStatusMsg
  | ReconClearMsg
  | SessionEventMsg
  | ErrorMsg;

/** Per-class display colours (Metroid-ish palette). */
export const CLASS_COLORS: Record<string, string> = {
  person: "#ff5252",
  chair: "#ffb74d",
  table: "#4db6ac",
  laptop: "#4fc3f7",
  cup: "#ba68c8",
  bottle: "#81c784",
  book: "#f06292",
  keyboard: "#ffd54f",
  monitor: "#90a4ae",
  default: "#7cf7c4",
};

const DEFAULT_COLOR = CLASS_COLORS.default ?? "#7cf7c4";

export function colorForLabel(label: string): string {
  return CLASS_COLORS[label] ?? DEFAULT_COLOR;
}
