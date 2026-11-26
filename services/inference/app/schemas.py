"""
Wire schemas for the inference service.

Field names are camelCase to match the TypeScript `@metroid-scan/shared`
protocol, so the gateway can pass them straight through.

Stage A only fills `detections` (boxes + labels). The `Aimed` object is the
placeholder for Stage B (3D contour / OBB in camera space) - see
`docs/LIFTING.md`.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Mask(BaseModel):
    """Segmentation mask in image space (Stage B)."""

    encoding: Literal["rle", "png"] = "rle"
    data: str


class Detection(BaseModel):
    classId: int
    label: str
    score: float
    # [x1, y1, x2, y2] in pixels of the submitted image.
    box: list[float]
    # Mask boundary in image pixels (closed polyline).
    contour2d: list[list[float]] | None = None
    mask: Mask | None = None


class Obb(BaseModel):
    """Oriented bounding box in camera space (Stage B)."""

    center: list[float] = Field(..., min_length=3, max_length=3)
    size: list[float] = Field(..., min_length=3, max_length=3)
    quat: list[float] = Field(..., min_length=4, max_length=4)


class Aimed(BaseModel):
    """
    The object under the reticle - the one the user is scanning.

    Stage A: only `detectionIndex`/`classId`.
    Stage B: adds the camera-space geometry the phone lifts into the world.
    """

    detectionIndex: int
    classId: int
    name: str | None = None
    # The mask boundary in image pixels (closed polyline), Stage B.
    contour2d: list[list[float]] | None = None
    # All Stage-B fields below are in CAMERA space (camera at origin, +z forward).
    contourCam: list[list[float]] | None = None
    obbCam: Obb | None = None
    depthAtCentre: float | None = None
    scaleHint: str | None = None
    # Number of masked points unprojected (debug / quality signal).
    pointCount: int | None = None


class DepthResult(BaseModel):
    """Colourised depth map for the `mode="depth"` view."""

    png: str  # base64 PNG, same pixel size as the submitted image
    width: int
    height: int
    min: float
    max: float


class ScanResponse(BaseModel):
    frameId: str
    width: int
    height: int
    method: str = ""
    kind: str = ""
    detections: list[Detection]
    aimed: Aimed | None = None
    depth: DepthResult | None = None
    timing: dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    ok: bool
    device: str
    model: str
    loaded: bool


class ScanRequest(BaseModel):
    """A keyframe to scan. `image` is base64 (no data-URI prefix)."""

    image: str
    frameId: str = ""
    aimX: float = 0.5
    aimY: float = 0.5
    # Which wheel method to run (see GET /methods).
    method: str = "yolo"
    # Horizontal FOV in degrees; falls back to the server default when omitted.
    fovDeg: float | None = None


# ---------------------------------------------------------------------------
# MS3.0 - reconstruction keyframes (docs/RECONSTRUCTION.md)
# ---------------------------------------------------------------------------


class ReconFrameRequest(BaseModel):
    """One streamed reconstruction keyframe. `image` is base64 JPEG."""

    sessionId: str = ""
    frameId: str = ""
    # Camera world pose at the capture instant (ARCore `local` frame).
    position: list[float] = Field(default_factory=list)
    quaternion: list[float] = Field(default_factory=list)
    # Row-major 4x4 worldFromCamera in the OpenCV camera convention (+y down).
    # This is what fusion actually uses; position/quaternion are for the viewer.
    pose: list[float] = Field(default_factory=list)
    fovDeg: float = 60.0
    width: int = 0
    height: int = 0
    image: str = ""
    reason: str = ""
    sharpness: float = 0.0
    # Metric distance from the reticle to the ARCore hit-test surface, if any.
    hitDistance: float | None = None
    # False when this frame is being replayed from a recording.
    record: bool = True
    # Current metric ARCore planes, attached by the gateway (for the scale fit).
    planes: list[dict] = Field(default_factory=list)


class ReconCloud(BaseModel):
    """A decimated point-cloud snapshot for the viewer (base64 buffers)."""

    count: int
    positions: str  # base64 little-endian float32, xyz triples
    colors: str  # base64 uint8, rgb triples
    voxel: float = 0.0
    scaleA: float = 0.0
    scaleB: float = 0.0
    scaleMethod: str = ""
    scaleRms: float = 0.0
    scaleSamples: int = 0


class ReconStatusResponse(BaseModel):
    keyframes: int = 0
    fused: int = 0
    skipped: int = 0
    sessions: dict[str, int] = Field(default_factory=dict)
    lastFrameId: str | None = None
    lastJpegBytes: int = 0
    lastMs: float | None = None
    voxels: int = 0
    scaleA: float | None = None
    scaleB: float | None = None
    scaleMethod: str | None = None


class ReconFrameResponse(BaseModel):
    ok: bool = True
    keyframes: int = 0
    # Frames this session's map has actually fused (scale failures etc. skip).
    fused: int = 0
    skipped: int = 0
    voxels: int = 0
    cloud: ReconCloud | None = None
    timing: dict[str, float] = Field(default_factory=dict)
    # Why this frame was not fused, if it was not. Surfaced to the viewer so
    # "no points" is never silent.
    reason: str | None = None
    # The scale actually used, so the viewer can show it on every frame (not
    # only on the frames that happen to carry a cloud snapshot).
    scaleA: float | None = None
    scaleB: float | None = None
    scaleMethod: str | None = None
    # Independent estimate from the ARCore hit-test: hitDistance / rel@reticle.
    # If this disagrees with scaleA, the plane fit is the suspect.
    scaleHit: float | None = None
