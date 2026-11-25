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
    # "scan" = detect + segment + lift; "depth" = depth map only.
    mode: Literal["scan", "depth"] = "scan"
    # Override the server's segmenter for this request.
    segmenter: Literal["sam2", "yolo11"] | None = None
    # Horizontal FOV in degrees; falls back to the server default when omitted.
    fovDeg: float | None = None
