"""The target reparameterisation that made this problem tractable.

Ground truth is an axis-aligned box for a person who is *entirely outside* the
frame: every one of the 1692 training boxes lies fully to the left of x=0 or
fully to the right of x=720. Regressing ``(xmin, ymin, xmax, ymax)`` directly
means asking a network to emit coordinates outside the image it is looking at,
on a scale it has no anchor for -- which is exactly what the first two
iterations of this project did, and why they stalled just above the
constant-prediction baseline (see ``docs/experiments.md``).

Instead the box is decomposed into five quantities that are each bounded,
roughly zero-centred once standardised, and physically meaningful:

===================  ==========================================================
``side``             which edge the person is behind (binary classification)
``distance_from_edge``  how far past that edge, in pixels (always positive)
``bbox_width``       box width in pixels
``bbox_height``      box height in pixels
``y_center``         vertical centre of the box, in pixels
===================  ==========================================================

:func:`decompose` and :func:`reconstruct` are exact inverses, which
``tests/test_geometry.py`` checks against every training annotation shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

#: Person is behind the left edge of the frame (``xmin < 0``).
LEFT: Final[int] = 0
#: Person is behind the right edge of the frame (``xmax > frame width``).
RIGHT: Final[int] = 1

#: Order of the four regression targets everywhere in this project. The dataset,
#: the model's regression head and the inference code all rely on this order, so
#: it lives in exactly one place.
REGRESSION_TARGETS: Final[tuple[str, ...]] = (
    "distance_from_edge",
    "bbox_width",
    "bbox_height",
    "y_center",
)


@dataclass(frozen=True, slots=True)
class FrameSize:
    """Pixel dimensions of the camera frame.

    The challenge images are all 720x480, but nothing below hardcodes that.
    """

    width: int = 720
    height: int = 480


@dataclass(frozen=True, slots=True)
class DecomposedTarget:
    """A ground-truth box expressed in the coordinates the model predicts."""

    side: int
    distance_from_edge: float
    bbox_width: float
    bbox_height: float
    y_center: float

    def as_regression_tuple(self) -> tuple[float, float, float, float]:
        """The four continuous targets, in :data:`REGRESSION_TARGETS` order."""
        return (
            self.distance_from_edge,
            self.bbox_width,
            self.bbox_height,
            self.y_center,
        )


def decompose(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    frame: FrameSize | None = None,
) -> DecomposedTarget:
    """Turn an off-frame box into ``(side, distance, width, height, y_center)``.

    ``side`` is read off the sign of ``xmin``: because every annotated person is
    fully outside the frame, ``xmin < 0`` is equivalent to "behind the left
    edge". ``distance_from_edge`` is measured to the box's *outer* corner, i.e.
    the one further from the frame, so it is the largest of the two horizontal
    offsets and is always positive.
    """
    frame = frame or FrameSize()
    side = LEFT if xmin < 0 else RIGHT
    distance = abs(xmin) if side == LEFT else xmax - frame.width
    return DecomposedTarget(
        side=side,
        distance_from_edge=distance,
        bbox_width=xmax - xmin,
        bbox_height=ymax - ymin,
        y_center=(ymin + ymax) / 2,
    )


def reconstruct(
    side: int,
    distance_from_edge: float,
    bbox_width: float,
    bbox_height: float,
    y_center: float,
    frame: FrameSize | None = None,
) -> tuple[float, float, float, float]:
    """Inverse of :func:`decompose`; returns ``(xmin, ymin, xmax, ymax)``.

    The horizontal anchor is the outer corner, so the box is built outwards from
    the frame edge and the width extends back towards it.
    """
    frame = frame or FrameSize()
    if side == LEFT:
        xmin = -distance_from_edge
        xmax = xmin + bbox_width
    else:
        xmax = frame.width + distance_from_edge
        xmin = xmax - bbox_width
    ymin = y_center - bbox_height / 2
    ymax = y_center + bbox_height / 2
    return xmin, ymin, xmax, ymax


def iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Intersection over union for two ``(xmin, ymin, xmax, ymax)`` boxes.

    This is the local stand-in for the challenge metric. Corners are sorted
    first so a box the model emitted inside-out still scores rather than
    producing a negative area.
    """
    ax1, ay1, ax2, ay2 = (
        min(box_a[0], box_a[2]),
        min(box_a[1], box_a[3]),
        max(box_a[0], box_a[2]),
        max(box_a[1], box_a[3]),
    )
    bx1, by1, bx2, by2 = (
        min(box_b[0], box_b[2]),
        min(box_b[1], box_b[3]),
        max(box_b[0], box_b[2]),
        max(box_b[1], box_b[3]),
    )

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0
