"""Locating an off-frame pedestrian from the shadow they cast into the frame.

The public surface is deliberately small. The three pieces worth importing
directly are the target reparameterisation (:mod:`shadow_detection.geometry`),
the hand-crafted shadow descriptors (:mod:`shadow_detection.features`) and the
network (:mod:`shadow_detection.model`).
"""

from shadow_detection.geometry import (
    LEFT,
    RIGHT,
    DecomposedTarget,
    FrameSize,
    decompose,
    iou,
    reconstruct,
)

__all__ = [
    "LEFT",
    "RIGHT",
    "DecomposedTarget",
    "FrameSize",
    "decompose",
    "iou",
    "reconstruct",
    "__version__",
]

__version__ = "1.0.0"
