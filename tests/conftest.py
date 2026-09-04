"""Synthetic fixtures, so the suite runs without the challenge dataset.

The real data is not redistributable (see ``docs/dataset.md``), so the tests
build images and annotations that share the dataset's structure: 720x480
frames, a bright road with a top-to-bottom brightness gradient, and a tapered
dark wedge standing in for a shadow.

The wedge is tapered rather than rectangular on purpose. A rectangular blob
gives every one of its columns the same shadow-pixel count, and
:attr:`~shadow_detection.features.Feature.SHADOW_PEAK_COLUMN` is an ``argmax``
over those counts -- which breaks ties by taking the first index and is
therefore not mirror-equivariant on a plateau. Real shadows taper, so the
fixture does too; ``test_features.py`` covers the plateau case separately.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from shadow_detection.geometry import FrameSize

FRAME = FrameSize(720, 480)

_SHADOW_DARKENING = 0.45
_EDGE_STRIP_DARKENING = 0.60
_EDGE_STRIP_WIDTH = 25
_EDGE_STRIP_HEIGHT = 40


def make_road_image(
    shadow_centre_x: int = 200,
    shadow_width: int = 120,
    edge_strip: str | None = "left",
    taper: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """A road-like frame with one dark wedge, deliberately left/right asymmetric.

    Args:
        shadow_centre_x: horizontal centre of the wedge, in pixels.
        shadow_width: horizontal extent of the wedge.
        edge_strip: ``"left"``, ``"right"`` or ``None``. Darkens a strip in that
            margin, which is what gives the edge-brightness and margin-density
            features something asymmetric to measure.
        taper: when ``False`` the shadow is a rectangle, so its column density
            is a flat plateau. Used to exercise the ``argmax`` tie case.
        seed: noise seed.

    The vertical brightness gradient matters: it is what makes a global
    threshold fail and the per-row median in
    :func:`shadow_detection.features.extract_features` necessary.
    """
    rng = np.random.default_rng(seed)
    height, width = FRAME.height, FRAME.width

    gradient = np.linspace(200, 120, height, dtype=np.float32)[:, None]
    image = np.repeat(gradient, width, axis=1)
    image += rng.normal(0.0, 3.0, size=image.shape).astype(np.float32)

    top = int(height * 0.62)
    left = max(shadow_centre_x - shadow_width // 2, 0)
    right = min(shadow_centre_x + shadow_width // 2, width)

    for offset, column in enumerate(range(left, right)):
        if taper:
            # Thickness grows left to right, so the column density has a single
            # unambiguous maximum at the far end of the wedge.
            fraction = offset / max(right - left - 1, 1)
            thickness = int(20 + 60 * fraction)
        else:
            thickness = 60
        image[top : min(top + thickness, height), column] *= _SHADOW_DARKENING

    if edge_strip == "left":
        image[top : top + _EDGE_STRIP_HEIGHT, :_EDGE_STRIP_WIDTH] *= _EDGE_STRIP_DARKENING
    elif edge_strip == "right":
        image[top : top + _EDGE_STRIP_HEIGHT, -_EDGE_STRIP_WIDTH:] *= _EDGE_STRIP_DARKENING

    return np.repeat(np.clip(image, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)


@pytest.fixture
def road_image() -> np.ndarray:
    return make_road_image()


@pytest.fixture
def flat_image() -> np.ndarray:
    """A uniform frame with no shadow at all, to exercise the fallback path."""
    return np.full((FRAME.height, FRAME.width, 3), 180, dtype=np.uint8)


@pytest.fixture
def annotation_file(tmp_path):
    """Write one annotation JSON in the challenge's exact schema."""

    def _write(
        name: str = "image_0",
        xmin: float = -228.877,
        ymin: float = 223.619,
        xmax: float = -110.615,
        ymax: float = 416.359,
        direction: int = 1,
    ):
        payload = {
            "file_name": name,
            "walking_into_frame_bool": direction,
            "bbox": {
                "top_left": [xmin, ymin],
                "top_right": [xmax, ymin],
                "bottom_left": [xmin, ymax],
                "bottom_right": [xmax, ymax],
            },
        }
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def submission_frame():
    """Build a minimal valid submission table."""
    import pandas as pd

    def _make(ids, xmin, direction=-1):
        directions = [direction] * len(ids) if isinstance(direction, int) else list(direction)
        return pd.DataFrame(
            {
                "id": list(ids),
                "xmin": list(xmin),
                "ymin": [220.0] * len(ids),
                "xmax": [x + 80.0 for x in xmin],
                "ymax": [400.0] * len(ids),
                "direction": directions,
            }
        )

    return _make
