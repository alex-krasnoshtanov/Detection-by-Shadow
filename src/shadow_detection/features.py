"""Nineteen hand-crafted descriptors of the shadow, fused into the CNN.

A ResNet pooled to a 2048-vector is good at "what kind of scene is this" and
surprisingly poor at "how far left of the frame does this shadow point". The
descriptors below inject that geometry explicitly: where the dark region sits,
how its mass is split between the two halves of the road, which way its
principal axis leans, and how deep the shadow is relative to the surrounding
tarmac. They are concatenated onto the pooled backbone features before the
prediction heads (see :mod:`shadow_detection.model`).

Two groups, computed from two different definitions of "shadow":

* **0-13, row-median group.** A pixel is shadow if it is below 85% of the
  median brightness of *its own row*. Per-row normalisation is what makes this
  robust to the strong top-to-bottom brightness gradient in the challenge
  images, where the far end of the road is much brighter than the near end.
* **14-18, depth group.** Subtracting the image from a heavily blurred copy
  (sigma=20) gives a local-contrast map; values above 8 count as shadow. This
  measures how *dark* the shadow is rather than where it is, which separates a
  crisp nearby shadow from a faint distant one.

The numeric constants here are the ones the published results were produced
with. They are named rather than inlined, but deliberately not re-tuned; see
"Known issues" in ``docs/method.md``.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from scipy.ndimage import gaussian_filter, sobel

#: Length of the feature vector. The model's input width depends on this.
NUM_FEATURES = 19

# --- Thresholds and window sizes --------------------------------------------

#: A pixel is shadow below this fraction of its own row's median brightness.
ROW_MEDIAN_SHADOW_RATIO = 0.85
#: The row-median group only looks below this fraction of the image height;
#: above it is sky and background rather than road.
ROAD_TOP_FRACTION = 0.40
#: Edge-brightness probes are taken from below this fraction of the height.
EDGE_PROBE_TOP_FRACTION = 0.625
#: Width in pixels of the left/right edge-brightness probes.
EDGE_PROBE_WIDTH = 20
#: Width in pixels of the left/right margin-density probes.
MARGIN_PROBE_WIDTH = 30
#: Below this many shadow pixels the positional features are meaningless and
#: fall back to a neutral 0.5.
MIN_SHADOW_PIXELS = 50
#: The principal-axis fit needs more support than the other positional features.
MIN_PIXELS_FOR_ORIENTATION = 100
#: Blur radius for the local-contrast (depth) shadow map.
DEPTH_BLUR_SIGMA = 20
#: The depth group looks below this fraction of the image height.
DEPTH_TOP_FRACTION = 0.58
#: Local-contrast value above which a pixel counts as shadow in the depth group.
DEPTH_SHADOW_THRESHOLD = 8

# Scale divisors, chosen so every feature lands roughly in [0, 1].
_GRAY_SCALE = 255.0
_DEPTH_SCALE = 100.0
_GRADIENT_SCALE = 50.0
_EPS = 1e-6


class Feature(IntEnum):
    """Named index into the 19-vector returned by :func:`extract_features`."""

    # -- row-median group ----------------------------------------------------
    LEFT_EDGE_BRIGHTNESS = 0
    RIGHT_EDGE_BRIGHTNESS = 1
    EDGE_BRIGHTNESS_RATIO = 2
    SHADOW_CENTROID_X = 3
    SHADOW_CENTROID_Y = 4
    SHADOW_SPREAD_X = 5
    SHADOW_SPREAD_Y = 6
    SHADOW_AREA_FRACTION = 7
    SHADOW_MASS_LEFT_FRACTION = 8
    SHADOW_PEAK_COLUMN = 9
    SHADOW_WEIGHTED_COLUMN = 10
    LEFT_MARGIN_DENSITY = 11
    RIGHT_MARGIN_DENSITY = 12
    SHADOW_PRINCIPAL_ANGLE = 13
    # -- depth group ---------------------------------------------------------
    DEPTH_MEAN = 14
    DEPTH_MAX = 15
    DEPTH_GRADIENT_X = 16
    DEPTH_FILL_RATIO = 17
    DEPTH_P90 = 18


#: Features that swap with their partner under a horizontal flip.
MIRROR_SWAPS: tuple[tuple[Feature, Feature], ...] = (
    (Feature.LEFT_EDGE_BRIGHTNESS, Feature.RIGHT_EDGE_BRIGHTNESS),
    (Feature.LEFT_MARGIN_DENSITY, Feature.RIGHT_MARGIN_DENSITY),
)

#: Features that map to ``1 - value`` under a horizontal flip, because they are
#: positions or mass fractions normalised along the x axis.
MIRROR_COMPLEMENTS: tuple[Feature, ...] = (
    Feature.SHADOW_CENTROID_X,
    Feature.SHADOW_MASS_LEFT_FRACTION,
    Feature.SHADOW_PEAK_COLUMN,
    Feature.SHADOW_WEIGHTED_COLUMN,
)


def extract_features(image: np.ndarray) -> np.ndarray:
    """Compute the 19 shadow descriptors for one RGB image.

    Args:
        image: ``(H, W, 3)`` or ``(H, W, 4)`` uint8 array. An alpha channel is
            dropped. Grayscale is the unweighted channel mean, matching how the
            published models were trained.

    Returns:
        ``float32`` array of shape ``(19,)``.
    """
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    gray = np.mean(image.astype(np.float32), axis=2)
    height, width = gray.shape

    f = np.zeros(NUM_FEATURES, dtype=np.float32)

    # --- Edge brightness: is the near road darker on the left or the right? --
    probe_top = int(height * EDGE_PROBE_TOP_FRACTION)
    f[Feature.LEFT_EDGE_BRIGHTNESS] = np.mean(gray[probe_top:, :EDGE_PROBE_WIDTH]) / _GRAY_SCALE
    f[Feature.RIGHT_EDGE_BRIGHTNESS] = np.mean(gray[probe_top:, -EDGE_PROBE_WIDTH:]) / _GRAY_SCALE
    f[Feature.EDGE_BRIGHTNESS_RATIO] = f[Feature.LEFT_EDGE_BRIGHTNESS] / (
        f[Feature.RIGHT_EDGE_BRIGHTNESS] + _EPS
    )

    # --- Row-median shadow mask over the road region ------------------------
    row_median = np.median(gray, axis=1, keepdims=True)
    shadow_mask = gray < (row_median * ROW_MEDIAN_SHADOW_RATIO)
    road_top = int(height * ROAD_TOP_FRACTION)
    road = shadow_mask[road_top:, :]

    ys, xs = np.where(road)
    if len(xs) > MIN_SHADOW_PIXELS:
        f[Feature.SHADOW_CENTROID_X] = np.mean(xs) / width
        f[Feature.SHADOW_CENTROID_Y] = (np.mean(ys) + road_top) / height
        f[Feature.SHADOW_SPREAD_X] = np.std(xs) / width
        f[Feature.SHADOW_SPREAD_Y] = np.std(ys) / height
        f[Feature.SHADOW_AREA_FRACTION] = len(xs) / (road.shape[0] * road.shape[1])

        left_mass = np.sum(road[:, : width // 2])
        right_mass = np.sum(road[:, width // 2 :])
        f[Feature.SHADOW_MASS_LEFT_FRACTION] = left_mass / (left_mass + right_mass + _EPS)

        column_density = np.sum(road, axis=0).astype(float)
        f[Feature.SHADOW_PEAK_COLUMN] = np.argmax(column_density) / width
        f[Feature.SHADOW_WEIGHTED_COLUMN] = (
            np.average(np.arange(width).astype(float), weights=column_density + _EPS) / width
        )

        margin_area = road.shape[0] * MARGIN_PROBE_WIDTH
        f[Feature.LEFT_MARGIN_DENSITY] = np.sum(road[:, :MARGIN_PROBE_WIDTH]) / margin_area
        f[Feature.RIGHT_MARGIN_DENSITY] = np.sum(road[:, -MARGIN_PROBE_WIDTH:]) / margin_area

        if len(xs) > MIN_PIXELS_FOR_ORIENTATION:
            covariance = np.cov(xs - np.mean(xs), ys - np.mean(ys))
            _, eigenvectors = np.linalg.eigh(covariance)
            # eigh returns ascending eigenvalues, so column 1 is the major axis.
            major = eigenvectors[:, 1]
            f[Feature.SHADOW_PRINCIPAL_ANGLE] = np.arctan2(major[1], major[0]) / np.pi
    else:
        # Not enough shadow to say anything: neutral values beat noise.
        f[Feature.SHADOW_CENTROID_X : Feature.DEPTH_MEAN] = 0.5

    # --- Local-contrast depth group -----------------------------------------
    blurred = gaussian_filter(gray, sigma=DEPTH_BLUR_SIGMA)
    contrast = blurred - gray
    depth_region = contrast[int(height * DEPTH_TOP_FRACTION) :, :]
    depth_mask = depth_region > DEPTH_SHADOW_THRESHOLD

    depth_ys, depth_xs = np.where(depth_mask)
    if len(depth_xs) > MIN_SHADOW_PIXELS:
        values = depth_region[depth_mask]
        f[Feature.DEPTH_MEAN] = np.mean(values) / _DEPTH_SCALE
        f[Feature.DEPTH_MAX] = np.max(values) / _DEPTH_SCALE
        f[Feature.DEPTH_GRADIENT_X] = (
            np.mean(np.abs(sobel(depth_region, axis=1))[depth_mask]) / _GRADIENT_SCALE
        )
        bounding_area = (np.max(depth_ys) - np.min(depth_ys) + 1) * (
            np.max(depth_xs) - np.min(depth_xs) + 1
        )
        f[Feature.DEPTH_FILL_RATIO] = len(depth_xs) / (bounding_area + _EPS)
        f[Feature.DEPTH_P90] = np.percentile(values, 90) / _DEPTH_SCALE

    return f


def mirror_features(f: np.ndarray) -> np.ndarray:
    """Transform a feature vector to match a horizontally flipped image.

    Horizontal flip is the only geometric augmentation this problem admits --
    it maps a left-edge sample onto a valid right-edge sample -- but it is only
    label-preserving if the hand-crafted features are flipped in step with the
    pixels. Getting this wrong teaches the network that a left-leaning shadow
    can belong to a person on the right, which is precisely the signal the
    features exist to provide.

    Returns a new array; the input is not modified.

    .. note::
       :attr:`Feature.SHADOW_PRINCIPAL_ANGLE` is passed through unchanged, which
       is not strictly correct: a mirrored axis should negate its angle. This
       matches the code that produced the published leaderboard scores and is
       kept for reproducibility. ``docs/method.md`` discusses the impact.
    """
    out = np.asarray(f, dtype=np.float32).copy()

    for a, b in MIRROR_SWAPS:
        out[a], out[b] = out[b], out[a]

    # Recompute the ratio from the already-swapped brightnesses so it inverts.
    out[Feature.EDGE_BRIGHTNESS_RATIO] = out[Feature.LEFT_EDGE_BRIGHTNESS] / (
        out[Feature.RIGHT_EDGE_BRIGHTNESS] + _EPS
    )

    for index in MIRROR_COMPLEMENTS:
        out[index] = 1.0 - out[index]

    return out
