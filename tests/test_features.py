"""The mirror map is the part of this codebase most likely to break silently.

Flip augmentation doubles the effective dataset and is the single most valuable
augmentation available here, but only if the hand-crafted features are flipped
in step with the pixels. If they are not, the network is shown a left-leaning
shadow labelled "person on the right" half the time -- which is worse than not
augmenting, and produces no error, no warning and a quietly worse model.

So the tests below check the property directly: computing features on a flipped
image should equal mirroring the features of the original.
"""

from __future__ import annotations

import numpy as np
import pytest

from shadow_detection.features import (
    MIRROR_COMPLEMENTS,
    MIRROR_SWAPS,
    NUM_FEATURES,
    Feature,
    extract_features,
    mirror_features,
)
from tests.conftest import make_road_image

# Positions normalised by width pick up an off-by-one under mirroring: a pixel
# at column x lands at W-1-x, so a normalised position maps to
# (W-1)/W - x/W rather than exactly 1 - x/W. With W=720 that is a 1/720 gap.
POSITION_TOLERANCE = 1.5 / 720


def test_shape_and_dtype(road_image):
    values = extract_features(road_image)
    assert values.shape == (NUM_FEATURES,)
    assert values.dtype == np.float32
    assert np.isfinite(values).all()


def test_alpha_channel_is_dropped(road_image):
    rgba = np.dstack([road_image, np.full(road_image.shape[:2], 255, dtype=np.uint8)])
    assert extract_features(rgba) == pytest.approx(extract_features(road_image))


def test_deterministic(road_image):
    assert extract_features(road_image) == pytest.approx(extract_features(road_image))


class TestMirrorConsistency:
    """Features of a flipped image == mirror_features(features of the original)."""

    @pytest.fixture
    def pair(self):
        image = make_road_image(shadow_centre_x=180)
        direct = extract_features(image[:, ::-1, :])
        mapped = mirror_features(extract_features(image))
        return direct, mapped

    @pytest.mark.parametrize("swap", MIRROR_SWAPS, ids=lambda s: f"{s[0].name}<->{s[1].name}")
    def test_swapped_features_match(self, pair, swap):
        direct, mapped = pair
        for index in swap:
            assert mapped[index] == pytest.approx(direct[index], rel=1e-4, abs=1e-6)

    @pytest.mark.parametrize("index", MIRROR_COMPLEMENTS, ids=lambda f: f.name)
    def test_complemented_features_match(self, pair, index):
        direct, mapped = pair
        assert mapped[index] == pytest.approx(direct[index], abs=POSITION_TOLERANCE)

    @pytest.mark.parametrize(
        "index",
        [
            Feature.SHADOW_CENTROID_Y,
            Feature.SHADOW_SPREAD_X,
            Feature.SHADOW_SPREAD_Y,
            Feature.SHADOW_AREA_FRACTION,
            Feature.DEPTH_MEAN,
            Feature.DEPTH_MAX,
            Feature.DEPTH_GRADIENT_X,
            Feature.DEPTH_FILL_RATIO,
            Feature.DEPTH_P90,
        ],
        ids=lambda f: f.name,
    )
    def test_invariant_features_are_left_alone(self, pair, index):
        """These describe size, depth or vertical position, none of which a
        horizontal flip changes. The mirror map must not touch them."""
        direct, mapped = pair
        assert mapped[index] == pytest.approx(direct[index], rel=1e-3, abs=1e-5)

    def test_edge_brightness_ratio_inverts(self, pair):
        direct, mapped = pair
        assert mapped[Feature.EDGE_BRIGHTNESS_RATIO] == pytest.approx(
            direct[Feature.EDGE_BRIGHTNESS_RATIO], rel=1e-4
        )

    def test_principal_angle_is_a_known_exception(self, pair):
        """Documented shortcoming, asserted so it cannot regress unnoticed.

        A mirrored principal axis should negate its angle. The published models
        were trained without that correction, so :func:`mirror_features` passes
        the angle through and the mapped value does *not* match the directly
        computed one. If someone fixes the mirror map, this test fails and
        points at the docs entry that has to change with it.
        """
        direct, mapped = pair
        assert mapped[Feature.SHADOW_PRINCIPAL_ANGLE] != pytest.approx(
            direct[Feature.SHADOW_PRINCIPAL_ANGLE], abs=1e-3
        )


def test_mirror_is_an_involution_except_for_the_angle(road_image):
    """Mirroring twice returns the original, which is what makes the flip
    augmentation composable with TTA."""
    values = extract_features(road_image)
    twice = mirror_features(mirror_features(values))
    for feature in Feature:
        if feature is Feature.SHADOW_PRINCIPAL_ANGLE:
            continue
        assert twice[feature] == pytest.approx(values[feature], rel=1e-5, abs=1e-6)


def test_mirror_does_not_mutate_its_input(road_image):
    values = extract_features(road_image)
    before = values.copy()
    mirror_features(values)
    assert values == pytest.approx(before)


class TestFallbackPath:
    def test_flat_image_gets_neutral_positions(self, flat_image):
        """With no shadow to measure, the positional block falls back to 0.5
        rather than reporting a centroid of zero, which would read as
        'shadow hard against the left edge'."""
        values = extract_features(flat_image)
        neutral = values[Feature.SHADOW_CENTROID_X : Feature.DEPTH_MEAN]
        assert neutral == pytest.approx(0.5)

    def test_flat_image_has_no_depth_signal(self, flat_image):
        values = extract_features(flat_image)
        assert values[Feature.DEPTH_MEAN : NUM_FEATURES] == pytest.approx(0.0)


def test_shadow_position_tracks_the_blob():
    """A sanity check that the descriptors measure what they claim: moving the
    shadow right must move the centroid and the mass split with it."""
    left = extract_features(make_road_image(shadow_centre_x=120, edge_strip=None))
    right = extract_features(make_road_image(shadow_centre_x=600, edge_strip=None))
    assert left[Feature.SHADOW_CENTROID_X] < right[Feature.SHADOW_CENTROID_X]
    assert left[Feature.SHADOW_MASS_LEFT_FRACTION] > right[Feature.SHADOW_MASS_LEFT_FRACTION]


def test_margin_density_responds_to_a_shadow_in_the_margin():
    """The margin probes only look at the outer 30 pixels, so a shadow has to
    actually reach the frame edge to register."""
    in_margin = extract_features(make_road_image(shadow_centre_x=20, edge_strip=None))
    mid_frame = extract_features(make_road_image(shadow_centre_x=400, edge_strip=None))
    assert in_margin[Feature.LEFT_MARGIN_DENSITY] > mid_frame[Feature.LEFT_MARGIN_DENSITY]
    assert mid_frame[Feature.LEFT_MARGIN_DENSITY] == pytest.approx(0.0, abs=1e-6)


def test_edge_brightness_detects_which_margin_is_darker():
    darker_left = extract_features(make_road_image(edge_strip="left"))
    darker_right = extract_features(make_road_image(edge_strip="right"))
    assert darker_left[Feature.LEFT_EDGE_BRIGHTNESS] < darker_left[Feature.RIGHT_EDGE_BRIGHTNESS]
    assert darker_right[Feature.RIGHT_EDGE_BRIGHTNESS] < darker_right[Feature.LEFT_EDGE_BRIGHTNESS]
    assert (
        darker_left[Feature.EDGE_BRIGHTNESS_RATIO]
        < 1.0
        < darker_right[Feature.EDGE_BRIGHTNESS_RATIO]
    )


def test_peak_column_mirrors_only_up_to_argmax_tie_breaking():
    """An honest limit of the mirror map, worth pinning down.

    ``SHADOW_PEAK_COLUMN`` is an ``argmax`` over per-column shadow-pixel counts.
    ``argmax`` returns the *first* maximal index, so on a rectangular shadow --
    where every column of the shadow has an identical count -- the original
    picks the left end of the plateau and the flipped image picks what was the
    right end. The two disagree by the plateau's width.

    Real shadows taper, so this is a corner case rather than a live problem, but
    it is the reason the equivariance test uses a tapered fixture.
    """
    plateau = make_road_image(shadow_width=120, edge_strip=None, taper=False)
    direct = extract_features(plateau[:, ::-1, :])
    mapped = mirror_features(extract_features(plateau))
    disagreement = abs(
        float(mapped[Feature.SHADOW_PEAK_COLUMN] - direct[Feature.SHADOW_PEAK_COLUMN])
    )
    assert disagreement == pytest.approx(120 / 720, abs=2 / 720)
