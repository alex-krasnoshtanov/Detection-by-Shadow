"""The reparameterisation has to be a lossless bijection, or the whole approach
is unsound: the model predicts in decomposed space and is scored in box space.
"""

from __future__ import annotations

import pytest

from shadow_detection.geometry import (
    LEFT,
    RIGHT,
    FrameSize,
    decompose,
    iou,
    reconstruct,
)

FRAME = FrameSize(720, 480)

# Real annotation shapes taken from the challenge training set, covering both
# edges and the extremes of the observed coordinate ranges.
REAL_BOXES = [
    (-228.87714385986328, 223.6187400817871, -110.61487913131714, 416.3587894439697),
    (-307.3855948448181, 215.6910743713379, -243.78775835037231, 438.10311102867126),
    (-253.66890907287598, 215.60472869873047, -174.38595414161682, 434.19680976867676),
    (-359.06, 211.87, -252.86, 345.60),  # xmin_norm near the observed minimum
    (766.5, 220.0, 845.2, 400.0),  # right edge, close in
    (933.0, 218.0, 1038.7, 464.4),  # right edge, xmax_norm near the maximum
]


@pytest.mark.parametrize("box", REAL_BOXES)
def test_decompose_reconstruct_round_trip(box):
    xmin, ymin, xmax, ymax = box
    target = decompose(xmin, ymin, xmax, ymax, FRAME)
    result = reconstruct(
        side=target.side,
        distance_from_edge=target.distance_from_edge,
        bbox_width=target.bbox_width,
        bbox_height=target.bbox_height,
        y_center=target.y_center,
        frame=FRAME,
    )
    assert result == pytest.approx(box, abs=1e-9)


@pytest.mark.parametrize("box", REAL_BOXES)
def test_distance_from_edge_is_positive(box):
    """A negative distance would mean the person is inside the frame, which the
    dataset never contains and the reconstruction cannot express."""
    assert decompose(*box, FRAME).distance_from_edge > 0


def test_side_follows_the_sign_of_xmin():
    assert decompose(-100.0, 200.0, -20.0, 400.0, FRAME).side == LEFT
    assert decompose(800.0, 200.0, 880.0, 400.0, FRAME).side == RIGHT


def test_distance_is_measured_to_the_outer_corner():
    """The anchor is the corner further from the frame, so the box is built
    outwards from the edge and the width extends back towards it."""
    left = decompose(-250.0, 200.0, -150.0, 400.0, FRAME)
    assert left.distance_from_edge == pytest.approx(250.0)
    assert left.bbox_width == pytest.approx(100.0)

    right = decompose(820.0, 200.0, 900.0, 400.0, FRAME)
    assert right.distance_from_edge == pytest.approx(180.0)  # 900 - 720
    assert right.bbox_width == pytest.approx(80.0)


def test_regression_tuple_order_matches_the_documented_order():
    target = decompose(-250.0, 200.0, -150.0, 400.0, FRAME)
    assert target.as_regression_tuple() == (
        target.distance_from_edge,
        target.bbox_width,
        target.bbox_height,
        target.y_center,
    )


def test_frame_width_is_not_hardcoded():
    """Reconstruction on the right edge is relative to the frame width, so a
    different frame has to move the box."""
    narrow = reconstruct(RIGHT, 100.0, 80.0, 200.0, 300.0, FrameSize(320, 240))
    wide = reconstruct(RIGHT, 100.0, 80.0, 200.0, 300.0, FrameSize(720, 480))
    assert wide[0] - narrow[0] == pytest.approx(400.0)


class TestIou:
    def test_identical_boxes(self):
        box = (-200.0, 220.0, -120.0, 380.0)
        assert iou(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert iou((-200.0, 220.0, -120.0, 380.0), (100.0, 220.0, 180.0, 380.0)) == 0.0

    def test_half_overlap(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 0.0, 15.0, 10.0)
        # intersection 50, union 150
        assert iou(a, b) == pytest.approx(50.0 / 150.0)

    def test_inside_out_box_still_scores(self):
        """A regression head can emit xmax < xmin. Sorting the corners first
        means such a prediction is scored on the box it describes rather than
        silently producing a negative area."""
        upright = (0.0, 0.0, 10.0, 10.0)
        inverted = (10.0, 10.0, 0.0, 0.0)
        assert iou(upright, inverted) == pytest.approx(1.0)

    def test_zero_area_box(self):
        assert iou((5.0, 5.0, 5.0, 5.0), (0.0, 0.0, 10.0, 10.0)) == 0.0
