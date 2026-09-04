"""Inference: TTA, ensembling, and the submission guard rails.

The guard rails in :func:`~shadow_detection.predict.write_submission` exist
because of a specific hackathon failure mode: a submission with the right rows
in the wrong order, or with an id quietly filled by a left-join, is accepted by
the leaderboard and scores near zero. It looks like a modelling problem and it
is not. Checking locally costs nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from shadow_detection.data import TargetStats
from shadow_detection.geometry import LEFT, REGRESSION_TARGETS, RIGHT, FrameSize
from shadow_detection.predict import (
    DIRECTION_ABSTAIN,
    SUBMISSION_COLUMNS,
    RawPrediction,
    average_predictions,
    predict_with_model,
    read_submission_ids,
    to_submission,
    write_submission,
)

FRAME = FrameSize(720, 480)

# Identity statistics, so a regression value of 200 means 200 pixels and the
# reconstruction arithmetic is readable in the assertions below.
IDENTITY_STATS = TargetStats({key: (0.0, 1.0) for key in REGRESSION_TARGETS})


def raw(
    name="image_0",
    distance=200.0,
    width=80.0,
    height=192.0,
    y_center=310.0,
    side_probs=(0.9, 0.1),
    direction_probs=(0.2, 0.8),
):
    return RawPrediction(
        name=name,
        regression=np.array([distance, width, height, y_center], dtype=np.float32),
        side_probs=np.array(side_probs, dtype=np.float32),
        direction_probs=np.array(direction_probs, dtype=np.float32),
    )


class TestToSubmission:
    def test_left_side_box_is_built_outwards_from_the_edge(self):
        table = to_submission([raw(side_probs=(0.9, 0.1))], IDENTITY_STATS, FRAME)
        row = table.iloc[0]
        assert row["xmin"] == pytest.approx(-200.0)
        assert row["xmax"] == pytest.approx(-120.0)
        assert row["ymin"] == pytest.approx(310.0 - 96.0)
        assert row["ymax"] == pytest.approx(310.0 + 96.0)

    def test_right_side_box_is_relative_to_the_frame_width(self):
        table = to_submission([raw(side_probs=(0.1, 0.9))], IDENTITY_STATS, FRAME)
        row = table.iloc[0]
        assert row["xmax"] == pytest.approx(920.0)
        assert row["xmin"] == pytest.approx(840.0)

    def test_column_order(self):
        table = to_submission([raw()], IDENTITY_STATS, FRAME)
        assert list(table.columns) == SUBMISSION_COLUMNS

    def test_confident_direction_is_emitted(self):
        table = to_submission([raw(direction_probs=(0.1, 0.9))], IDENTITY_STATS, FRAME)
        assert table.iloc[0]["direction"] == 1

    def test_unconfident_direction_abstains(self):
        """Validation direction accuracy never rose meaningfully above chance,
        so a coin flip was strictly worse than declining to answer."""
        table = to_submission([raw(direction_probs=(0.45, 0.55))], IDENTITY_STATS, FRAME)
        assert table.iloc[0]["direction"] == DIRECTION_ABSTAIN

    def test_threshold_of_zero_always_commits(self):
        table = to_submission(
            [raw(direction_probs=(0.49, 0.51))], IDENTITY_STATS, FRAME, direction_threshold=0.0
        )
        assert table.iloc[0]["direction"] == 1

    def test_target_stats_are_applied(self):
        stats = TargetStats(
            {
                "distance_from_edge": (208.59, 45.14),
                "bbox_width": (80.87, 29.86),
                "bbox_height": (172.91, 34.67),
                "y_center": (309.34, 14.27),
            }
        )
        # A zero-vector prediction is "the training mean" for every target.
        table = to_submission(
            [raw(distance=0.0, width=0.0, height=0.0, y_center=0.0)], stats, FRAME
        )
        row = table.iloc[0]
        assert row["xmin"] == pytest.approx(-208.59, abs=1e-3)
        assert row["xmax"] - row["xmin"] == pytest.approx(80.87, abs=1e-3)
        assert row["ymax"] - row["ymin"] == pytest.approx(172.91, abs=1e-3)


class TestAveragePredictions:
    def test_averages_every_field(self):
        a = raw(distance=180.0, side_probs=(0.8, 0.2), direction_probs=(0.3, 0.7))
        b = raw(distance=220.0, side_probs=(0.6, 0.4), direction_probs=(0.5, 0.5))
        merged = average_predictions([[a], [b]])[0]
        assert merged.regression[0] == pytest.approx(200.0)
        assert merged.side_probs == pytest.approx([0.7, 0.3])
        assert merged.direction_probs == pytest.approx([0.4, 0.6])

    def test_averaging_happens_before_reconstruction(self):
        """Two models that disagree about the side must not produce a box
        spanning the frame. Averaging in decomposed space and taking the argmax
        once means the ensemble commits to one side."""
        left = raw(distance=200.0, side_probs=(0.9, 0.1))
        right = raw(distance=200.0, side_probs=(0.4, 0.6))
        merged = average_predictions([[left], [right]])
        table = to_submission(merged, IDENTITY_STATS, FRAME)
        row = table.iloc[0]
        assert int(merged[0].side_probs.argmax()) == LEFT
        assert row["xmax"] < 0  # entirely off the left edge, not straddling

    def test_rejects_mismatched_image_order(self):
        with pytest.raises(ValueError, match="disagree on image order"):
            average_predictions([[raw(name="image_0")], [raw(name="image_1")]])

    def test_rejects_different_lengths(self):
        with pytest.raises(ValueError, match="same number of images"):
            average_predictions([[raw()], [raw(), raw(name="image_1")]])

    def test_rejects_nothing(self):
        with pytest.raises(ValueError, match="nothing to average"):
            average_predictions([])


class TestWriteSubmission:
    def test_writes_a_valid_file(self, tmp_path, submission_frame):
        ids = ["image_1", "image_2"]
        table = submission_frame(ids, [-200.0, -180.0])
        path = tmp_path / "submission.csv"
        write_submission(table, path, ids)

        assert path.exists()
        assert read_submission_ids(path) == ids

    def test_rejects_reordered_ids(self, tmp_path, submission_frame):
        """Row order is part of the contract; a shuffled submission scores
        every row against the wrong image."""
        table = submission_frame(["image_2", "image_1"], [-200.0, -180.0])
        with pytest.raises(ValueError, match="do not match the sample submission"):
            write_submission(table, tmp_path / "s.csv", ["image_1", "image_2"])

    def test_rejects_missing_ids(self, tmp_path, submission_frame):
        table = submission_frame(["image_1"], [-200.0])
        with pytest.raises(ValueError, match="do not match the sample submission"):
            write_submission(table, tmp_path / "s.csv", ["image_1", "image_2"])

    def test_rejects_nan_coordinates(self, tmp_path, submission_frame):
        """A left-join against the sample submission fills unmatched ids with
        NaN. That has to fail loudly rather than reach the leaderboard."""
        table = submission_frame(["image_1"], [np.nan])
        with pytest.raises(ValueError, match="missing coordinates"):
            write_submission(table, tmp_path / "s.csv", ["image_1"])

    def test_rejects_wrong_columns(self, tmp_path, submission_frame):
        table = submission_frame(["image_1"], [-200.0]).drop(columns=["direction"])
        with pytest.raises(ValueError, match="columns must be"):
            write_submission(table, tmp_path / "s.csv", ["image_1"])


class TestPredictWithModel:
    """End-to-end inference on a randomly initialised model.

    The numbers are meaningless; the shapes, the id ordering and the TTA
    plumbing are what is under test.
    """

    @pytest.fixture
    def model(self):
        from shadow_detection.model import ShadowNet

        return ShadowNet(pretrained=False).eval()

    @pytest.fixture
    def test_dir(self, tmp_path):
        from PIL import Image

        from tests.conftest import make_road_image

        for index in range(3):
            Image.fromarray(make_road_image(shadow_centre_x=150 + 150 * index)).save(
                tmp_path / f"image_{index}.png"
            )
        return tmp_path

    def test_returns_one_prediction_per_id_in_order(self, model, test_dir):
        ids = ["image_2", "image_0", "image_1"]
        predictions = predict_with_model(model, ids, test_dir, input_size=(64, 64), tta=False)
        assert [p.name for p in predictions] == ids
        assert all(p.regression.shape == (4,) for p in predictions)
        assert all(p.side_probs.shape == (2,) for p in predictions)

    def test_probabilities_sum_to_one(self, model, test_dir):
        predictions = predict_with_model(
            model, ["image_0"], test_dir, input_size=(64, 64), tta=True
        )
        assert predictions[0].side_probs.sum() == pytest.approx(1.0, abs=1e-5)
        assert predictions[0].direction_probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_batching_does_not_change_the_result(self, model, test_dir):
        """Eval-mode batch norm uses running statistics, so grouping is
        irrelevant to the output. Asserting it means the batched fast path
        cannot silently diverge from the one-at-a-time original."""
        ids = ["image_0", "image_1", "image_2"]
        one_at_a_time = predict_with_model(
            model, ids, test_dir, input_size=(64, 64), tta=True, batch_size=1
        )
        batched = predict_with_model(
            model, ids, test_dir, input_size=(64, 64), tta=True, batch_size=8
        )
        for a, b in zip(one_at_a_time, batched, strict=True):
            assert a.regression == pytest.approx(b.regression, abs=1e-5)
            assert a.side_probs == pytest.approx(b.side_probs, abs=1e-5)

    def test_missing_image_is_reported_with_its_id(self, model, test_dir):
        with pytest.raises(FileNotFoundError, match="image_99"):
            predict_with_model(model, ["image_99"], test_dir, input_size=(64, 64))

    def test_tta_changes_the_prediction(self, model, test_dir):
        """If TTA were wired up wrong -- flipping nothing, or averaging a
        tensor with itself -- this would silently pass through unchanged."""
        plain = predict_with_model(model, ["image_0"], test_dir, input_size=(64, 64), tta=False)
        augmented = predict_with_model(model, ["image_0"], test_dir, input_size=(64, 64), tta=True)
        assert plain[0].regression != pytest.approx(augmented[0].regression, abs=1e-6)

    def test_side_probabilities_are_reversed_for_the_flipped_pass(self, model, test_dir):
        """The flipped image describes a mirrored world, so its side
        probabilities have to be un-flipped before averaging. Averaging them
        raw would pull every prediction towards 50/50 -- and a broken TTA that
        does exactly that is indistinguishable from a well-calibrated model
        unless you check.
        """
        ids = ["image_0"]
        with_tta = predict_with_model(model, ids, test_dir, input_size=(64, 64), tta=True)[0]

        # Recreate the averaging by hand from the two single passes.
        import torch
        from PIL import Image
        from torch.nn import functional as F

        from shadow_detection.features import extract_features, mirror_features
        from shadow_detection.predict import _build_transform

        transform = _build_transform((64, 64))
        image = Image.open(test_dir / "image_0.png").convert("RGB")
        features = extract_features(np.array(image))
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            side_a, _, _ = model(tensor, torch.from_numpy(features).float().unsqueeze(0))
            side_b, _, _ = model(
                torch.flip(tensor, dims=[3]),
                torch.from_numpy(mirror_features(features)).float().unsqueeze(0),
            )
        probs_a = F.softmax(side_a.float(), dim=1).numpy()[0]
        probs_b = F.softmax(side_b.float(), dim=1).numpy()[0]

        expected = (probs_a + probs_b[::-1]) / 2.0
        assert with_tta.side_probs == pytest.approx(expected, abs=1e-5)
        # And the naive version, without reversing, is genuinely different.
        assert with_tta.side_probs != pytest.approx((probs_a + probs_b) / 2.0, abs=1e-4)


def test_left_and_right_are_the_documented_class_indices():
    """The side head's class order is load-bearing: the TTA reversal and the
    box reconstruction both assume index 0 is left."""
    assert (LEFT, RIGHT) == (0, 1)


class TestOutputClamps:
    """A regression head can emit values no box can have.

    A negative distance places a left-edge person *inside* the frame; a
    two-pixel-tall box scores zero against anything. Both produce a submission
    row that looks numerically fine and is silently worthless, so the floors
    are applied before reconstruction.
    """

    def test_negative_distance_is_floored_at_the_frame_edge(self):
        table = to_submission([raw(distance=-150.0, side_probs=(0.9, 0.1))], IDENTITY_STATS, FRAME)
        row = table.iloc[0]
        assert row["xmin"] == pytest.approx(0.0)
        # Still built outwards from the edge, so the box does not invert.
        assert row["xmax"] > row["xmin"]

    def test_negative_distance_on_the_right_edge_too(self):
        table = to_submission([raw(distance=-150.0, side_probs=(0.1, 0.9))], IDENTITY_STATS, FRAME)
        row = table.iloc[0]
        assert row["xmax"] == pytest.approx(float(FRAME.width))

    def test_degenerate_width_and_height_are_floored(self):
        table = to_submission([raw(width=1.0, height=3.0)], IDENTITY_STATS, FRAME)
        row = table.iloc[0]
        assert row["xmax"] - row["xmin"] == pytest.approx(10.0)
        assert row["ymax"] - row["ymin"] == pytest.approx(50.0)

    def test_clamps_never_bind_on_a_plausible_prediction(self):
        """The floors are slack against the real data -- the smallest annotated
        box is 26.8 x 116.5 px -- so they must not perturb ordinary output."""
        table = to_submission(
            [raw(distance=208.0, width=81.0, height=173.0, y_center=309.0)],
            IDENTITY_STATS,
            FRAME,
        )
        row = table.iloc[0]
        assert row["xmin"] == pytest.approx(-208.0)
        assert row["xmax"] - row["xmin"] == pytest.approx(81.0)
        assert row["ymax"] - row["ymin"] == pytest.approx(173.0)

    def test_clamped_boxes_are_never_inverted(self):
        """Whatever the head emits, the submission must not contain a box with
        xmin >= xmax -- the exact defect that made the v1 submission useless."""
        wild = [
            raw(distance=-500.0, width=-40.0, height=-90.0, side_probs=(0.9, 0.1)),
            raw(distance=-500.0, width=-40.0, height=-90.0, side_probs=(0.1, 0.9)),
        ]
        table = to_submission(wild, IDENTITY_STATS, FRAME)
        assert (table["xmin"] < table["xmax"]).all()
        assert (table["ymin"] < table["ymax"]).all()
