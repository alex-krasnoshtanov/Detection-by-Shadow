"""Submission blending, including the failure mode it can produce.

Averaging box coordinates across models is the crudest possible ensembling and
the only one available when all you have is the other team's CSV. Its specific
danger is that averaging across a *side* disagreement lands the box inside the
frame, overlapping neither answer and scoring zero on both. That never fired on
the submissions actually blended -- side classification was solved -- but the
warning is cheap insurance.
"""

from __future__ import annotations

import numpy as np
import pytest

from shadow_detection.blend import blend, parse_source
from shadow_detection.geometry import FrameSize
from shadow_detection.predict import DIRECTION_ABSTAIN, SUBMISSION_COLUMNS

FRAME = FrameSize(720, 480)
IDS = ["image_1", "image_2", "image_3"]


@pytest.fixture
def write_csv(tmp_path, submission_frame):
    def _write(name, xmin, direction=-1, ids=None):
        table = submission_frame(ids or IDS, xmin, direction)
        path = tmp_path / name
        table.to_csv(path, index=False)
        return path

    return _write


class TestParseSource:
    def test_bare_path_defaults_to_weight_one(self):
        path, weight = parse_source("submission.csv")
        assert path.name == "submission.csv"
        assert weight == 1.0

    def test_weighted_path(self):
        path, weight = parse_source("runs/v5/submission.csv=3")
        assert path.name == "submission.csv"
        assert weight == 3.0

    def test_windows_path_with_a_weight(self):
        """``rpartition`` on ``=`` rather than ``split`` means a path
        containing an equals sign still parses."""
        path, weight = parse_source("out/a=b/submission.csv=2.5")
        assert str(path).replace("\\", "/") == "out/a=b/submission.csv"
        assert weight == 2.5


class TestBlend:
    def test_equal_weights_are_a_plain_mean(self, write_csv):
        a = write_csv("a.csv", [-200.0, -300.0, -400.0])
        b = write_csv("b.csv", [-100.0, -200.0, -300.0])
        result = blend([(a, 1.0), (b, 1.0)], frame=FRAME)
        assert result["xmin"].to_numpy() == pytest.approx([-150.0, -250.0, -350.0])

    def test_weights_shift_the_result(self, write_csv):
        """The final hackathon submission weighted this repository's ensemble
        0.7 against the other model's 0.3."""
        ours = write_csv("ours.csv", [-200.0, -200.0, -200.0])
        theirs = write_csv("theirs.csv", [-300.0, -300.0, -300.0])
        result = blend([(ours, 0.7), (theirs, 0.3)], frame=FRAME)
        assert result["xmin"].to_numpy() == pytest.approx([-230.0] * 3)

    def test_weights_are_normalised(self, write_csv):
        a = write_csv("a.csv", [-200.0] * 3)
        b = write_csv("b.csv", [-300.0] * 3)
        as_fractions = blend([(a, 0.7), (b, 0.3)], frame=FRAME)
        as_counts = blend([(a, 7.0), (b, 3.0)], frame=FRAME)
        assert as_fractions["xmin"].to_numpy() == pytest.approx(as_counts["xmin"].to_numpy())

    def test_single_source_is_a_copy(self, write_csv):
        a = write_csv("a.csv", [-200.0, -300.0, -400.0])
        result = blend([(a, 1.0)], frame=FRAME)
        assert result["xmin"].to_numpy() == pytest.approx([-200.0, -300.0, -400.0])

    def test_column_order_and_id_order_come_from_the_first_source(self, write_csv):
        a = write_csv("a.csv", [-200.0, -300.0, -400.0])
        b = write_csv("b.csv", [-100.0, -200.0, -300.0], ids=list(reversed(IDS)))
        result = blend([(a, 1.0), (b, 1.0)], frame=FRAME)
        assert list(result.columns) == SUBMISSION_COLUMNS
        assert result["id"].tolist() == IDS

    def test_rows_are_matched_by_id_not_position(self, write_csv):
        """A source listing the same ids in a different order must still be
        aligned per image, not blended row-by-row."""
        a = write_csv("a.csv", [-100.0, -200.0, -300.0])
        b = write_csv("b.csv", [-300.0, -200.0, -100.0], ids=list(reversed(IDS)))
        result = blend([(a, 1.0), (b, 1.0)], frame=FRAME)
        # b's image_1 is -100, matching a's, so every average is unchanged.
        assert result["xmin"].to_numpy() == pytest.approx([-100.0, -200.0, -300.0])

    def test_rejects_sources_covering_different_ids(self, write_csv):
        a = write_csv("a.csv", [-200.0] * 3)
        b = write_csv("b.csv", [-200.0] * 2, ids=["image_1", "image_9"])
        with pytest.raises(ValueError, match="does not cover the same ids"):
            blend([(a, 1.0), (b, 1.0)], frame=FRAME)

    def test_rejects_a_missing_column(self, tmp_path, submission_frame):
        path = tmp_path / "broken.csv"
        submission_frame(IDS, [-200.0] * 3).drop(columns=["ymax"]).to_csv(path, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            blend([(path, 1.0)], frame=FRAME)

    def test_rejects_no_sources(self):
        with pytest.raises(ValueError, match="at least one source"):
            blend([], frame=FRAME)

    def test_rejects_zero_total_weight(self, write_csv):
        a = write_csv("a.csv", [-200.0] * 3)
        with pytest.raises(ValueError, match="sum to something positive"):
            blend([(a, 0.0)], frame=FRAME)


class TestDirectionPolicy:
    def test_abstain_emits_minus_one_everywhere(self, write_csv):
        a = write_csv("a.csv", [-200.0] * 3, direction=1)
        b = write_csv("b.csv", [-200.0] * 3, direction=0)
        result = blend([(a, 1.0), (b, 1.0)], direction="abstain", frame=FRAME)
        assert (result["direction"] == DIRECTION_ABSTAIN).all()

    def test_first_copies_the_leading_source(self, write_csv):
        a = write_csv("a.csv", [-200.0] * 3, direction=[1, 0, 1])
        b = write_csv("b.csv", [-200.0] * 3, direction=0)
        result = blend([(a, 3.0), (b, 1.0)], direction="first", frame=FRAME)
        assert result["direction"].tolist() == [1, 0, 1]

    def test_majority_uses_the_weights(self, write_csv):
        a = write_csv("a.csv", [-200.0] * 3, direction=[1, 1, 0])
        b = write_csv("b.csv", [-200.0] * 3, direction=[0, 1, 0])
        result = blend([(a, 3.0), (b, 1.0)], direction="majority", frame=FRAME)
        assert result["direction"].tolist() == [1, 1, 0]

    def test_majority_abstains_on_a_tie(self, write_csv):
        a = write_csv("a.csv", [-200.0] * 3, direction=1)
        b = write_csv("b.csv", [-200.0] * 3, direction=0)
        result = blend([(a, 1.0), (b, 1.0)], direction="majority", frame=FRAME)
        assert (result["direction"] == DIRECTION_ABSTAIN).all()

    def test_majority_ignores_abstaining_sources(self, write_csv):
        a = write_csv("a.csv", [-200.0] * 3, direction=-1)
        b = write_csv("b.csv", [-200.0] * 3, direction=[1, 1, 0])
        result = blend([(a, 5.0), (b, 1.0)], direction="majority", frame=FRAME)
        assert result["direction"].tolist() == [1, 1, 0]

    def test_majority_abstains_when_nobody_votes(self, write_csv):
        a = write_csv("a.csv", [-200.0] * 3, direction=-1)
        result = blend([(a, 1.0)], direction="majority", frame=FRAME)
        assert (result["direction"] == DIRECTION_ABSTAIN).all()


class TestDegenerateBoxWarning:
    def test_warns_when_sources_disagree_about_the_side(self, write_csv, capsys):
        """The failure this whole module has to be careful about: averaging a
        left-edge box with a right-edge box yields a box inside the frame,
        which overlaps neither answer."""
        left = write_csv("left.csv", [-250.0] * 3)
        right = write_csv("right.csv", [800.0] * 3)
        result = blend([(left, 1.0), (right, 1.0)], frame=FRAME)

        assert "not fully outside the frame" in capsys.readouterr().out
        # The average of a box off the left edge and one off the right edge
        # sits in the middle of the frame, where no annotated person ever is.
        assert (result["xmin"] > 0).all()
        assert (result["xmax"] < FRAME.width).all()

    def test_silent_when_sources_agree(self, write_csv, capsys):
        a = write_csv("a.csv", [-200.0] * 3)
        b = write_csv("b.csv", [-260.0] * 3)
        blend([(a, 1.0), (b, 1.0)], frame=FRAME)
        assert "warning" not in capsys.readouterr().out

    def test_warns_on_inverted_boxes(self, tmp_path, submission_frame, capsys):
        table = submission_frame(IDS, [-200.0] * 3)
        table["xmax"] = table["xmin"] - 10.0
        path = tmp_path / "inverted.csv"
        table.to_csv(path, index=False)
        blend([(path, 1.0)], frame=FRAME)
        assert "xmin >= xmax" in capsys.readouterr().out


def test_blend_preserves_dtypes_for_a_valid_submission(write_csv):
    a = write_csv("a.csv", [-200.0] * 3, direction=1)
    result = blend([(a, 1.0)], direction="majority", frame=FRAME)
    assert result["direction"].dtype == np.dtype(int)
    assert result["xmin"].dtype == np.dtype(float)
