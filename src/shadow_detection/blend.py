"""Weighted blending of finished submission CSVs.

The last hour of the hackathon was spent here rather than in training. Three
people had each developed a model -- this repository's seed ensemble among them
-- and averaging their box coordinates is the cheapest possible way to combine
them: it needs no shared code, no shared checkpoints and no retraining, just
the CSVs.

It is also the crudest. Averaging coordinates across models that disagree about
which *side* the person is on lands the box in the middle of the frame -- where
no annotated person ever is -- overlapping neither of the two answers it came
from. That never fired on the submissions actually blended, because side
classification was solved (100% validation accuracy), but :func:`blend` warns
when it happens rather than quietly emitting nonsense.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from shadow_detection.geometry import FrameSize
from shadow_detection.predict import DIRECTION_ABSTAIN, SUBMISSION_COLUMNS

COORDINATE_COLUMNS = ["xmin", "ymin", "xmax", "ymax"]

DirectionPolicy = Literal["abstain", "majority", "first"]


def blend(
    sources: Sequence[tuple[Path, float]],
    direction: DirectionPolicy = "abstain",
    frame: FrameSize | None = None,
) -> pd.DataFrame:
    """Weighted-average several submissions into one.

    Args:
        sources: ``(path, weight)`` pairs. Weights are normalised, so
            ``[(a, 3), (b, 2)]`` and ``[(a, 0.6), (b, 0.4)]`` are equivalent.
        direction: how to combine the direction column. ``"abstain"`` emits -1
            everywhere, which is what the final submissions did and what the
            validation numbers justify; ``"majority"`` takes a weighted vote
            among the sources that committed to an answer; ``"first"`` copies
            the highest-weighted source.
        frame: used only to describe suspicious boxes in warnings.

    Returns:
        A submission frame with the id order of the first source.
    """
    if not sources:
        raise ValueError("at least one source is required")
    frame_size = frame or FrameSize()

    tables = []
    for path, weight in sources:
        table = pd.read_csv(path)
        missing = set(SUBMISSION_COLUMNS) - set(table.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        tables.append((table.set_index("id"), float(weight)))

    reference_ids = pd.read_csv(sources[0][0])["id"].astype(str).tolist()
    for (table, _), (path, _) in zip(tables, sources, strict=True):
        if sorted(table.index.astype(str)) != sorted(reference_ids):
            raise ValueError(f"{path} does not cover the same ids as {sources[0][0]}")

    total_weight = sum(weight for _, weight in tables)
    if total_weight <= 0:
        raise ValueError("weights must sum to something positive")

    out = pd.DataFrame({"id": reference_ids}).set_index("id")
    for column in COORDINATE_COLUMNS:
        out[column] = (
            sum(table.loc[reference_ids, column].to_numpy() * weight for table, weight in tables)
            / total_weight
        )

    out["direction"] = _blend_direction(
        [(table.loc[reference_ids, "direction"].to_numpy(), weight) for table, weight in tables],
        direction,
    )

    result = out.reset_index()[SUBMISSION_COLUMNS]
    _warn_on_degenerate_boxes(result, frame_size)
    return result


def _blend_direction(
    columns: Sequence[tuple[np.ndarray, float]],
    policy: DirectionPolicy,
) -> np.ndarray:
    n = len(columns[0][0])
    if policy == "abstain":
        return np.full(n, DIRECTION_ABSTAIN, dtype=int)
    if policy == "first":
        return columns[0][0].astype(int)

    votes = np.zeros((n, 2), dtype=float)
    for values, weight in columns:
        for label in (0, 1):
            votes[:, label] += (values == label) * weight
    winner = votes.argmax(axis=1)
    decided = votes.max(axis=1) > votes.min(axis=1)
    return np.where(decided, winner, DIRECTION_ABSTAIN).astype(int)


def _warn_on_degenerate_boxes(table: pd.DataFrame, frame: FrameSize) -> None:
    """Flag blended boxes that are not fully outside the frame.

    Every annotated person in this dataset is *entirely* off-screen, so a valid
    box lies wholly left of x=0 or wholly right of the frame width. Averaging a
    left-edge box with a right-edge one lands the result somewhere in the middle
    of the frame, which is not merely wrong but impossible -- and it overlaps
    neither of the two answers it came from, so it scores zero either way.
    That is the signature of a side disagreement between sources.
    """
    fully_left = table["xmax"] <= 0
    fully_right = table["xmin"] >= frame.width
    on_frame = ~(fully_left | fully_right)
    inverted = table["xmin"] >= table["xmax"]

    if on_frame.any():
        print(
            f"warning: {int(on_frame.sum())} blended boxes are not fully outside the frame, "
            "which means the sources disagreed about which side the person is on"
        )
    if inverted.any():
        print(f"warning: {int(inverted.sum())} blended boxes have xmin >= xmax")


def parse_source(spec: str) -> tuple[Path, float]:
    """Parse a ``path`` or ``path=weight`` command-line source specification."""
    if "=" in spec:
        path, _, weight = spec.rpartition("=")
        return Path(path), float(weight)
    return Path(spec), 1.0
