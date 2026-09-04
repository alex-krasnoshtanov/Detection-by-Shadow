"""Inference: test-time augmentation, seed ensembling, submission assembly.

The target reparameterisation pays off twice here. Under a horizontal flip all
four regression targets -- distance from the edge, box width, box height,
vertical centre -- are *invariant*, because they are measured relative to
whichever edge the person is behind rather than in absolute image coordinates.
So flip-TTA needs no coordinate surgery at all: average the four regressed
values directly, and only the two classifier outputs need their class order
reversed. Compare that with the earlier direct-regression version, which had to
map ``xmin <- 1 - xmax_flipped`` and got it wrong twice before it was right.

Ensembling is a plain average over seeds in the same space, applied before the
box is reconstructed, so three models disagreeing about the side never produce
a box that belongs to neither of them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

from shadow_detection.config import CHANNEL_MEAN, CHANNEL_STD
from shadow_detection.data import TargetStats
from shadow_detection.features import extract_features, mirror_features
from shadow_detection.geometry import REGRESSION_TARGETS, FrameSize, reconstruct
from shadow_detection.model import ShadowNet

#: Columns of a valid submission, in order.
SUBMISSION_COLUMNS = ["id", "xmin", "ymin", "xmax", "ymax", "direction"]

#: Below this ensemble confidence the direction prediction is replaced by -1.
#: The challenge accepts -1 as "no answer", and validation direction accuracy
#: never rose meaningfully above chance, so guessing was strictly worse than
#: abstaining. See docs/experiments.md.
DEFAULT_DIRECTION_THRESHOLD = 0.6

#: The challenge's sentinel for "no direction prediction".
DIRECTION_ABSTAIN = -1


@dataclass(slots=True)
class RawPrediction:
    """One image's model outputs, before the box is reconstructed.

    Kept in this pre-reconstruction space so several models can be averaged.
    """

    name: str
    regression: np.ndarray
    side_probs: np.ndarray
    direction_probs: np.ndarray


def read_submission_ids(sample_csv: Path) -> list[str]:
    """Read the required prediction ids, in the order the challenge expects."""
    return pd.read_csv(sample_csv)["id"].astype(str).tolist()


def _build_transform(input_size: tuple[int, int] | None) -> transforms.Compose:
    steps: list = [transforms.Resize(input_size)] if input_size else []
    steps += [transforms.ToTensor(), transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD)]
    return transforms.Compose(steps)


@torch.no_grad()
def predict_with_model(
    model: ShadowNet,
    ids: Sequence[str],
    test_dir: Path,
    input_size: tuple[int, int] | None = (384, 384),
    device: torch.device | str = "cpu",
    tta: bool = True,
    batch_size: int = 32,
) -> list[RawPrediction]:
    """Run one model over the test ids, optionally with horizontal-flip TTA.

    Images are batched rather than fed one at a time. That is safe because the
    model is in eval mode, so batch norm uses its running statistics and the
    result is independent of how the images are grouped.
    """
    device = torch.device(device)
    model = model.to(device).eval()
    transform = _build_transform(input_size)
    use_amp = device.type == "cuda"

    predictions: list[RawPrediction] = []
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        images, features = [], []
        for name in chunk:
            image = _open_test_image(test_dir, name)
            features.append(extract_features(np.array(image)))
            images.append(transform(image))

        image_batch = torch.stack(images).to(device)
        feature_batch = torch.from_numpy(np.stack(features)).float().to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            side_logits, regression, direction_logits = model(image_batch, feature_batch)
        side_probs = F.softmax(side_logits.float(), dim=1).cpu().numpy()
        direction_probs = F.softmax(direction_logits.float(), dim=1).cpu().numpy()
        regression_np = regression.float().cpu().numpy()

        if tta:
            flipped_images = torch.flip(image_batch, dims=[3])
            flipped_features = (
                torch.from_numpy(np.stack([mirror_features(f) for f in features]))
                .float()
                .to(device)
            )
            with torch.amp.autocast("cuda", enabled=use_amp):
                f_side, f_regression, f_direction = model(flipped_images, flipped_features)
            # The four regressed quantities are flip-invariant, so they average
            # directly. The two classifiers predicted a mirrored world, so their
            # class order has to be reversed before averaging.
            regression_np = (regression_np + f_regression.float().cpu().numpy()) / 2.0
            side_probs = (
                side_probs + F.softmax(f_side.float(), dim=1).cpu().numpy()[:, ::-1]
            ) / 2.0
            direction_probs = (
                direction_probs + F.softmax(f_direction.float(), dim=1).cpu().numpy()[:, ::-1]
            ) / 2.0

        predictions.extend(
            RawPrediction(
                name=name,
                regression=regression_np[i],
                side_probs=side_probs[i],
                direction_probs=direction_probs[i],
            )
            for i, name in enumerate(chunk)
        )

    return predictions


def _open_test_image(test_dir: Path, name: str) -> Image.Image:
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = test_dir / f"{name}{suffix}"
        if candidate.exists():
            return Image.open(candidate).convert("RGB")
    raise FileNotFoundError(f"no image for id {name!r} under {test_dir}")


def average_predictions(runs: Sequence[Sequence[RawPrediction]]) -> list[RawPrediction]:
    """Average several models' raw outputs image by image.

    Args:
        runs: one sequence of predictions per model. All must cover the same
            ids in the same order.
    """
    if not runs:
        raise ValueError("nothing to average")
    if len({len(run) for run in runs}) != 1:
        raise ValueError("all runs must cover the same number of images")

    averaged = []
    for index in range(len(runs[0])):
        entries = [run[index] for run in runs]
        names = {entry.name for entry in entries}
        if len(names) != 1:
            raise ValueError(f"runs disagree on image order at position {index}: {sorted(names)}")
        averaged.append(
            RawPrediction(
                name=entries[0].name,
                regression=np.mean([e.regression for e in entries], axis=0),
                side_probs=np.mean([e.side_probs for e in entries], axis=0),
                direction_probs=np.mean([e.direction_probs for e in entries], axis=0),
            )
        )
    return averaged


def to_submission(
    predictions: Sequence[RawPrediction],
    target_stats: TargetStats,
    frame: FrameSize | None = None,
    direction_threshold: float = DEFAULT_DIRECTION_THRESHOLD,
) -> pd.DataFrame:
    """Reconstruct boxes and assemble a submission frame.

    ``direction_threshold`` of 0 or below disables abstention and always emits
    the argmax.
    """
    frame = frame or FrameSize()
    rows = []
    for prediction in predictions:
        pixels = target_stats.denormalize_vector(prediction.regression)
        xmin, ymin, xmax, ymax = reconstruct(
            side=int(prediction.side_probs.argmax()),
            frame=frame,
            **{key: pixels[key] for key in REGRESSION_TARGETS},
        )
        confidence = float(prediction.direction_probs.max())
        direction = (
            int(prediction.direction_probs.argmax())
            if confidence > direction_threshold
            else DIRECTION_ABSTAIN
        )
        rows.append(
            {
                "id": prediction.name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "direction": direction,
            }
        )
    return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)


def write_submission(frame: pd.DataFrame, path: Path, expected_ids: Sequence[str]) -> None:
    """Validate against the required id list, then write the CSV.

    A submission with the right rows in the wrong order, or with a missing id
    silently filled by a merge, scores zero on those rows. Checking here is
    cheaper than finding out from the leaderboard.
    """
    if list(frame.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"columns must be {SUBMISSION_COLUMNS}, got {list(frame.columns)}")
    if frame["id"].astype(str).tolist() != list(expected_ids):
        raise ValueError("submission ids do not match the sample submission, in order")
    if frame[["xmin", "ymin", "xmax", "ymax"]].isna().any().any():
        raise ValueError("submission contains missing coordinates")

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    counts = frame["direction"].value_counts().to_dict()
    print(f"wrote {path} ({len(frame)} rows), direction counts: {counts}")
