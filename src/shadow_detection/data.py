"""Annotation loading, target standardisation and the training dataset.

Three things here are deliberate improvements over the notebooks this code came
from, all of them lessons from running out of time during the hackathon:

1. :class:`TargetStats` is saved to disk. The original computed the target
   mean/std from whatever data was loaded and never persisted it, which made a
   checkpoint useless without also re-reading the full training set.
2. The hand-crafted features can be cached. Computing them for 1692 images
   takes about 100 seconds, which was paid on every single run.
3. The flip augmentation calls :func:`~shadow_detection.features.mirror_features`
   rather than open-coding the index juggling, which the notebooks did in three
   separate places.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from shadow_detection.config import CHANNEL_MEAN, CHANNEL_STD
from shadow_detection.features import NUM_FEATURES, extract_features, mirror_features
from shadow_detection.geometry import REGRESSION_TARGETS, FrameSize, decompose


@dataclass(slots=True)
class Sample:
    """One training image with its decomposed target and cached features."""

    image_path: Path
    name: str
    side: int
    distance_from_edge: float
    bbox_width: float
    bbox_height: float
    y_center: float
    direction: int
    features: np.ndarray

    def target(self, key: str) -> float:
        return float(getattr(self, key))


def load_annotation(path: Path, frame: FrameSize | None = None) -> dict:
    """Parse one challenge annotation JSON into a flat dict.

    The annotation stores all four corners of an axis-aligned box, so the two
    redundant pairs are collapsed by taking the extremes. ``walking_into_frame_bool``
    becomes ``direction``: 1 if the person is walking into the frame, 0 if away.
    """
    frame = frame or FrameSize()
    with path.open(encoding="utf-8") as fh:
        annotation = json.load(fh)

    bbox = annotation["bbox"]
    xs = [bbox[corner][0] for corner in ("top_left", "top_right", "bottom_left", "bottom_right")]
    ys = [bbox[corner][1] for corner in ("top_left", "top_right", "bottom_left", "bottom_right")]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    decomposed = decompose(xmin, ymin, xmax, ymax, frame)
    return {
        "name": annotation["file_name"],
        "direction": int(annotation["walking_into_frame_bool"]),
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "side": decomposed.side,
        "distance_from_edge": decomposed.distance_from_edge,
        "bbox_width": decomposed.bbox_width,
        "bbox_height": decomposed.bbox_height,
        "y_center": decomposed.y_center,
    }


def build_samples(
    data_dir: Path,
    frame: FrameSize | None = None,
    features_cache: Path | None = None,
    progress: bool = True,
) -> list[Sample]:
    """Load every annotated image under ``data_dir`` and attach its features.

    Args:
        data_dir: searched recursively for ``*.json`` annotations, each paired
            with the ``*.png`` whose stem matches the annotation's ``file_name``.
        features_cache: ``.npz`` file to read the hand-crafted features from, or
            write them to if it does not exist. A cache whose image names do not
            match the loaded set is ignored and overwritten.
        progress: print a line every 300 images.

    Returns:
        Samples in sorted annotation order, so the list is reproducible.
    """
    frame = frame or FrameSize()
    images = {path.stem: path for path in data_dir.rglob("*.png")}
    annotation_paths = sorted(data_dir.rglob("*.json"))
    if not annotation_paths:
        raise FileNotFoundError(f"no *.json annotations under {data_dir}")

    cached = _load_feature_cache(features_cache)
    samples: list[Sample] = []

    for index, annotation_path in enumerate(annotation_paths):
        if progress and index % 300 == 0:
            print(f"  {index}/{len(annotation_paths)}")
        record = load_annotation(annotation_path, frame)
        image_path = images.get(record["name"])
        if image_path is None:
            continue

        features = cached.get(record["name"]) if cached else None
        if features is None:
            features = extract_features(np.array(Image.open(image_path).convert("RGB")))

        samples.append(
            Sample(
                image_path=image_path,
                name=record["name"],
                side=record["side"],
                distance_from_edge=record["distance_from_edge"],
                bbox_width=record["bbox_width"],
                bbox_height=record["bbox_height"],
                y_center=record["y_center"],
                direction=record["direction"],
                features=features,
            )
        )

    if not samples:
        raise RuntimeError(f"found {len(annotation_paths)} annotations but no matching images")

    if features_cache is not None and len(cached) != len(samples):
        _write_feature_cache(features_cache, samples)

    return samples


def _load_feature_cache(path: Path | None) -> dict[str, np.ndarray]:
    if path is None or not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as archive:
        names = archive["names"]
        values = archive["features"]
    if values.shape[1] != NUM_FEATURES:
        return {}
    return {str(name): values[i] for i, name in enumerate(names)}


def _write_feature_cache(path: Path, samples: Sequence[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        names=np.array([s.name for s in samples]),
        features=np.stack([s.features for s in samples]),
    )


@dataclass(frozen=True, slots=True)
class TargetStats:
    """Mean and standard deviation of each regression target, in pixels.

    The four targets live on wildly different scales -- ``y_center`` is around
    309 with a standard deviation of 14, ``distance_from_edge`` around 209 with
    a standard deviation of 45 -- so a single SmoothL1 over raw pixels would be
    dominated by whichever happened to be largest. Standardising first puts them
    on comparable footing.
    """

    stats: dict[str, tuple[float, float]]

    @classmethod
    def from_samples(cls, samples: Iterable[Sample]) -> TargetStats:
        samples = list(samples)
        stats = {}
        for key in REGRESSION_TARGETS:
            values = np.array([s.target(key) for s in samples], dtype=np.float64)
            stats[key] = (float(values.mean()), float(values.std()))
        return cls(stats)

    def normalize(self, value: float, key: str) -> float:
        mean, std = self.stats[key]
        return (value - mean) / (std + 1e-8)

    def denormalize(self, value: float, key: str) -> float:
        mean, std = self.stats[key]
        return value * (std + 1e-8) + mean

    def denormalize_vector(self, values: Sequence[float]) -> dict[str, float]:
        """Turn a raw regression head output into named pixel quantities."""
        return {
            key: self.denormalize(float(values[i]), key) for i, key in enumerate(REGRESSION_TARGETS)
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: {"mean": mean, "std": std} for key, (mean, std) in self.stats.items()}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TargetStats:
        payload = json.loads(path.read_text(encoding="utf-8"))
        missing = set(REGRESSION_TARGETS) - payload.keys()
        if missing:
            raise ValueError(f"{path} is missing target stats for {sorted(missing)}")
        return cls({key: (payload[key]["mean"], payload[key]["std"]) for key in REGRESSION_TARGETS})

    def __str__(self) -> str:
        return "\n".join(
            f"  {key:>20s}: mean={mean:8.2f}  std={std:7.2f}"
            for key, (mean, std) in self.stats.items()
        )


class ShadowDataset(Dataset):
    """Images plus decomposed targets, with flip-consistent augmentation.

    The horizontal flip is the interesting part. Mirroring the image turns a
    person behind the left edge into a person behind the right edge, so the
    ``side`` label inverts; it also reverses which way they are walking, so
    ``direction`` inverts too; and the hand-crafted features have to be mirrored
    to match. All three happen together or the augmentation actively teaches the
    model the wrong thing.

    Photometric augmentation (colour jitter and a light blur) is applied on
    every augmented sample, matching the best run. The earlier full-resolution
    run gated them at p=0.8 and p=0.2 respectively.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        target_stats: TargetStats,
        input_size: tuple[int, int] | None = (384, 384),
        augment: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.target_stats = target_stats
        self.augment = augment

        resize: list = [transforms.Resize(input_size)] if input_size else []
        photometric = [
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
        ]
        tail = [transforms.ToTensor(), transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD)]

        self._plain = transforms.Compose([*resize, *tail])
        self._augmented = transforms.Compose([*resize, *photometric, *tail])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        side = sample.side
        direction = sample.direction
        features = sample.features

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            side = 1 - side
            direction = 1 - direction
            features = mirror_features(features)

        tensor = self._augmented(image) if self.augment else self._plain(image)
        regression = torch.tensor(
            [self.target_stats.normalize(sample.target(key), key) for key in REGRESSION_TARGETS],
            dtype=torch.float32,
        )

        return {
            "image": tensor,
            "side": torch.tensor(side, dtype=torch.long),
            "regression": regression,
            "direction": torch.tensor(direction, dtype=torch.long),
            "features": torch.from_numpy(np.ascontiguousarray(features)).float(),
        }


def stratification_labels(samples: Sequence[Sample]) -> list[int]:
    """Four-way labels for a stratified split.

    Splitting on ``side`` alone leaves the direction balance to chance, and with
    only ~250 validation images that swing was large enough to make runs
    incomparable. Crossing the two gives a stable split.
    """
    return [s.side * 2 + s.direction for s in samples]
