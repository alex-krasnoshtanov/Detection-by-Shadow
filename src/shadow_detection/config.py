"""Run configuration, plus the two presets that produced the published scores.

Everything that was a bare module-level constant in the original notebooks
lives here instead, so a run is described by one object that can be printed,
diffed and saved next to its checkpoints. The two presets at the bottom are the
exact settings behind the ``v4-full-res`` and ``v5-ensemble`` rows of
``docs/experiments.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from shadow_detection.geometry import FrameSize

#: Per-channel mean of the challenge training images. Measured on the 1692
#: training frames rather than reused from ImageNet, because these are all
#: overhead-lit road scenes and noticeably darker and flatter than ImageNet.
CHANNEL_MEAN: tuple[float, float, float] = (0.422, 0.413, 0.394)
#: Per-channel standard deviation of the challenge training images.
CHANNEL_STD: tuple[float, float, float] = (0.167, 0.174, 0.233)


@dataclass(frozen=True, slots=True)
class DataPaths:
    """Where the challenge data lives.

    The dataset is not redistributed with this repository; see
    ``docs/dataset.md`` for the expected layout.
    """

    train_dir: Path
    test_dir: Path | None = None
    sample_csv: Path | None = None

    def __post_init__(self) -> None:
        if not self.train_dir.exists():
            raise FileNotFoundError(f"train_dir does not exist: {self.train_dir}")


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Hyperparameters for one training run.

    Args:
        input_size: ``(height, width)`` the images are resized to, or ``None``
            to feed them at their native resolution. Note that ``(384, 384)``
            squashes the native 3:2 frame to 1:1; that is what the best run
            actually did, and ``docs/experiments.md`` explains why it won.
        val_split: fraction held out for validation. ``0.0`` trains on every
            sample, which is what the final ensemble did -- there is then no
            validation signal, so ``epochs`` is fixed rather than early-stopped.
        scheduler: ``"plateau"`` needs a validation split to react to;
            ``"cosine"`` is the right choice when training for a fixed budget.
        seeds: one model is trained per seed. A single-element tuple is an
            ordinary single run.
    """

    output_dir: Path
    input_size: tuple[int, int] | None = (384, 384)
    batch_size: int = 128
    epochs: int = 40
    lr: float = 3e-3
    backbone_lr: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.3
    num_workers: int = 4
    val_split: float = 0.0
    patience: int = 15
    scheduler: Literal["cosine", "plateau"] = "cosine"
    seeds: tuple[int, ...] = (42, 123, 777)
    frame: FrameSize = field(default_factory=FrameSize)

    # Loss weights. The regression term carries 5x because it is a SmoothL1 on
    # standardised targets, which is numerically much smaller than the two
    # cross-entropy terms and would otherwise be drowned out.
    weight_side: float = 1.0
    weight_regression: float = 5.0
    weight_direction: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.val_split < 1.0:
            raise ValueError(f"val_split must be in [0, 1), got {self.val_split}")
        if self.scheduler == "plateau" and self.val_split == 0.0:
            raise ValueError(
                "scheduler='plateau' needs a validation split to monitor; "
                "use scheduler='cosine' when training on all data"
            )
        if not self.seeds:
            raise ValueError("at least one seed is required")

    def to_dict(self) -> dict:
        """JSON-serialisable view, saved alongside checkpoints."""
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        data["frame"] = {"width": self.frame.width, "height": self.frame.height}
        return data


#: The v4 run: native 720x480, held-out validation, early stopping. Best single
#: model before the switch to all-data training.
FULL_RES_PRESET = TrainConfig(
    output_dir=Path("runs/v4-full-res"),
    input_size=None,
    batch_size=32,
    epochs=80,
    lr=1e-3,
    backbone_lr=1e-4,
    val_split=0.15,
    patience=15,
    scheduler="plateau",
    seeds=(42,),
)

#: The v5 run: 384x384, all 1692 samples, three seeds averaged. Best result.
ENSEMBLE_PRESET = TrainConfig(
    output_dir=Path("runs/v5-ensemble"),
    input_size=(384, 384),
    batch_size=128,
    epochs=40,
    lr=3e-3,
    backbone_lr=3e-4,
    val_split=0.0,
    scheduler="cosine",
    seeds=(42, 123, 777),
)

PRESETS: dict[str, TrainConfig] = {
    "full-res": FULL_RES_PRESET,
    "ensemble": ENSEMBLE_PRESET,
}
