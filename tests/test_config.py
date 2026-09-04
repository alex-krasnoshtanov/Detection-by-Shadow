"""Configuration validation and the presets that reproduce the published runs.

The presets are the documentation's contract: ``docs/experiments.md`` quotes
specific numbers for specific settings, so a preset drifting silently would
make the write-up wrong. These tests pin the settings the results were produced
with.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from shadow_detection.config import (
    ENSEMBLE_PRESET,
    FULL_RES_PRESET,
    PRESETS,
    DataPaths,
    TrainConfig,
)


class TestValidation:
    def test_plateau_scheduler_requires_a_validation_split(self):
        """``ReduceLROnPlateau`` monitors validation loss. With no split there
        is nothing to monitor, and the schedule would silently never step."""
        with pytest.raises(ValueError, match="needs a validation split"):
            TrainConfig(output_dir=Path("runs/x"), val_split=0.0, scheduler="plateau")

    def test_rejects_an_out_of_range_split(self):
        with pytest.raises(ValueError, match="val_split must be in"):
            TrainConfig(output_dir=Path("runs/x"), val_split=1.0)
        with pytest.raises(ValueError, match="val_split must be in"):
            TrainConfig(output_dir=Path("runs/x"), val_split=-0.1)

    def test_rejects_an_empty_seed_list(self):
        with pytest.raises(ValueError, match="at least one seed"):
            TrainConfig(output_dir=Path("runs/x"), seeds=())

    def test_cosine_with_no_split_is_fine(self):
        cfg = TrainConfig(output_dir=Path("runs/x"), val_split=0.0, scheduler="cosine")
        assert cfg.val_split == 0.0

    def test_data_paths_reject_a_missing_train_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="train_dir does not exist"):
            DataPaths(train_dir=tmp_path / "nope")

    def test_data_paths_accept_an_existing_dir(self, tmp_path):
        assert DataPaths(train_dir=tmp_path).train_dir == tmp_path


class TestPresets:
    def test_both_presets_are_registered(self):
        assert set(PRESETS) == {"full-res", "ensemble"}

    def test_full_res_matches_the_v4_run(self):
        """Native resolution, a held-out split, early stopping, one seed."""
        assert FULL_RES_PRESET.input_size is None
        assert FULL_RES_PRESET.batch_size == 32
        assert FULL_RES_PRESET.val_split == 0.15
        assert FULL_RES_PRESET.scheduler == "plateau"
        assert FULL_RES_PRESET.seeds == (42,)

    def test_ensemble_matches_the_v5_run(self):
        """384x384, all data, three seeds, cosine schedule, 40 fixed epochs."""
        assert ENSEMBLE_PRESET.input_size == (384, 384)
        assert ENSEMBLE_PRESET.batch_size == 128
        assert ENSEMBLE_PRESET.epochs == 40
        assert ENSEMBLE_PRESET.val_split == 0.0
        assert ENSEMBLE_PRESET.scheduler == "cosine"
        assert ENSEMBLE_PRESET.seeds == (42, 123, 777)

    def test_regression_loss_dominates_the_weighting(self):
        """SmoothL1 on standardised targets is numerically much smaller than
        the two cross-entropy terms, so without the 5x it gets drowned out."""
        for preset in PRESETS.values():
            assert preset.weight_regression > preset.weight_side
            assert preset.weight_regression > preset.weight_direction

    def test_presets_survive_a_json_round_trip(self):
        import json

        for preset in PRESETS.values():
            payload = json.dumps(preset.to_dict())
            assert json.loads(payload)["frame"] == {"width": 720, "height": 480}

    def test_replace_still_validates(self):
        """Overrides go through ``dataclasses.replace``, so ``__post_init__``
        has to run again -- otherwise the CLI could build an invalid config."""
        with pytest.raises(ValueError, match="needs a validation split"):
            replace(FULL_RES_PRESET, val_split=0.0)
