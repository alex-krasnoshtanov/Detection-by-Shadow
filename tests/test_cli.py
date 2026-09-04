"""The command line, including a full train-then-predict round trip.

The round trip is the most valuable test in the suite. It is the only one that
exercises the whole path -- annotations to features to training to checkpoint to
inference to a validated submission -- and it is exactly the path that was
stitched together by hand across three notebooks during the hackathon. It runs
in a few seconds on CPU: two epochs over eight synthetic 64x64 images, with an
untrained trunk. The predictions are garbage; the plumbing is what matters.
"""

from __future__ import annotations

import json

import pytest
import torch
from PIL import Image

from shadow_detection.cli import build_parser, main
from tests.conftest import make_road_image


@pytest.fixture
def tiny_dataset(tmp_path, annotation_file):
    """Eight annotated images, four per side, both directions represented.

    Stratified splitting needs at least two of every (side, direction) class,
    so the mix is deliberate rather than random.
    """
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    for index in range(8):
        side_left = index < 4
        xmin = -250.0 - index if side_left else 800.0 + index
        xmax = xmin + 100.0
        name = f"image_{index}"
        payload = {
            "file_name": name,
            "walking_into_frame_bool": index % 2,
            "bbox": {
                "top_left": [xmin, 220.0],
                "top_right": [xmax, 220.0],
                "bottom_left": [xmin, 400.0],
                "bottom_right": [xmax, 400.0],
            },
        }
        (train_dir / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
        Image.fromarray(
            make_road_image(shadow_centre_x=150 if side_left else 570, seed=index)
        ).save(train_dir / f"{name}.png")
    return train_dir


@pytest.fixture
def tiny_testset(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    ids = [f"image_{i}" for i in range(3)]
    for index, name in enumerate(ids):
        Image.fromarray(make_road_image(shadow_centre_x=200 + 100 * index)).save(
            test_dir / f"{name}.png"
        )
    sample_csv = tmp_path / "submission_example.csv"
    sample_csv.write_text(
        "id,xmin,ymin,xmax,ymax,direction\n"
        + "".join(f"{name},-200.0,220.0,-120.0,380.0,0\n" for name in ids),
        encoding="utf-8",
    )
    return test_dir, sample_csv, ids


class TestParser:
    def test_requires_a_subcommand(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_train_requires_a_train_dir(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["train"])

    def test_train_defaults_to_the_best_preset(self):
        args = build_parser().parse_args(["train", "--train-dir", "data/train"])
        assert args.preset == "ensemble"

    def test_predict_input_size_defaults_to_the_training_resolution(self):
        args = build_parser().parse_args(
            [
                "predict",
                "--test-dir",
                "t",
                "--sample-csv",
                "s.csv",
                "--checkpoints",
                "a.pt",
                "-o",
                "out.csv",
            ]
        )
        assert tuple(args.input_size) == (384, 384)
        assert args.direction_threshold == 0.6

    def test_blend_accepts_weighted_sources(self):
        args = build_parser().parse_args(["blend", "a.csv=3", "b.csv=1", "-o", "out.csv"])
        assert args.sources == ["a.csv=3", "b.csv=1"]
        assert args.direction == "abstain"


class TestTrainConfigOverrides:
    def test_dropping_the_split_switches_the_scheduler(self):
        """``--val-split 0`` and a plateau schedule are incompatible, so the
        override has to move both together or the config rejects itself."""
        from shadow_detection.cli import _resolve_train_config

        args = build_parser().parse_args(
            ["train", "--train-dir", "d", "--preset", "full-res", "--val-split", "0"]
        )
        cfg = _resolve_train_config(args)
        assert cfg.val_split == 0.0
        assert cfg.scheduler == "cosine"

    def test_adding_a_split_switches_to_plateau(self):
        from shadow_detection.cli import _resolve_train_config

        args = build_parser().parse_args(
            ["train", "--train-dir", "d", "--preset", "ensemble", "--val-split", "0.2"]
        )
        cfg = _resolve_train_config(args)
        assert cfg.scheduler == "plateau"

    def test_unspecified_options_keep_preset_values(self):
        from shadow_detection.cli import _resolve_train_config
        from shadow_detection.config import ENSEMBLE_PRESET

        args = build_parser().parse_args(["train", "--train-dir", "d", "--epochs", "3"])
        cfg = _resolve_train_config(args)
        assert cfg.epochs == 3
        assert cfg.batch_size == ENSEMBLE_PRESET.batch_size
        assert cfg.seeds == ENSEMBLE_PRESET.seeds


def test_features_subcommand_prints_every_descriptor(tmp_path, capsys):
    from shadow_detection.features import Feature

    path = tmp_path / "frame.png"
    Image.fromarray(make_road_image()).save(path)

    assert main(["features", str(path), "--mirrored"]) == 0
    out = capsys.readouterr().out
    assert "mirrored" in out
    for feature in Feature:
        assert feature.name.lower() in out


class TestRoundTrip:
    """Train, predict, blend -- the whole pipeline, end to end."""

    def test_train_writes_a_self_contained_run_directory(self, tiny_dataset, tmp_path):
        run_dir = tmp_path / "run"
        assert (
            main(
                [
                    "train",
                    "--train-dir",
                    str(tiny_dataset),
                    "--output-dir",
                    str(run_dir),
                    "--seeds",
                    "1",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--num-workers",
                    "0",
                    "--device",
                    "cpu",
                    "--features-cache",
                    str(tmp_path / "features.npz"),
                ]
            )
            == 0
        )
        # A run directory has to carry everything inference needs: the weights,
        # the standardisation constants, and a record of how it was produced.
        assert (run_dir / "model_seed1.pt").exists()
        assert (run_dir / "target_stats.json").exists()
        assert (run_dir / "run.json").exists()

        manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert manifest["num_samples"] == 8
        assert manifest["seeds"] == [1]
        assert manifest["config"]["epochs"] == 1

    def test_features_cache_is_reused_on_a_second_run(self, tiny_dataset, tmp_path):
        cache = tmp_path / "features.npz"
        common = [
            "train",
            "--train-dir",
            str(tiny_dataset),
            "--seeds",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--features-cache",
            str(cache),
        ]
        main([*common, "--output-dir", str(tmp_path / "a")])
        assert cache.exists()
        first_mtime = cache.stat().st_mtime_ns

        main([*common, "--output-dir", str(tmp_path / "b")])
        assert cache.stat().st_mtime_ns == first_mtime, "cache should not be rewritten"

    def test_predict_produces_a_valid_submission(self, tiny_dataset, tiny_testset, tmp_path):
        test_dir, sample_csv, ids = tiny_testset
        run_dir = tmp_path / "run"

        main(
            [
                "train",
                "--train-dir",
                str(tiny_dataset),
                "--output-dir",
                str(run_dir),
                "--seeds",
                "1",
                "2",
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--num-workers",
                "0",
                "--device",
                "cpu",
            ]
        )

        output = tmp_path / "submission.csv"
        assert (
            main(
                [
                    "predict",
                    "--test-dir",
                    str(test_dir),
                    "--sample-csv",
                    str(sample_csv),
                    "--checkpoints",
                    str(run_dir / "model_seed1.pt"),
                    str(run_dir / "model_seed2.pt"),
                    "--output",
                    str(output),
                    "--input-size",
                    "64",
                    "64",
                    "--device",
                    "cpu",
                ]
            )
            == 0
        )

        import pandas as pd

        submission = pd.read_csv(output)
        assert submission["id"].tolist() == ids
        assert list(submission.columns) == ["id", "xmin", "ymin", "xmax", "ymax", "direction"]
        assert submission[["xmin", "ymin", "xmax", "ymax"]].notna().all().all()
        assert submission["direction"].isin([-1, 0, 1]).all()

    def test_predict_explains_a_missing_target_stats_file(self, tmp_path, tiny_testset):
        test_dir, sample_csv, _ = tiny_testset
        with pytest.raises(SystemExit, match="target statistics not found"):
            main(
                [
                    "predict",
                    "--test-dir",
                    str(test_dir),
                    "--sample-csv",
                    str(sample_csv),
                    "--checkpoints",
                    str(tmp_path / "missing.pt"),
                    "--output",
                    str(tmp_path / "out.csv"),
                    "--device",
                    "cpu",
                ]
            )

    def test_blend_round_trips_two_submissions(self, tmp_path, tiny_testset, submission_frame):
        _, _, ids = tiny_testset
        first = tmp_path / "a.csv"
        second = tmp_path / "b.csv"
        submission_frame(ids, [-200.0, -300.0, -400.0]).to_csv(first, index=False)
        submission_frame(ids, [-100.0, -200.0, -300.0]).to_csv(second, index=False)

        output = tmp_path / "blended.csv"
        assert main(["blend", f"{first}=3", f"{second}=1", "-o", str(output)]) == 0

        import pandas as pd

        blended = pd.read_csv(output)
        assert blended["id"].tolist() == ids
        assert blended["xmin"].tolist() == pytest.approx([-175.0, -275.0, -375.0])
        assert (blended["direction"] == -1).all()


def test_cosine_schedule_with_a_validation_split_trains(tiny_dataset, tmp_path):
    """A held-out split combined with a cosine schedule is a legal but unusual
    pairing, and it is the case where the scheduler could be stepped with a
    loss instead of nothing. Exercising it here means the mistake cannot
    reappear silently.
    """
    from dataclasses import replace

    from shadow_detection.config import ENSEMBLE_PRESET
    from shadow_detection.data import build_samples
    from shadow_detection.train import train

    cfg = replace(
        ENSEMBLE_PRESET,
        output_dir=tmp_path / "cosine-with-val",
        input_size=(64, 64),
        batch_size=2,
        epochs=2,
        num_workers=0,
        seeds=(1,),
        val_split=0.5,
        scheduler="cosine",
    )
    samples = build_samples(tiny_dataset, cfg.frame, progress=False)
    results = train(cfg, samples, device=torch.device("cpu"))

    assert len(results[0].history["val_loss"]) == 2
    assert results[0].best_val_loss is not None
    assert results[0].checkpoint.exists()


def test_stratified_split_explains_a_too_small_holdout(tiny_dataset, tmp_path):
    """sklearn's own error talks about "the number of classes", which means
    nothing unless you already know the split is stratified over
    side x direction. The guard says what to change instead."""
    from dataclasses import replace

    from shadow_detection.config import ENSEMBLE_PRESET
    from shadow_detection.data import build_samples
    from shadow_detection.train import train

    cfg = replace(
        ENSEMBLE_PRESET,
        output_dir=tmp_path / "too-small",
        input_size=(64, 64),
        batch_size=2,
        epochs=1,
        num_workers=0,
        seeds=(1,),
        val_split=0.125,
        scheduler="plateau",
    )
    samples = build_samples(tiny_dataset, cfg.frame, progress=False)
    with pytest.raises(ValueError, match=r"stratified across 4 \(side, direction\) groups"):
        train(cfg, samples, device=torch.device("cpu"))
