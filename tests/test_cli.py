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
from pathlib import Path

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
        """A checkpoint with no stats beside it cannot denormalise anything, so
        say that rather than failing later with a KeyError."""
        from shadow_detection.model import ShadowNet

        test_dir, sample_csv, _ = tiny_testset
        orphan = tmp_path / "orphan" / "model_seed1.pt"
        orphan.parent.mkdir()
        torch.save(ShadowNet(pretrained=False).state_dict(), orphan)

        with pytest.raises(SystemExit, match="target statistics not found"):
            main(
                [
                    "predict",
                    "--test-dir",
                    str(test_dir),
                    "--sample-csv",
                    str(sample_csv),
                    "--checkpoints",
                    str(orphan),
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


class TestExport:
    def test_export_writes_a_loadable_trace_and_the_stats_beside_it(self, tiny_dataset, tmp_path):
        """A trace on its own is not deployable: without target_stats.json the
        regression head's output cannot be turned back into pixels. The pair is
        the artifact, so export ships both."""
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

        deploy = tmp_path / "deploy" / "model.pt"
        assert (
            main(
                [
                    "export",
                    str(run_dir / "model_seed1.pt"),
                    "-o",
                    str(deploy),
                    "--input-size",
                    "64",
                    "64",
                ]
            )
            == 0
        )
        assert deploy.exists()
        assert (deploy.parent / "target_stats.json").exists()

        traced = torch.jit.load(str(deploy), map_location="cpu")
        images = torch.randn(2, 3, 64, 64)
        features = torch.randn(2, 19)
        with torch.no_grad():
            side, regression, direction = traced(images, features)
        assert side.shape == (2, 2)
        assert regression.shape == (2, 4)
        assert direction.shape == (2, 2)

    def test_export_warns_when_the_stats_are_missing(self, tmp_path, capsys):
        from shadow_detection.model import ShadowNet

        checkpoint = tmp_path / "orphan.pt"
        torch.save(ShadowNet(pretrained=False).state_dict(), checkpoint)

        main(
            [
                "export",
                str(checkpoint),
                "-o",
                str(tmp_path / "out" / "model.pt"),
                "--input-size",
                "64",
                "64",
            ]
        )
        assert "no target_stats.json" in capsys.readouterr().out


def test_validation_reports_mean_iou(tiny_dataset, tmp_path):
    """The metric the challenge actually scored, which the hackathon runs never
    computed for the decomposed model -- leaving its improvement over direct
    regression visible only on a leaderboard measured differently.

    Both boxes are reconstructed through the same standardisation, so a
    perfect prediction must score exactly 1.0 rather than approximately.
    """
    from dataclasses import replace

    from shadow_detection.config import ENSEMBLE_PRESET
    from shadow_detection.data import build_samples
    from shadow_detection.train import train

    cfg = replace(
        ENSEMBLE_PRESET,
        output_dir=tmp_path / "iou",
        input_size=(64, 64),
        batch_size=2,
        epochs=1,
        num_workers=0,
        seeds=(1,),
        val_split=0.5,
        scheduler="cosine",
    )
    samples = build_samples(tiny_dataset, cfg.frame, progress=False)
    results = train(cfg, samples, device=torch.device("cpu"))

    scores = results[0].history["val_mean_iou"]
    assert len(scores) == 1
    # An untrained model scores badly but must produce a valid IoU.
    assert 0.0 <= scores[0] <= 1.0


def test_perfect_predictions_score_iou_one():
    """Pins the reconstruction path inside the metric: if predictions and truth
    are identical, the score is exactly 1, whatever the standardisation."""
    import numpy as np

    from shadow_detection.data import TargetStats
    from shadow_detection.geometry import REGRESSION_TARGETS, FrameSize
    from shadow_detection.train import _batch_iou

    stats = TargetStats(
        {
            "distance_from_edge": (208.58, 45.16),
            "bbox_width": (80.85, 29.86),
            "bbox_height": (172.90, 34.67),
            "y_center": (309.33, 14.27),
        }
    )
    assert set(stats.stats) == set(REGRESSION_TARGETS)

    regression = np.array([[0.4, -0.2, 0.7, 0.1], [-1.1, 0.3, -0.5, 0.0]], dtype=np.float32)
    sides = np.array([0, 1])

    total = _batch_iou(
        predicted_sides=sides,
        predicted_regression=regression,
        true_sides=sides,
        true_regression=regression,
        target_stats=stats,
        frame=FrameSize(720, 480),
    )
    assert total == pytest.approx(2.0)


def test_a_wrong_side_scores_zero_iou():
    """A box behind the wrong edge cannot overlap the truth at all, which is
    why side accuracy saturating at 100% matters so much to the final score."""
    import numpy as np

    from shadow_detection.data import TargetStats
    from shadow_detection.geometry import FrameSize
    from shadow_detection.train import _batch_iou

    stats = TargetStats(
        {k: (0.0, 1.0) for k in ("distance_from_edge", "bbox_width", "bbox_height", "y_center")}
    )
    regression = np.array([[200.0, 80.0, 190.0, 300.0]], dtype=np.float32)

    total = _batch_iou(
        predicted_sides=np.array([1]),
        predicted_regression=regression,
        true_sides=np.array([0]),
        true_regression=regression,
        target_stats=stats,
        frame=FrameSize(720, 480),
    )
    assert total == pytest.approx(0.0)


class TestCheckpointExpansion:
    """POSIX shells expand `model_seed*.pt`; PowerShell hands the pattern
    through verbatim. The documented command has to work in both."""

    @pytest.fixture
    def run_dir(self, tmp_path, monkeypatch):
        from shadow_detection.model import ShadowNet

        directory = tmp_path / "runs" / "v5"
        directory.mkdir(parents=True)
        for seed in (42, 123, 777):
            torch.save(ShadowNet(pretrained=False).state_dict(), directory / f"model_seed{seed}.pt")
        monkeypatch.chdir(tmp_path)
        return Path("runs/v5")

    def test_a_glob_expands_and_sorts(self, run_dir):
        from shadow_detection.cli import _expand_checkpoints

        found = _expand_checkpoints([run_dir / "model_seed*.pt"])
        assert [p.name for p in found] == [
            "model_seed123.pt",
            "model_seed42.pt",
            "model_seed777.pt",
        ]

    def test_explicit_paths_pass_through_in_order(self, run_dir):
        from shadow_detection.cli import _expand_checkpoints

        given = [run_dir / "model_seed777.pt", run_dir / "model_seed42.pt"]
        assert [p.name for p in _expand_checkpoints(given)] == [
            "model_seed777.pt",
            "model_seed42.pt",
        ]

    def test_overlapping_patterns_do_not_double_weight_a_model(self, run_dir):
        """An ensemble average over a duplicated checkpoint silently weights
        that seed twice."""
        from shadow_detection.cli import _expand_checkpoints

        found = _expand_checkpoints([run_dir / "model_seed*.pt", run_dir / "model_seed42.pt"])
        assert len(found) == 3

    def test_a_pattern_matching_nothing_is_an_error(self, run_dir):
        from shadow_detection.cli import _expand_checkpoints

        with pytest.raises(SystemExit, match="no checkpoints matched"):
            _expand_checkpoints([run_dir / "model_seedNONE*.pt"])

    def test_a_missing_explicit_path_is_an_error(self, run_dir):
        from shadow_detection.cli import _expand_checkpoints

        with pytest.raises(SystemExit, match="checkpoint not found"):
            _expand_checkpoints([run_dir / "absent.pt"])


def test_predict_reports_a_missing_checkpoint_before_anything_else(tmp_path, tiny_testset):
    """The path that used to reach the stats check with a nonexistent file."""
    test_dir, sample_csv, _ = tiny_testset
    with pytest.raises(SystemExit, match="checkpoint not found"):
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


class TestCheckpointSelection:
    """Which epoch's weights end up on disk.

    On a real 40-epoch run the composite validation loss bottomed at epoch 38
    with IoU 0.6037, while IoU peaked at epoch 40 with 0.6126. Selecting on
    loss therefore ships the worse model, and the direction term -- which sits
    at chance throughout -- is a large part of why the loss disagrees.
    """

    def _run(self, tiny_dataset, tmp_path, select_by):
        from dataclasses import replace

        from shadow_detection.config import ENSEMBLE_PRESET
        from shadow_detection.data import build_samples
        from shadow_detection.train import train

        cfg = replace(
            ENSEMBLE_PRESET,
            output_dir=tmp_path / select_by,
            input_size=(64, 64),
            batch_size=2,
            epochs=2,
            num_workers=0,
            seeds=(1,),
            val_split=0.5,
            scheduler="cosine",
            select_by=select_by,
        )
        samples = build_samples(tiny_dataset, cfg.frame, progress=False)
        return cfg, train(cfg, samples, device=torch.device("cpu"))[0]

    @pytest.mark.parametrize("select_by", ["iou", "loss"])
    def test_both_modes_produce_a_checkpoint_and_a_manifest(
        self, tiny_dataset, tmp_path, select_by
    ):
        cfg, result = self._run(tiny_dataset, tmp_path, select_by)
        assert result.checkpoint.exists()

        manifest = json.loads((cfg.output_dir / "run.json").read_text(encoding="utf-8"))
        assert manifest["config"]["select_by"] == select_by
        assert manifest["results"][0]["selected_by"] == select_by

    def test_iou_selection_picks_the_best_iou_epoch(self, tiny_dataset, tmp_path):
        _, result = self._run(tiny_dataset, tmp_path, "iou")
        scores = result.history["val_mean_iou"]
        assert result.best_epoch == scores.index(max(scores))
        assert result.best_mean_iou == pytest.approx(max(scores))

    def test_loss_selection_picks_the_best_loss_epoch(self, tiny_dataset, tmp_path):
        _, result = self._run(tiny_dataset, tmp_path, "loss")
        losses = result.history["val_loss"]
        assert result.best_epoch == losses.index(min(losses))

    def test_best_loss_is_recorded_whichever_mode_is_used(self, tiny_dataset, tmp_path):
        """The plateau scheduler and the manifest both still want the loss."""
        _, result = self._run(tiny_dataset, tmp_path, "iou")
        assert result.best_val_loss == pytest.approx(min(result.history["val_loss"]))
