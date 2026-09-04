"""Command line entry point: ``shadow-detection {train,predict,blend,features}``.

The hackathon version of this project was five notebooks with hardcoded server
paths. Reproducing the winning submission meant opening three of them in the
right order and editing paths in each. These four subcommands are the same work
with the paths as arguments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from shadow_detection.config import PRESETS, DataPaths, TrainConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shadow-detection",
        description="Locate an off-frame pedestrian from the shadow they cast into the frame.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train --------------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="train one model per seed",
        description="Train one model per seed and write checkpoints plus a run manifest.",
    )
    train_parser.add_argument(
        "--train-dir", type=Path, required=True, help="directory of *.png / *.json pairs"
    )
    train_parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="ensemble",
        help="starting point for hyperparameters (default: ensemble, the best run)",
    )
    train_parser.add_argument("--output-dir", type=Path, help="override the preset output dir")
    train_parser.add_argument("--seeds", type=int, nargs="+", help="override the preset seeds")
    train_parser.add_argument("--epochs", type=int, help="override the preset epoch count")
    train_parser.add_argument("--batch-size", type=int, help="override the preset batch size")
    train_parser.add_argument(
        "--val-split",
        type=float,
        help="held-out fraction; 0 trains on everything (forces the cosine schedule)",
    )
    train_parser.add_argument("--num-workers", type=int, help="dataloader workers")
    train_parser.add_argument(
        "--features-cache",
        type=Path,
        help="npz file to cache the hand-crafted features in (saves ~100s per run)",
    )
    train_parser.add_argument("--device", help="e.g. cuda, cuda:4, cpu")

    # --- predict ------------------------------------------------------------
    predict_parser = subparsers.add_parser(
        "predict",
        help="run checkpoints over the test set and write a submission",
        description="Predict with one or more checkpoints, ensembling if several are given.",
    )
    predict_parser.add_argument("--test-dir", type=Path, required=True)
    predict_parser.add_argument("--sample-csv", type=Path, required=True, help="defines id order")
    predict_parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="one or more model_seed*.pt, or a released TorchScript model.pt",
    )
    predict_parser.add_argument(
        "--target-stats",
        type=Path,
        help="target_stats.json; defaults to the first checkpoint's directory",
    )
    predict_parser.add_argument("--output", "-o", type=Path, required=True)
    predict_parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=(384, 384),
        help="resize before inference; must match training (default: 384 384)",
    )
    predict_parser.add_argument(
        "--native-resolution",
        action="store_true",
        help="feed images at their original size, for checkpoints trained that way",
    )
    predict_parser.add_argument("--no-tta", action="store_true", help="skip horizontal-flip TTA")
    predict_parser.add_argument(
        "--direction-threshold",
        type=float,
        default=0.6,
        help="below this confidence, emit -1; use 0 to always commit (default: 0.6)",
    )
    predict_parser.add_argument("--batch-size", type=int, default=32)
    predict_parser.add_argument("--device", help="e.g. cuda, cpu")

    # --- blend --------------------------------------------------------------
    blend_parser = subparsers.add_parser(
        "blend",
        help="weighted-average several submission CSVs",
        description="Combine finished submissions by averaging their box coordinates.",
    )
    blend_parser.add_argument(
        "sources",
        nargs="+",
        metavar="CSV[=WEIGHT]",
        help="submission files, optionally weighted, e.g. ensemble.csv=3 single.csv=1",
    )
    blend_parser.add_argument("--output", "-o", type=Path, required=True)
    blend_parser.add_argument(
        "--direction",
        choices=("abstain", "majority", "first"),
        default="abstain",
        help="how to combine the direction column (default: abstain, i.e. -1)",
    )

    # --- export -------------------------------------------------------------
    export_parser = subparsers.add_parser(
        "export",
        help="convert a trained checkpoint to TorchScript for deployment",
        description=(
            "Trace a checkpoint to a self-contained TorchScript archive, which loads "
            "with torch.jit.load and needs neither this package nor its dependencies."
        ),
    )
    export_parser.add_argument("checkpoint", type=Path, help="a model_seed*.pt from train")
    export_parser.add_argument("--output", "-o", type=Path, required=True, help="the .pt to write")
    export_parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=(384, 384),
        help="tracing resolution; must match how the model was trained (default: 384 384)",
    )

    # --- features -----------------------------------------------------------
    features_parser = subparsers.add_parser(
        "features",
        help="print the 19 hand-crafted descriptors for one image",
        description="Inspect the geometric features, and check they mirror correctly.",
    )
    features_parser.add_argument("image", type=Path)
    features_parser.add_argument(
        "--mirrored", action="store_true", help="also print the horizontally mirrored vector"
    )

    return parser


def _resolve_train_config(args: argparse.Namespace) -> TrainConfig:
    """Apply command-line overrides on top of the chosen preset."""
    preset = PRESETS[args.preset]
    overrides: dict = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.seeds:
        overrides["seeds"] = tuple(args.seeds)
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if args.val_split is not None:
        overrides["val_split"] = args.val_split
        # A plateau schedule needs validation loss to react to, so dropping the
        # split has to drop the scheduler with it.
        overrides["scheduler"] = "plateau" if args.val_split > 0 else "cosine"

    from dataclasses import replace

    return replace(preset, **overrides)


def _command_train(args: argparse.Namespace) -> int:
    from shadow_detection.data import build_samples
    from shadow_detection.train import pick_device, train

    cfg = _resolve_train_config(args)
    paths = DataPaths(train_dir=args.train_dir)

    print(f"loading annotations from {paths.train_dir}")
    samples = build_samples(paths.train_dir, frame=cfg.frame, features_cache=args.features_cache)
    print(f"loaded {len(samples)} samples")

    train(cfg, samples, device=pick_device(args.device))
    return 0


def _command_predict(args: argparse.Namespace) -> int:
    import torch

    from shadow_detection.data import TargetStats
    from shadow_detection.model import load_for_inference
    from shadow_detection.predict import (
        average_predictions,
        predict_with_model,
        read_submission_ids,
        to_submission,
        write_submission,
    )
    from shadow_detection.train import pick_device

    stats_path = args.target_stats or args.checkpoints[0].parent / "target_stats.json"
    if not stats_path.exists():
        raise SystemExit(
            f"target statistics not found at {stats_path}. Training writes this file next to "
            "the checkpoints; pass --target-stats if it lives elsewhere."
        )
    target_stats = TargetStats.load(stats_path)

    device = pick_device(args.device)
    input_size = None if args.native_resolution else tuple(args.input_size)
    ids = read_submission_ids(args.sample_csv)
    print(f"{len(ids)} test ids, {len(args.checkpoints)} checkpoint(s), device={device}")

    runs = []
    for index, checkpoint in enumerate(args.checkpoints, start=1):
        print(f"predicting with {checkpoint.name} ({index}/{len(args.checkpoints)})")
        model = load_for_inference(checkpoint, device=device)
        runs.append(
            predict_with_model(
                model,
                ids,
                args.test_dir,
                input_size=input_size,
                device=device,
                tta=not args.no_tta,
                batch_size=args.batch_size,
            )
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    predictions = average_predictions(runs) if len(runs) > 1 else runs[0]
    submission = to_submission(
        predictions,
        target_stats,
        direction_threshold=args.direction_threshold,
    )
    write_submission(submission, args.output, ids)
    return 0


def _command_blend(args: argparse.Namespace) -> int:
    from shadow_detection.blend import blend, parse_source
    from shadow_detection.predict import read_submission_ids, write_submission

    sources = [parse_source(spec) for spec in args.sources]
    for path, weight in sources:
        print(f"  {path} (weight {weight:g})")
    blended = blend(sources, direction=args.direction)
    write_submission(blended, args.output, read_submission_ids(sources[0][0]))
    return 0


def _command_export(args: argparse.Namespace) -> int:
    import shutil

    from shadow_detection.model import ShadowNet

    model = ShadowNet.from_checkpoint(args.checkpoint, device="cpu")
    path = model.export_torchscript(args.output, input_size=tuple(args.input_size))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")

    # A trace alone is not deployable: without the standardisation constants
    # its regression outputs cannot be turned back into pixels. Ship the pair.
    stats = args.checkpoint.parent / "target_stats.json"
    if stats.exists():
        destination = shutil.copy(stats, args.output.parent / "target_stats.json")
        print(f"wrote {destination}")
    else:
        print(
            f"warning: no target_stats.json beside {args.checkpoint.name}. Inference needs "
            "it to denormalise the regression head; copy it next to the exported model."
        )
    return 0


def _command_features(args: argparse.Namespace) -> int:
    import numpy as np
    from PIL import Image

    from shadow_detection.features import Feature, extract_features, mirror_features

    image = np.array(Image.open(args.image).convert("RGB"))
    values = extract_features(image)
    mirrored = mirror_features(values) if args.mirrored else None

    header = f"{'feature':>28s}  {'value':>9s}"
    if mirrored is not None:
        header += f"  {'mirrored':>9s}"
    print(header)
    for feature in Feature:
        line = f"{feature.name.lower():>28s}  {values[feature]:9.4f}"
        if mirrored is not None:
            line += f"  {mirrored[feature]:9.4f}"
        print(line)
    return 0


_COMMANDS = {
    "train": _command_train,
    "predict": _command_predict,
    "blend": _command_blend,
    "export": _command_export,
    "features": _command_features,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
