"""Package a trained checkpoint into the release archive the demo downloads.

Takes a checkpoint, traces it to TorchScript, pairs it with the run's
``target_stats.json`` and writes ``model_artifacts.tar.gz`` -- the exact layout
:mod:`shadow_detection.demo.weights` expects. Then prints the digest to pin and
the ``gh`` command to publish.

    python scripts/package_release.py runs/v5-ensemble/model_seed42.pt

The pairing is the point. A TorchScript trace carries the graph and the weights
but not the standardisation constants, so on its own it cannot turn a regression
output back into pixels. Shipping them separately is how you get a demo that
loads happily and predicts nonsense.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import tempfile
from pathlib import Path

from shadow_detection.demo.weights import ARCHIVE_NAME, MODEL_FILE, STATS_FILE
from shadow_detection.model import ShadowNet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path, help="a model_seed*.pt from train")
    parser.add_argument(
        "--target-stats",
        type=Path,
        help="target_stats.json; defaults to the checkpoint's own run directory",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("release"), help="where to write the archive"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=(384, 384),
        help="tracing resolution; must match training (default: 384 384)",
    )
    parser.add_argument("--tag", default="weights-v1", help="release tag for the printed command")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        # A truncated copy-paste is the likely cause, so list what is actually
        # there rather than letting torch.load raise a bare FileNotFoundError.
        siblings = sorted(args.checkpoint.parent.glob("*.pt"))
        hint = "\n  found alongside it: " + ", ".join(s.name for s in siblings) if siblings else ""
        raise SystemExit(f"checkpoint not found: {args.checkpoint}{hint}")

    stats = args.target_stats or args.checkpoint.parent / STATS_FILE
    if not stats.exists():
        raise SystemExit(
            f"{stats} not found. Training writes it beside the checkpoints; the archive is "
            "useless without it, so this refuses rather than shipping half of one."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / ARCHIVE_NAME

    with tempfile.TemporaryDirectory() as scratch:
        staged = Path(scratch)
        model = ShadowNet.from_checkpoint(args.checkpoint, device="cpu")
        model.export_torchscript(staged / MODEL_FILE, input_size=tuple(args.input_size))
        shutil.copy(stats, staged / STATS_FILE)

        # Flat members, no directory prefix, so the extractor finds them by
        # basename whatever it is handed.
        with tarfile.open(archive, "w:gz") as tar:
            for name in (MODEL_FILE, STATS_FILE):
                tar.add(staged / name, arcname=name)

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    size_mb = archive.stat().st_size / 1e6

    print(f"\nwrote {archive} ({size_mb:.1f} MB)")
    print(f"sha256 {digest}\n")
    print("Publish it:")
    print(f"  gh release create {args.tag} {archive} \\")
    print(f'    --title "Trained weights ({args.tag})" \\')
    print('    --notes "TorchScript model plus target_stats.json. See docs/demo.md."')
    print("\nThen pin the digest so a corrupted download is caught:")
    print(f'  EXPECTED_SHA256 = "{digest}"   # in src/shadow_detection/demo/weights.py')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
