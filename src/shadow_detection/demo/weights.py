"""Fetch the trained model from a GitHub release, once, and cache it.

The demo has to be runnable by someone who has just cloned the repository and
has no dataset, no GPU and no intention of training anything. So the weights
are not in git -- they are ~95 MB of binary that would bloat every clone -- and
are instead pulled from a release on first start and cached under the user's
cache directory.

Everything is overridable by environment variable, because the two cases that
matter both need it: a container wants the cache on a mounted volume, and a
developer testing a new checkpoint wants to point at a local file without
publishing anything.

    SHADOW_MODEL_DIR     where to cache (default: platform cache dir)
    SHADOW_MODEL_URL     full URL of the .tar.gz to fetch
    SHADOW_RELEASE_TAG   release tag to build the default URL from
    SHADOW_MODEL_SHA256  expected digest of the archive; skips the built-in one
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

#: Repository the default release URL points at.
REPOSITORY = "alex-krasnoshtanov/Detection-by-Shadow"

#: Release tag fetched when nothing overrides it.
DEFAULT_TAG = "weights-v1"

#: Name of the release asset, and of the two files it must contain.
ARCHIVE_NAME = "model_artifacts.tar.gz"
MODEL_FILE = "model.pt"
STATS_FILE = "target_stats.json"

#: SHA-256 of the published archive. Set once the release exists; when None the
#: digest is printed on download so it can be recorded here, and integrity is
#: unverified until then.
EXPECTED_SHA256: str | None = None

#: Refuse anything absurd rather than filling the disk on a bad URL.
MAX_ARCHIVE_BYTES = 500 * 1024 * 1024


class WeightsUnavailableError(RuntimeError):
    """Raised when the model could not be obtained, with what to do about it."""


def cache_dir() -> Path:
    """Where the extracted model lives between runs."""
    override = os.environ.get("SHADOW_MODEL_DIR")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME") or os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "shadow-detection" / "models"


def archive_url() -> str:
    """The archive to download, honouring both override variables."""
    explicit = os.environ.get("SHADOW_MODEL_URL")
    if explicit:
        return explicit
    tag = os.environ.get("SHADOW_RELEASE_TAG", DEFAULT_TAG)
    return f"https://github.com/{REPOSITORY}/releases/download/{tag}/{ARCHIVE_NAME}"


def ensure_weights(destination: Path | None = None) -> tuple[Path, Path]:
    """Return paths to ``(model.pt, target_stats.json)``, downloading if needed.

    Idempotent: if both files are already cached the network is never touched,
    which is what makes container restarts fast and offline runs possible.
    """
    destination = destination or cache_dir()
    model_path = destination / MODEL_FILE
    stats_path = destination / STATS_FILE

    if model_path.exists() and stats_path.exists():
        return model_path, stats_path

    url = archive_url()
    print(f"model not cached; fetching {url}")
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / ARCHIVE_NAME
        _download(url, archive)
        _verify(archive)
        _extract(archive, destination)

    if not (model_path.exists() and stats_path.exists()):
        raise WeightsUnavailableError(
            f"{ARCHIVE_NAME} did not contain both {MODEL_FILE} and {STATS_FILE}. "
            f"Extracted into {destination}; contents: "
            f"{sorted(p.name for p in destination.iterdir())}"
        )

    print(f"model ready in {destination}")
    return model_path, stats_path


def _download(url: str, target: Path) -> None:
    try:
        # The URL is a release asset over https, or whatever the operator
        # explicitly set SHADOW_MODEL_URL to.
        with urllib.request.urlopen(url, timeout=60) as response:
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > MAX_ARCHIVE_BYTES:
                raise WeightsUnavailableError(
                    f"{url} declares {int(declared) / 1e6:.0f} MB, over the "
                    f"{MAX_ARCHIVE_BYTES / 1e6:.0f} MB ceiling"
                )
            written = 0
            with target.open("wb") as handle:
                while chunk := response.read(1 << 20):
                    written += len(chunk)
                    if written > MAX_ARCHIVE_BYTES:
                        raise WeightsUnavailableError(f"{url} exceeded the size ceiling")
                    handle.write(chunk)
    except WeightsUnavailableError:
        raise
    except Exception as error:  # URLError, HTTPError, socket.timeout, ...
        raise WeightsUnavailableError(
            f"could not download the model from {url} ({error}).\n"
            "If no release exists yet, train one and point the demo at it:\n"
            "  shadow-detection train --train-dir data/train_data/train_data "
            "--preset ensemble --output-dir runs/v5\n"
            "  shadow-detection export runs/v5/model_seed42.pt -o local/model.pt\n"
            "  SHADOW_MODEL_DIR=local  (then restart)"
        ) from error
    print(f"downloaded {target.stat().st_size / 1e6:.1f} MB")


def _verify(archive: Path) -> None:
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    expected = os.environ.get("SHADOW_MODEL_SHA256") or EXPECTED_SHA256
    if expected is None:
        print(f"sha256 {digest} (unverified: no expected digest configured)")
        return
    if digest != expected:
        raise WeightsUnavailableError(
            f"checksum mismatch for {archive.name}: expected {expected}, got {digest}"
        )
    print(f"sha256 {digest} verified")


def _extract(archive: Path, destination: Path) -> None:
    """Extract the two known members, flattened, ignoring any path prefix.

    Only ``model.pt`` and ``target_stats.json`` are taken, by basename, and
    written directly into the destination. A tar member cannot therefore
    escape the destination directory however its name is constructed, which is
    the traversal problem ``tarfile`` is careless about by default.
    """
    wanted = {MODEL_FILE, STATS_FILE}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if not member.isfile() or name not in wanted:
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            with source, (destination / name).open("wb") as handle:
                shutil.copyfileobj(source, handle)
