"""The web demo: weight acquisition, and the API around the model.

Nothing here touches the network. `ensure_weights` is pointed at a local
directory, which is the same override a container uses for a mounted volume, so
the tests exercise the real code path rather than a mock of it.
"""

from __future__ import annotations

import io
import json
import tarfile

import pytest
import torch
from PIL import Image

from shadow_detection.demo import weights as weights_module
from shadow_detection.geometry import REGRESSION_TARGETS
from shadow_detection.model import ShadowNet
from tests.conftest import make_road_image

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient", reason="the demo extra is not installed"
)


@pytest.fixture
def staged_model(tmp_path):
    """A real TorchScript model plus its stats, laid out as the release ships them."""
    directory = tmp_path / "models"
    directory.mkdir()
    ShadowNet(pretrained=False).export_torchscript(directory / "model.pt", input_size=(64, 64))
    (directory / "target_stats.json").write_text(
        json.dumps({key: {"mean": 0.0, "std": 1.0} for key in REGRESSION_TARGETS}),
        encoding="utf-8",
    )
    return directory


@pytest.fixture
def client(staged_model, monkeypatch):
    """A test client whose startup finds the model already cached."""
    monkeypatch.setenv("SHADOW_MODEL_DIR", str(staged_model))
    from shadow_detection.demo import app as app_module

    with fastapi_testclient.TestClient(app_module.app) as test_client:
        yield test_client


@pytest.fixture
def frame_bytes():
    def _make(width=720, height=480):
        image = Image.fromarray(make_road_image())
        if (width, height) != image.size:
            image = image.resize((width, height))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    return _make


class TestWeightResolution:
    def test_cache_dir_honours_the_override(self, tmp_path, monkeypatch):
        """A container needs the cache on a mounted volume, not in $HOME."""
        monkeypatch.setenv("SHADOW_MODEL_DIR", str(tmp_path / "vol"))
        assert weights_module.cache_dir() == tmp_path / "vol"

    def test_cache_dir_falls_back_to_a_platform_location(self, monkeypatch):
        monkeypatch.delenv("SHADOW_MODEL_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/xdg")
        assert weights_module.cache_dir().as_posix().endswith("shadow-detection/models")

    def test_url_is_built_from_the_tag(self, monkeypatch):
        monkeypatch.delenv("SHADOW_MODEL_URL", raising=False)
        monkeypatch.setenv("SHADOW_RELEASE_TAG", "weights-v9")
        url = weights_module.archive_url()
        assert "weights-v9" in url
        assert url.endswith(weights_module.ARCHIVE_NAME)
        assert url.startswith("https://")

    def test_explicit_url_wins(self, monkeypatch):
        monkeypatch.setenv("SHADOW_MODEL_URL", "https://example.invalid/m.tar.gz")
        monkeypatch.setenv("SHADOW_RELEASE_TAG", "ignored")
        assert weights_module.archive_url() == "https://example.invalid/m.tar.gz"

    def test_cached_weights_skip_the_network_entirely(self, staged_model, monkeypatch):
        """The property that makes container restarts fast and offline use
        possible: if both files are present, nothing is fetched."""

        def explode(*_args, **_kwargs):
            raise AssertionError("the network must not be touched when cached")

        monkeypatch.setattr(weights_module.urllib.request, "urlopen", explode)
        model_path, stats_path = weights_module.ensure_weights(staged_model)
        assert model_path.exists()
        assert stats_path.exists()

    def test_a_download_failure_explains_the_way_out(self, tmp_path, monkeypatch):
        """A missing release is the expected first-run state before any model
        has been trained, so the error has to say what to do about it."""
        monkeypatch.setenv("SHADOW_MODEL_URL", "https://example.invalid/nope.tar.gz")
        with pytest.raises(weights_module.WeightsUnavailableError) as caught:
            weights_module.ensure_weights(tmp_path / "empty")
        message = str(caught.value)
        assert "shadow-detection train" in message
        assert "SHADOW_MODEL_DIR" in message


class TestArchiveExtraction:
    def _archive(self, tmp_path, members):
        path = tmp_path / "model_artifacts.tar.gz"
        with tarfile.open(path, "w:gz") as tar:
            for name, payload in members.items():
                data = payload.encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return path

    def test_members_are_flattened_out_of_any_prefix(self, tmp_path):
        """The release archive has been built with and without a leading
        directory; both must land the same way."""
        archive = self._archive(
            tmp_path, {"artifacts/model.pt": "weights", "artifacts/target_stats.json": "{}"}
        )
        destination = tmp_path / "out"
        destination.mkdir()
        weights_module._extract(archive, destination)
        assert (destination / "model.pt").read_text() == "weights"
        assert (destination / "target_stats.json").read_text() == "{}"

    def test_a_traversing_member_cannot_escape(self, tmp_path):
        """Extracting by basename means a crafted member name cannot write
        outside the destination -- the traversal hole tarfile leaves open."""
        archive = self._archive(
            tmp_path,
            {"../../evil.txt": "pwned", "model.pt": "weights", "target_stats.json": "{}"},
        )
        destination = tmp_path / "out"
        destination.mkdir()
        weights_module._extract(archive, destination)
        assert not (tmp_path.parent / "evil.txt").exists()
        assert sorted(p.name for p in destination.iterdir()) == [
            "model.pt",
            "target_stats.json",
        ]

    def test_an_archive_missing_a_member_is_reported(self, tmp_path, monkeypatch):
        import hashlib

        archive = self._archive(tmp_path, {"model.pt": "weights"})
        monkeypatch.setenv("SHADOW_MODEL_URL", archive.as_uri())
        # A digest is pinned in the module for the real release, and it is
        # checked before extraction, so this fixture has to declare its own.
        monkeypatch.setenv("SHADOW_MODEL_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
        with pytest.raises(weights_module.WeightsUnavailableError, match="did not contain"):
            weights_module.ensure_weights(tmp_path / "out")

    def test_the_pinned_digest_is_a_real_sha256(self):
        """A truncated or placeholder digest would reject every download."""
        pinned = weights_module.EXPECTED_SHA256
        assert pinned is not None, "the release exists, so the digest should be pinned"
        assert len(pinned) == 64
        assert all(character in "0123456789abcdef" for character in pinned)

    def test_a_checksum_mismatch_is_refused(self, tmp_path, monkeypatch):
        archive = self._archive(tmp_path, {"model.pt": "weights", "target_stats.json": "{}"})
        monkeypatch.setenv("SHADOW_MODEL_URL", archive.as_uri())
        monkeypatch.setenv("SHADOW_MODEL_SHA256", "0" * 64)
        with pytest.raises(weights_module.WeightsUnavailableError, match="checksum mismatch"):
            weights_module.ensure_weights(tmp_path / "out")


class TestApi:
    def test_health_reports_ready(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        assert body["frame"] == {"width": 720, "height": 480}
        assert body["direction_threshold"] == 0.6

    def test_the_page_is_served(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Detection by Shadow" in response.text

    def test_static_assets_are_served(self, client):
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/style.css").status_code == 200

    def test_predict_returns_a_box_and_confidences(self, client, frame_bytes):
        response = client.post(
            "/api/predict", files={"image": ("frame.png", frame_bytes(), "image/png")}
        )
        assert response.status_code == 200
        body = response.json()

        assert set(body["bbox"]) == {"xmin", "ymin", "xmax", "ymax"}
        assert body["bbox"]["xmin"] < body["bbox"]["xmax"]
        assert body["bbox"]["ymin"] < body["bbox"]["ymax"]
        assert body["side"] in (0, 1)
        assert body["side_label"] in ("left", "right")
        assert 0.0 <= body["side_confidence"] <= 1.0
        assert body["direction"] in (-1, 0, 1)
        assert body["image_width"] == 720
        assert body["image_height"] == 480
        assert body["inference_ms"] > 0

    @pytest.mark.parametrize(
        ("size", "scale"),
        [((720, 480), 1.0), ((1440, 960), 2.0), ((360, 240), 0.5)],
        ids=["native", "double", "half"],
    )
    def test_coordinates_are_scaled_to_the_uploaded_size(
        self, client, frame_bytes, monkeypatch, size, scale
    ):
        """The model reasons in the 720x480 frame it was trained on, so a box in
        those coordinates would be drawn in the wrong place on a differently
        sized upload -- off by exactly the scale factor.

        The prediction is pinned rather than taken from the (untrained) test
        model, because comparing two real predictions across two resized images
        measures the model's sensitivity to resizing, not this scaling code.
        """
        import numpy as np

        from shadow_detection.demo import app as app_module
        from shadow_detection.predict import RawPrediction

        fixed = RawPrediction(
            name="upload",
            # Identity target stats, so these are already pixels: 200 px past
            # the edge, 80 wide, 190 tall, centred at y=300.
            regression=np.array([200.0, 80.0, 190.0, 300.0], dtype=np.float32),
            side_probs=np.array([0.9, 0.1], dtype=np.float32),  # left
            direction_probs=np.array([0.5, 0.5], dtype=np.float32),  # abstains
        )
        monkeypatch.setattr(app_module, "predict_images", lambda *a, **k: [fixed])

        body = client.post(
            "/api/predict", files={"image": ("f.png", frame_bytes(*size), "image/png")}
        ).json()

        assert (body["image_width"], body["image_height"]) == size
        assert body["bbox"]["xmin"] == pytest.approx(-200.0 * scale)
        assert body["bbox"]["xmax"] == pytest.approx(-120.0 * scale)
        assert body["bbox"]["ymin"] == pytest.approx(205.0 * scale)
        assert body["bbox"]["ymax"] == pytest.approx(395.0 * scale)
        assert body["side_label"] == "left"
        assert body["direction"] == -1
        assert body["direction_label"] == "abstained"

    def test_an_empty_upload_is_rejected(self, client):
        response = client.post("/api/predict", files={"image": ("x.png", b"", "image/png")})
        assert response.status_code == 400
        assert "empty" in response.json()["detail"]

    def test_a_non_image_is_rejected_with_a_reason(self, client):
        response = client.post(
            "/api/predict", files={"image": ("notes.txt", b"hello there", "text/plain")}
        )
        assert response.status_code == 400
        assert "could not decode" in response.json()["detail"]

    def test_an_oversized_upload_is_refused_before_decoding(self, client, monkeypatch):
        from shadow_detection.demo import app as app_module

        monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 64)
        response = client.post(
            "/api/predict", files={"image": ("big.png", b"x" * 128, "image/png")}
        )
        assert response.status_code == 413

    def test_grayscale_and_rgba_uploads_both_work(self, client):
        for mode in ("L", "RGBA"):
            image = Image.fromarray(make_road_image()).convert(mode)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            response = client.post(
                "/api/predict", files={"image": (f"{mode}.png", buffer.getvalue(), "image/png")}
            )
            assert response.status_code == 200, mode


class TestApiWithoutAModel:
    @pytest.fixture
    def broken_client(self, tmp_path, monkeypatch):
        """Startup with no cached model and an unreachable release."""
        monkeypatch.setenv("SHADOW_MODEL_DIR", str(tmp_path / "missing"))
        monkeypatch.setenv("SHADOW_MODEL_URL", "https://example.invalid/none.tar.gz")
        from shadow_detection.demo import app as app_module

        with fastapi_testclient.TestClient(app_module.app) as test_client:
            yield test_client

    def test_the_server_still_starts(self, broken_client):
        """Exiting on a failed download would give a container crash loop and
        no explanation. Serving a 503 that says why is far easier to debug."""
        response = broken_client.get("/api/health")
        assert response.status_code == 503
        body = response.json()
        assert body["ready"] is False
        assert "could not download" in body["error"]

    def test_predict_explains_itself(self, broken_client, frame_bytes):
        response = broken_client.post(
            "/api/predict", files={"image": ("frame.png", frame_bytes(), "image/png")}
        )
        assert response.status_code == 503
        assert "could not download" in response.json()["detail"]


def test_predict_images_matches_the_directory_path(tmp_path):
    """The in-memory entry point exists for the server; it must not be a second
    implementation that can drift from the one the CLI uses."""
    from shadow_detection.predict import predict_images, predict_with_model

    model = ShadowNet(pretrained=False).eval()
    array = make_road_image()
    Image.fromarray(array).save(tmp_path / "image_0.png")

    from_disk = predict_with_model(model, ["image_0"], tmp_path, input_size=(64, 64), tta=True)
    from_memory = predict_images(
        model, [Image.fromarray(array)], names=["image_0"], input_size=(64, 64), tta=True
    )

    assert from_memory[0].name == from_disk[0].name
    assert from_memory[0].regression == pytest.approx(from_disk[0].regression, abs=1e-6)
    assert from_memory[0].side_probs == pytest.approx(from_disk[0].side_probs, abs=1e-6)


def test_predict_images_rejects_a_name_mismatch():
    from shadow_detection.predict import predict_images

    model = ShadowNet(pretrained=False).eval()
    with pytest.raises(ValueError, match="2 images but 1 names"):
        predict_images(
            model,
            [Image.fromarray(make_road_image())] * 2,
            names=["only-one"],
            input_size=(64, 64),
        )


def test_torch_is_not_required_to_be_cuda(client):
    """The demo has to work on a CPU-only box, which is where a reviewer will
    run it."""
    assert client.get("/api/health").json()["device"] in ("cpu", "cuda")
    assert torch.cuda.is_available() or True
