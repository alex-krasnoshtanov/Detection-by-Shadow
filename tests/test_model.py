"""The network's wiring: shapes, the geometry side-channel, and checkpointing.

Every model here is built with ``pretrained=False`` so the suite needs no
network access. The weights are therefore meaningless -- what is under test is
that the 19 hand-crafted features actually reach the heads, that the trunk's
global pool really does make the architecture resolution-independent, and that
a checkpoint round-trips.
"""

from __future__ import annotations

import pytest
import torch

from shadow_detection.features import NUM_FEATURES
from shadow_detection.model import ShadowNet


@pytest.fixture(scope="module")
def model() -> ShadowNet:
    return ShadowNet(pretrained=False).eval()


def _inputs(batch: int = 2, height: int = 64, width: int = 64):
    torch.manual_seed(0)
    return torch.randn(batch, 3, height, width), torch.randn(batch, NUM_FEATURES)


def test_output_shapes(model):
    images, features = _inputs()
    side, regression, direction = model(images, features)
    assert side.shape == (2, 2)
    assert regression.shape == (2, 4)
    assert direction.shape == (2, 2)


@pytest.mark.parametrize("size", [(64, 64), (96, 144), (120, 180)])
def test_any_input_resolution_works(model, size):
    """The trunk keeps ResNet's global average pool, so any resolution collapses
    to 2048 features. That is what let the same architecture run at both
    384x384 and the native 720x480 without a single change."""
    images, features = _inputs(height=size[0], width=size[1])
    side, regression, direction = model(images, features)
    assert side.shape == (2, 2)
    assert regression.shape == (2, 4)


def test_features_actually_influence_the_output(model):
    """If the concatenation were dropped or the tensor silently zeroed, every
    prediction would be identical regardless of the geometry -- and the whole
    point of the hand-crafted block would be lost with no error."""
    images, features = _inputs()
    with torch.no_grad():
        _, first, _ = model(images, features)
        _, second, _ = model(images, features + 1.0)
    assert not torch.allclose(first, second, atol=1e-6)


def test_missing_features_is_an_error_not_a_silent_zero(model):
    images, _ = _inputs()
    with pytest.raises(ValueError, match="19 hand-crafted features"):
        model(images)


def test_image_only_ablation_needs_no_features():
    """``num_features=0`` is the ablation that shows what the descriptors buy."""
    ablation = ShadowNet(num_features=0, pretrained=False).eval()
    images, _ = _inputs()
    side, regression, direction = ablation(images)
    assert side.shape == (2, 2)
    assert regression.shape == (2, 4)


class TestParameterGroups:
    def test_splits_trunk_from_heads(self, model):
        groups = model.parameter_groups(lr=3e-3, backbone_lr=3e-4)
        assert len(groups) == 2
        assert groups[0]["lr"] == 3e-4
        assert groups[1]["lr"] == 3e-3

    def test_every_parameter_lands_in_exactly_one_group(self, model):
        groups = model.parameter_groups(lr=1e-3, backbone_lr=1e-4)
        grouped = sum(len(group["params"]) for group in groups)
        assert grouped == len(list(model.parameters()))

    def test_the_trunk_group_is_the_larger_one(self, model):
        """A sanity check on the name-prefix split: ResNet-50 has far more
        parameters than three small heads, so if the grouping were inverted the
        pretrained weights would be the ones getting the aggressive rate."""
        trunk, heads = model.parameter_groups(lr=1e-3, backbone_lr=1e-4)
        assert sum(p.numel() for p in trunk["params"]) > sum(p.numel() for p in heads["params"])


class TestCheckpointing:
    def test_round_trips_through_disk(self, tmp_path, model):
        path = tmp_path / "model_seed42.pt"
        torch.save(model.state_dict(), path)

        reloaded = ShadowNet.from_checkpoint(path)
        images, features = _inputs()
        with torch.no_grad():
            original = model(images, features)
            restored = reloaded(images, features)
        for a, b in zip(original, restored, strict=True):
            assert torch.allclose(a, b, atol=1e-6)

    def test_loaded_model_is_in_eval_mode(self, tmp_path, model):
        """Dropout and batch norm behave differently in train mode, so a
        checkpoint loaded for inference must not be left training."""
        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), path)
        assert not ShadowNet.from_checkpoint(path).training


def test_parameter_count_matches_the_published_run():
    """The v4 notebook logged 24,864,456 parameters. Pinning that number means
    an accidental architecture change -- a different trunk, a resized
    bottleneck, an extra head -- cannot pass unnoticed while the docs continue
    to claim the published result."""
    model = ShadowNet(pretrained=False)
    assert sum(p.numel() for p in model.parameters()) == 24_864_456


class TestLoadForInference:
    """Prediction has to accept both artifact shapes this project produces.

    The training loop writes a state_dict; the team's released weights ship as
    a TorchScript archive so a deployment need not install this package. Both
    have to arrive at predict_with_model as something callable the same way.
    """

    def test_loads_a_state_dict(self, tmp_path, model):
        from shadow_detection.model import load_for_inference

        path = tmp_path / "model_seed42.pt"
        torch.save(model.state_dict(), path)
        loaded = load_for_inference(path)
        assert isinstance(loaded, ShadowNet)
        assert not loaded.training

    def test_loads_a_torchscript_archive(self, tmp_path, model):
        from shadow_detection.model import load_for_inference

        path = model.export_torchscript(tmp_path / "model.pt", input_size=(64, 64))
        loaded = load_for_inference(path)
        assert isinstance(loaded, torch.jit.ScriptModule)
        assert not loaded.training

    def test_both_forms_agree_numerically(self, tmp_path, model):
        """A trace that disagreed with its source would silently change every
        prediction made from a released artifact."""
        from shadow_detection.model import load_for_inference

        images, features = _inputs(height=64, width=64)
        state_path = tmp_path / "state.pt"
        torch.save(model.state_dict(), state_path)
        script_path = model.export_torchscript(tmp_path / "traced.pt", input_size=(64, 64))

        with torch.no_grad():
            a = load_for_inference(state_path)(images, features)
            b = load_for_inference(script_path)(images, features)
        for x, y in zip(a, b, strict=True):
            assert torch.allclose(x, y, atol=1e-5)

    def test_torchscript_works_through_the_prediction_path(self, tmp_path, model):
        """The point of the abstraction: predict_with_model should not know or
        care which form it was handed."""
        from PIL import Image

        from shadow_detection.model import load_for_inference
        from shadow_detection.predict import predict_with_model
        from tests.conftest import make_road_image

        Image.fromarray(make_road_image()).save(tmp_path / "image_0.png")
        script_path = model.export_torchscript(tmp_path / "traced.pt", input_size=(64, 64))

        predictions = predict_with_model(
            load_for_inference(script_path),
            ["image_0"],
            tmp_path,
            input_size=(64, 64),
            tta=True,
        )
        assert predictions[0].regression.shape == (4,)
        assert predictions[0].side_probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_a_corrupt_file_is_not_silently_accepted(self, tmp_path):
        from shadow_detection.model import load_for_inference

        path = tmp_path / "junk.pt"
        path.write_bytes(b"this is not a checkpoint")
        with pytest.raises(Exception):  # noqa: B017 - torch raises several types here
            load_for_inference(path)
