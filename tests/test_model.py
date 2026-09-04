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
