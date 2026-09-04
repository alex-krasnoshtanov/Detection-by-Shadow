"""Annotation parsing, target standardisation, and the dataset's augmentation.

:class:`~shadow_detection.data.TargetStats` gets the most attention here. It is
four numbers per target, it is needed to turn a model output back into pixels,
and the original notebooks recomputed it from the training set on every run and
never wrote it down -- which meant a saved checkpoint was unusable on its own.
Round-tripping it through disk is the thing that makes a run directory
self-contained.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from shadow_detection.data import (
    Sample,
    ShadowDataset,
    TargetStats,
    build_samples,
    load_annotation,
    stratification_labels,
)
from shadow_detection.features import NUM_FEATURES, extract_features
from shadow_detection.geometry import LEFT, REGRESSION_TARGETS, RIGHT, FrameSize
from tests.conftest import make_road_image

FRAME = FrameSize(720, 480)


class TestLoadAnnotation:
    def test_parses_the_challenge_schema(self, annotation_file):
        record = load_annotation(annotation_file(), FRAME)
        assert record["name"] == "image_0"
        assert record["direction"] == 1
        assert record["side"] == LEFT
        assert record["xmin"] == pytest.approx(-228.877)
        assert record["distance_from_edge"] == pytest.approx(228.877)
        assert record["bbox_width"] == pytest.approx(118.262, abs=1e-3)
        assert record["y_center"] == pytest.approx(319.989, abs=1e-3)

    def test_collapses_the_four_redundant_corners(self, annotation_file):
        """The annotation stores all four corners of an axis-aligned box, so
        two of the coordinates are redundant. Taking extremes rather than
        trusting ``top_left`` means a corner-order surprise cannot silently
        produce a negative-width box."""
        path = annotation_file(xmin=-300.0, ymin=200.0, xmax=-200.0, ymax=400.0)
        record = load_annotation(path, FRAME)
        assert record["xmin"] < record["xmax"]
        assert record["ymin"] < record["ymax"]

    def test_right_edge_sample(self, annotation_file):
        path = annotation_file(name="image_9", xmin=800.0, xmax=900.0)
        record = load_annotation(path, FRAME)
        assert record["side"] == RIGHT
        assert record["distance_from_edge"] == pytest.approx(180.0)


class TestTargetStats:
    @pytest.fixture
    def samples(self):
        rng = np.random.default_rng(0)
        return [
            Sample(
                image_path=None,
                name=f"image_{i}",
                side=i % 2,
                distance_from_edge=float(200 + rng.normal(0, 45)),
                bbox_width=float(80 + rng.normal(0, 30)),
                bbox_height=float(173 + rng.normal(0, 35)),
                y_center=float(309 + rng.normal(0, 14)),
                direction=i % 2,
                features=np.zeros(NUM_FEATURES, dtype=np.float32),
            )
            for i in range(200)
        ]

    def test_normalize_denormalize_round_trip(self, samples):
        stats = TargetStats.from_samples(samples)
        for key in REGRESSION_TARGETS:
            value = samples[0].target(key)
            assert stats.denormalize(stats.normalize(value, key), key) == pytest.approx(value)

    def test_standardised_targets_are_comparable(self, samples):
        """The point of standardising: the four targets span very different
        pixel ranges, and a single SmoothL1 over raw pixels would be dominated
        by whichever happened to be largest."""
        stats = TargetStats.from_samples(samples)
        normalised = np.array(
            [[stats.normalize(s.target(k), k) for k in REGRESSION_TARGETS] for s in samples]
        )
        assert normalised.mean(axis=0) == pytest.approx(np.zeros(4), abs=1e-6)
        assert normalised.std(axis=0) == pytest.approx(np.ones(4), abs=1e-6)

    def test_survives_a_round_trip_through_disk(self, samples, tmp_path):
        stats = TargetStats.from_samples(samples)
        path = tmp_path / "target_stats.json"
        stats.save(path)
        reloaded = TargetStats.load(path)
        for key in REGRESSION_TARGETS:
            assert reloaded.stats[key] == pytest.approx(stats.stats[key])

    def test_rejects_incomplete_files(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text('{"bbox_width": {"mean": 1, "std": 2}}', encoding="utf-8")
        with pytest.raises(ValueError, match="missing target stats"):
            TargetStats.load(path)

    def test_denormalize_vector_names_the_outputs(self, samples):
        stats = TargetStats.from_samples(samples)
        pixels = stats.denormalize_vector([0.0, 0.0, 0.0, 0.0])
        assert set(pixels) == set(REGRESSION_TARGETS)
        for key in REGRESSION_TARGETS:
            assert pixels[key] == pytest.approx(stats.stats[key][0])


def test_stratification_crosses_side_and_direction():
    samples = [
        Sample(None, "a", 0, 1, 1, 1, 1, 0, np.zeros(NUM_FEATURES, np.float32)),
        Sample(None, "b", 0, 1, 1, 1, 1, 1, np.zeros(NUM_FEATURES, np.float32)),
        Sample(None, "c", 1, 1, 1, 1, 1, 0, np.zeros(NUM_FEATURES, np.float32)),
        Sample(None, "d", 1, 1, 1, 1, 1, 1, np.zeros(NUM_FEATURES, np.float32)),
    ]
    assert stratification_labels(samples) == [0, 1, 2, 3]


class TestBuildSamples:
    @pytest.fixture
    def dataset_dir(self, tmp_path, annotation_file):
        """Two images and two annotations in the challenge's layout."""
        directory = tmp_path
        for index, (name, xmin, xmax) in enumerate(
            [("image_0", -250.0, -150.0), ("image_1", 800.0, 900.0)]
        ):
            annotation_file(name=name, xmin=xmin, xmax=xmax, direction=index)
            Image.fromarray(make_road_image(shadow_centre_x=150 + 300 * index)).save(
                directory / f"{name}.png"
            )
        return directory

    def test_loads_images_and_features(self, dataset_dir):
        samples = build_samples(dataset_dir, FRAME, progress=False)
        assert [s.name for s in samples] == ["image_0", "image_1"]
        assert [s.side for s in samples] == [LEFT, RIGHT]
        assert all(s.features.shape == (NUM_FEATURES,) for s in samples)

    def test_feature_cache_round_trips(self, dataset_dir, tmp_path):
        cache = tmp_path / "cache" / "features.npz"
        first = build_samples(dataset_dir, FRAME, features_cache=cache, progress=False)
        assert cache.exists()

        second = build_samples(dataset_dir, FRAME, features_cache=cache, progress=False)
        for a, b in zip(first, second, strict=True):
            assert a.features == pytest.approx(b.features)

    def test_cached_features_match_a_fresh_computation(self, dataset_dir, tmp_path):
        """A stale or wrong cache would silently poison training, so the cached
        values have to be exactly what recomputing produces."""
        cache = tmp_path / "features.npz"
        build_samples(dataset_dir, FRAME, features_cache=cache, progress=False)
        cached = build_samples(dataset_dir, FRAME, features_cache=cache, progress=False)
        for sample in cached:
            fresh = extract_features(np.array(Image.open(sample.image_path).convert("RGB")))
            assert sample.features == pytest.approx(fresh)

    def test_missing_annotations_is_an_error_not_an_empty_list(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no .* annotations"):
            build_samples(tmp_path, FRAME, progress=False)


class TestShadowDataset:
    @pytest.fixture
    def samples(self, tmp_path):
        path = tmp_path / "image_0.png"
        Image.fromarray(make_road_image()).save(path)
        return [
            Sample(
                image_path=path,
                name="image_0",
                side=LEFT,
                distance_from_edge=228.0,
                bbox_width=118.0,
                bbox_height=192.0,
                y_center=320.0,
                direction=1,
                features=extract_features(make_road_image()),
            )
        ]

    def test_item_shapes(self, samples):
        stats = TargetStats({k: (0.0, 1.0) for k in REGRESSION_TARGETS})
        item = ShadowDataset(samples, stats, input_size=(64, 64))[0]
        assert item["image"].shape == (3, 64, 64)
        assert item["features"].shape == (NUM_FEATURES,)
        assert item["regression"].shape == (4,)
        assert item["side"].item() == LEFT
        assert item["direction"].item() == 1

    def test_native_resolution_when_input_size_is_none(self, samples):
        stats = TargetStats({k: (0.0, 1.0) for k in REGRESSION_TARGETS})
        item = ShadowDataset(samples, stats, input_size=None)[0]
        assert item["image"].shape == (3, FRAME.height, FRAME.width)

    def test_flip_inverts_side_direction_and_features_together(self, samples, monkeypatch):
        """The core augmentation invariant. Mirroring the image turns a
        left-edge person into a right-edge person walking the other way, and
        the hand-crafted features have to move with them. All three or none.
        """
        import shadow_detection.data as data_module
        from shadow_detection.features import mirror_features

        stats = TargetStats({k: (0.0, 1.0) for k in REGRESSION_TARGETS})
        dataset = ShadowDataset(samples, stats, input_size=(64, 64), augment=True)

        monkeypatch.setattr(data_module.random, "random", lambda: 0.0)  # always flip
        flipped = dataset[0]

        assert flipped["side"].item() == RIGHT
        assert flipped["direction"].item() == 0
        assert flipped["features"].numpy() == pytest.approx(mirror_features(samples[0].features))

    def test_no_flip_leaves_labels_alone(self, samples, monkeypatch):
        import shadow_detection.data as data_module

        stats = TargetStats({k: (0.0, 1.0) for k in REGRESSION_TARGETS})
        dataset = ShadowDataset(samples, stats, input_size=(64, 64), augment=True)

        monkeypatch.setattr(data_module.random, "random", lambda: 0.99)  # never flip
        item = dataset[0]

        assert item["side"].item() == LEFT
        assert item["direction"].item() == 1
        assert item["features"].numpy() == pytest.approx(samples[0].features)

    def test_augmentation_off_is_deterministic(self, samples):
        stats = TargetStats({k: (0.0, 1.0) for k in REGRESSION_TARGETS})
        dataset = ShadowDataset(samples, stats, input_size=(64, 64), augment=False)
        assert dataset[0]["image"].numpy() == pytest.approx(dataset[0]["image"].numpy())
