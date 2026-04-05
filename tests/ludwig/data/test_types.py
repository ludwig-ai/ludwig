"""Tests for typed metadata classes."""

from ludwig.data.types import (
    CategoryMetadata,
    NumberMetadata,
    TextMetadata,
    TrainingSetMetadata,
)


class TestNumberMetadata:
    def test_from_dict(self):
        d = {"mean": 5.0, "std": 2.0, "min": 0.0, "max": 10.0}
        meta = NumberMetadata.from_dict(d)
        assert meta.mean == 5.0
        assert meta.std == 2.0

    def test_to_dict(self):
        meta = NumberMetadata(mean=5.0, std=2.0)
        d = meta.to_dict()
        assert d["mean"] == 5.0
        assert "min" not in d  # None values excluded

    def test_ignores_unknown_keys(self):
        d = {"mean": 5.0, "unknown_key": "value"}
        meta = NumberMetadata.from_dict(d)
        assert meta.mean == 5.0

    def test_ple_bin_edges(self):
        meta = NumberMetadata(ple_bin_edges=[0.0, 0.25, 0.5, 0.75, 1.0])
        assert len(meta.ple_bin_edges) == 5


class TestCategoryMetadata:
    def test_roundtrip(self):
        meta = CategoryMetadata(idx2str=["a", "b", "c"], vocab_size=3)
        d = meta.to_dict()
        meta2 = CategoryMetadata.from_dict(d)
        assert meta2.idx2str == ["a", "b", "c"]
        assert meta2.vocab_size == 3


class TestTextMetadata:
    def test_defaults(self):
        meta = TextMetadata()
        assert meta.vocab_size == 0
        assert meta.pad_idx == 0


class TestTrainingSetMetadata:
    def test_dict_like_access(self):
        meta = TrainingSetMetadata()
        meta["feature1"] = {"mean": 5.0}
        assert meta["feature1"] == {"mean": 5.0}
        assert "feature1" in meta

    def test_get_with_default(self):
        meta = TrainingSetMetadata()
        assert meta.get("missing", "default") == "default"

    def test_from_dict(self):
        d = {
            "age": {"mean": 30.0, "std": 10.0},
            "income": {"idx2str": [">50K", "<=50K"]},
            "data_train_parquet_fp": "/path/to/train.parquet",
        }
        meta = TrainingSetMetadata.from_dict(d)
        assert meta["age"]["mean"] == 30.0
        assert meta.data_train_parquet_fp == "/path/to/train.parquet"

    def test_to_dict(self):
        meta = TrainingSetMetadata()
        meta["age"] = {"mean": 30.0}
        meta.data_train_parquet_fp = "/path/to/train.parquet"
        d = meta.to_dict()
        assert d["age"] == {"mean": 30.0}
        assert d["data_train_parquet_fp"] == "/path/to/train.parquet"

    def test_skips_hdf5_paths(self):
        d = {"data_train_hdf5_fp": "/old/path.hdf5", "feature1": {"mean": 1.0}}
        meta = TrainingSetMetadata.from_dict(d)
        assert meta.get("data_train_hdf5_fp") is None
        assert meta["feature1"]["mean"] == 1.0

    def test_keys_and_items(self):
        meta = TrainingSetMetadata()
        meta["f1"] = {"a": 1}
        meta["f2"] = {"b": 2}
        assert set(meta.keys()) == {"f1", "f2"}
        assert len(list(meta.items())) == 2
