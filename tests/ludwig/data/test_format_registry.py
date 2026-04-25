"""Tests for format detection."""

from ludwig.data.format_registry import detect_format, detect_format_from_dataset


class TestDetectFormat:
    def test_csv(self):
        assert detect_format("data.csv") == "csv"

    def test_parquet(self):
        assert detect_format("data.parquet") == "parquet"

    def test_json(self):
        assert detect_format("data.json") == "json"

    def test_hdf5(self):
        assert detect_format("data.hdf5") == "hdf5"

    def test_unknown(self):
        assert detect_format("data.xyz") is None

    def test_case_insensitive(self):
        assert detect_format("DATA.CSV") == "csv"


class TestDetectFormatFromDataset:
    def test_dataframe(self):
        import pandas as pd

        assert detect_format_from_dataset(pd.DataFrame()) == "df"

    def test_dict(self):
        assert detect_format_from_dataset({"col": [1, 2]}) == "dict"

    def test_string_path(self):
        assert detect_format_from_dataset("data.csv") == "csv"
