"""Unit tests for FileBasedPreprocessor, ParquetPreprocessor, and data_format_preprocessor_registry."""

import pytest

from ludwig.data.preprocessing import (
    data_format_preprocessor_registry,
    DataFramePreprocessor,
    DictPreprocessor,
    FileBasedPreprocessor,
    HDF5Preprocessor,
    ParquetPreprocessor,
)
from ludwig.utils.data_utils import (
    CSV_FORMATS,
    FEATHER_FORMATS,
    HDF5_FORMATS,
    JSON_FORMATS,
    PARQUET_FORMATS,
    TSV_FORMATS,
)


class TestFileBasedPreprocessorInit:
    def test_stores_read_fn(self):
        read_fn = lambda path, df_lib: df_lib.read_csv(path)
        p = FileBasedPreprocessor(read_fn)
        assert p._read_fn is read_fn

    def test_distinct_instances_per_format(self):
        csv_preprocessors = [data_format_preprocessor_registry[fmt] for fmt in CSV_FORMATS]
        assert all(isinstance(p, FileBasedPreprocessor) for p in csv_preprocessors)
        # All CSV formats map to the same instance (dict.fromkeys semantics)
        assert len({id(p) for p in csv_preprocessors}) == 1

    def test_different_formats_get_different_instances(self):
        csv_inst = data_format_preprocessor_registry[next(iter(CSV_FORMATS))]
        tsv_inst = data_format_preprocessor_registry[next(iter(TSV_FORMATS))]
        assert csv_inst is not tsv_inst

    def test_json_instance_differs_from_csv(self):
        csv_inst = data_format_preprocessor_registry[next(iter(CSV_FORMATS))]
        json_inst = data_format_preprocessor_registry[next(iter(JSON_FORMATS))]
        assert csv_inst is not json_inst


class TestDataFormatPreprocessorRegistry:
    def test_all_csv_formats_registered(self):
        for fmt in CSV_FORMATS:
            assert fmt in data_format_preprocessor_registry

    def test_all_tsv_formats_registered(self):
        for fmt in TSV_FORMATS:
            assert fmt in data_format_preprocessor_registry

    def test_all_json_formats_registered(self):
        for fmt in JSON_FORMATS:
            assert fmt in data_format_preprocessor_registry

    def test_all_parquet_formats_registered(self):
        for fmt in PARQUET_FORMATS:
            assert fmt in data_format_preprocessor_registry

    def test_all_feather_formats_registered(self):
        for fmt in FEATHER_FORMATS:
            assert fmt in data_format_preprocessor_registry

    def test_all_hdf5_formats_registered(self):
        for fmt in HDF5_FORMATS:
            assert fmt in data_format_preprocessor_registry

    def test_parquet_maps_to_parquet_preprocessor(self):
        for fmt in PARQUET_FORMATS:
            assert isinstance(data_format_preprocessor_registry[fmt], ParquetPreprocessor)

    def test_hdf5_maps_to_hdf5_preprocessor(self):
        for fmt in HDF5_FORMATS:
            assert isinstance(data_format_preprocessor_registry[fmt], HDF5Preprocessor)

    def test_file_based_formats_use_file_based_preprocessor(self):
        file_based_formats = CSV_FORMATS | TSV_FORMATS | JSON_FORMATS | FEATHER_FORMATS
        for fmt in file_based_formats:
            inst = data_format_preprocessor_registry[fmt]
            assert isinstance(inst, FileBasedPreprocessor), f"{fmt} should use FileBasedPreprocessor, got {type(inst)}"

    def test_dict_format_uses_dict_preprocessor(self):
        assert isinstance(data_format_preprocessor_registry["dict"], DictPreprocessor)

    def test_dataframe_format_uses_dataframe_preprocessor(self):
        assert isinstance(data_format_preprocessor_registry["df"], DataFramePreprocessor)


class TestParquetPreprocessorInit:
    def test_is_file_based_preprocessor(self):
        assert isinstance(ParquetPreprocessor(), FileBasedPreprocessor)

    def test_uses_read_parquet(self):
        from ludwig.utils.data_utils import read_parquet

        p = ParquetPreprocessor()
        assert p._read_fn is read_parquet


class TestParquetPreprocessorPrepareProcessedData:
    def test_records_training_path(self):
        from ludwig.utils.data_utils import DATA_TRAIN_PARQUET_FP

        p = ParquetPreprocessor()
        metadata = {}
        p.prepare_processed_data(
            features=[],
            training_set="/tmp/train.parquet",
            training_set_metadata=metadata,
        )
        assert metadata[DATA_TRAIN_PARQUET_FP] == "/tmp/train.parquet"

    def test_does_not_overwrite_existing_training_path(self):
        from ludwig.utils.data_utils import DATA_TRAIN_PARQUET_FP

        p = ParquetPreprocessor()
        metadata = {DATA_TRAIN_PARQUET_FP: "/original/train.parquet"}
        p.prepare_processed_data(
            features=[],
            training_set="/new/train.parquet",
            training_set_metadata=metadata,
        )
        # Should not overwrite when key already exists
        assert metadata[DATA_TRAIN_PARQUET_FP] == "/original/train.parquet"

    def test_nonexistent_test_set_is_dropped(self, tmp_path):
        from ludwig.utils.data_utils import DATA_TEST_PARQUET_FP

        p = ParquetPreprocessor()
        metadata = {}
        training, test, val, meta = p.prepare_processed_data(
            features=[],
            training_set=str(tmp_path),
            test_set="/nonexistent/path.parquet",
            training_set_metadata=metadata,
        )
        # Non-existent paths should be treated as None
        assert test is None
        assert DATA_TEST_PARQUET_FP not in metadata

    def test_existing_test_set_is_recorded(self, tmp_path):
        from ludwig.utils.data_utils import DATA_TEST_PARQUET_FP

        test_file = tmp_path / "test.parquet"
        test_file.write_bytes(b"")  # create the file so path_exists returns True

        p = ParquetPreprocessor()
        metadata = {}
        training, test, val, meta = p.prepare_processed_data(
            features=[],
            training_set=str(tmp_path),
            test_set=str(test_file),
            training_set_metadata=metadata,
        )
        assert meta[DATA_TEST_PARQUET_FP] == str(test_file)


class TestHDF5PreprocessorPrepareProcessedData:
    def test_raises_on_no_dataset_or_training_set(self):
        p = HDF5Preprocessor()
        with pytest.raises(ValueError, match="One of `dataset` or `training_set` must be not None"):
            p.prepare_processed_data(features=[], training_set_metadata={"key": "val"})

    def test_raises_on_empty_metadata(self, tmp_path):
        fake_hdf5 = tmp_path / "data.hdf5"
        fake_hdf5.write_bytes(b"")
        p = HDF5Preprocessor()
        with pytest.raises(ValueError, match="training_set_metadata must not be None"):
            p.prepare_processed_data(features=[], dataset=str(fake_hdf5), training_set_metadata=None)

    def test_raises_on_empty_dict_metadata(self, tmp_path):
        p = HDF5Preprocessor()
        with pytest.raises(ValueError, match="training_set_metadata must not be None"):
            p.prepare_processed_data(features=[], dataset="/any/path.hdf5", training_set_metadata={})
