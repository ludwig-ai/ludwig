import tempfile

import pytest

from ludwig.constants import TEXT
from ludwig.utils.data_utils import read_csv

try:
    import dask.dataframe as dd

    from ludwig.automl.automl import create_auto_config, create_auto_config_with_dataset_profile
except ImportError:
    pass
from ludwig.automl.automl import create_auto_config, create_auto_config_with_dataset_profile


CSV_CONTENT = """
name,gender,lives_in_sf
Jessica,f,
Jim,m,FALSE
"""


@pytest.mark.distributed
def test_mixed_csv_data_source(ray_cluster_2cpu):
    try:
        temp = tempfile.NamedTemporaryFile(mode="w+")
        temp.write(CSV_CONTENT)
        temp.seek(0)
        ds = read_csv(temp.name, dtype=None)
        df = dd.from_pandas(ds, npartitions=1)
        config = create_auto_config(dataset=df, target=[], time_limit_s=3600, tune_for_memory=False)
        assert len(config["input_features"]) == 2
        assert config["input_features"][0]["type"] == TEXT
        assert config["input_features"][1]["type"] == TEXT
    finally:
        temp.close()


@pytest.mark.distributed
def test_mixed_csv_data_source_with_profile(ray_cluster_2cpu):
    try:
        temp = tempfile.NamedTemporaryFile(mode="w+")
        temp.write(CSV_CONTENT)
        temp.seek(0)
        ds = read_csv(temp.name, dtype=None)
        df = dd.from_pandas(ds, npartitions=1)
        config = create_auto_config_with_dataset_profile(dataset=df, target=[])
        assert len(config["input_features"]) == 3
    finally:
        temp.close()
