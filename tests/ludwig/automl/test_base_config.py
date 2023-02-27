import numpy as np
import pandas as pd
import pytest

ray = pytest.importorskip("ray")  # noqa

from ludwig.automl.base_config import get_dataset_info, get_reference_configs, is_field_boolean  # noqa
from ludwig.data.dataframe.dask import DaskEngine  # noqa
from ludwig.data.dataframe.pandas import PandasEngine  # noqa
from ludwig.schema.model_types.base import ModelConfig  # noqa
from ludwig.utils.automl.data_source import wrap_data_source  # noqa

pytestmark = pytest.mark.distributed


@pytest.fixture(scope="module")
def dummy_df():
    data = {
        "title": {
            0: " Donald Trump Sends ...Disturbing",
            1: " Drunk Bragging Trum...estigation",
            2: " Sheriff David Clark...n The Eye",
            3: " Trump Is So Obsesse...e (IMAGES)",
            4: " Pope Francis Just C...mas Speech",
        },
        "text": {
            0: "Donald Trump just co...ty Images.",
            1: "House Intelligence C...ty Images.",
            2: "On Friday, it was re...ty Images.",
            3: "On Christmas day, Do...ty Images.",
            4: "Pope Francis used hi...ty Images.",
        },
        "subject": {0: "News", 1: "News", 2: "News", 3: "News", 4: "News"},
        "date": {
            0: "December 31, 2017",
            1: "December 31, 2017",
            2: "December 30, 2017",
            3: "December 29, 2017",
            4: "December 25, 2017",
        },
        "label": {0: "Fake", 1: "Fake", 2: "Fake", 3: "Fake", 4: "Fake"},
    }

    return pd.DataFrame.from_dict(data)


@pytest.mark.parametrize(
    ("df_engine",),
    [
        pytest.param(PandasEngine(), id="pandas"),
        pytest.param(DaskEngine(_use_ray=False), id="dask", marks=pytest.mark.distributed),
    ],
)
def test_is_field_boolean(df_engine, dummy_df):
    assert np.array_equal(dummy_df.dtypes, ["object", "object", "object", "object", "object"])

    if isinstance(df_engine, DaskEngine):
        dummy_df = df_engine.df_lib.from_pandas(dummy_df, npartitions=1)

    source = wrap_data_source(dummy_df)

    for field in dummy_df.columns:
        assert not is_field_boolean(source, field)


@pytest.mark.parametrize(
    "df_engine",
    [
        pytest.param(PandasEngine(), id="pandas"),
        pytest.param(DaskEngine(_use_ray=False), id="dask", marks=pytest.mark.distributed),
    ],
)
def test_dataset_info(df_engine, dummy_df):
    assert np.array_equal(dummy_df.dtypes, ["object", "object", "object", "object", "object"])

    if isinstance(df_engine, DaskEngine):
        dummy_df = df_engine.df_lib.from_pandas(dummy_df, npartitions=1)

    ds_info = get_dataset_info(dummy_df)

    assert [f.dtype for f in ds_info.fields] == ["object", "object", "object", "object", "object"]


@pytest.mark.parametrize(
    "col,expected_dtype",
    [
        (["a", "b", "c", "d", "e", "a", "b", "b"], "object"),
        (["a", "b", "a", "b", np.nan], "object"),
        (["a", "b", "a", "b", None], "object"),
        ([True, False, True, True, ""], "object"),
        ([True, False, True, False, np.nan], "bool"),
    ],
)
def test_object_and_bool_type_inference(col, expected_dtype):
    df = pd.DataFrame({"col1": col})
    info = get_dataset_info(df)
    assert info.fields[0].dtype == expected_dtype


def test_reference_configs():
    ref_configs = get_reference_configs()
    for dataset in ref_configs["datasets"]:
        config = dataset["config"]

        # Ensure config is valid with the latest Ludwig schema
        ModelConfig.from_dict(config)
