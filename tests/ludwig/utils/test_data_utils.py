# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import functools
import json
import logging

import numpy as np
import pandas as pd
import pytest
from fsspec.config import conf

from ludwig.api import LudwigModel
from ludwig.data.cache.types import CacheableDataframe
from ludwig.data.dataset_synthesizer import build_synthetic_dataset_df
from ludwig.utils.data_utils import (
    add_sequence_feature_column,
    figure_data_format_dataset,
    get_abs_path,
    hash_dict,
    NumpyEncoder,
    PANDAS_DF,
    read_csv,
    read_html,
    read_parquet,
    sanitize_column_names,
    use_credentials,
)
from tests.integration_tests.utils import private_param

try:
    import dask.dataframe as dd
except ImportError:
    dd = None


def test_add_sequence_feature_column():
    df = pd.DataFrame([1, 2, 3, 4, 5], columns=["x"])

    add_sequence_feature_column(df, "x", 2)
    assert df.equals(
        pd.DataFrame(
            [
                [1, "1 2"],
                [2, "1 2"],
                [3, "1 2"],
                [4, "2 3"],
                [5, "3 4"],
            ],
            columns=["x", "x_feature"],
        )
    )

    add_sequence_feature_column(df, "x", 1)
    assert df.equals(
        pd.DataFrame(
            [
                [1, "1"],
                [2, "1"],
                [3, "2"],
                [4, "3"],
                [5, "4"],
            ],
            columns=["x", "x_feature"],
        )
    )

    df = pd.DataFrame([1, 2, 3, 4, 5], columns=["x"])

    add_sequence_feature_column(df, "y", 2)
    assert df.equals(pd.DataFrame([1, 2, 3, 4, 5], columns=["x"]))


def test_get_abs_path():
    assert get_abs_path("a", "b.jpg") == "a/b.jpg"
    assert get_abs_path(None, "b.jpg") == "b.jpg"


@pytest.mark.parametrize(
    "path, expected_format", [("s3://path/to.parquet ", "parquet"), ("/Users/path/to.csv \n", "csv")]
)
def test_figure_data_format_dataset_strip(path, expected_format):
    assert figure_data_format_dataset(path) == expected_format


@pytest.mark.distributed
def test_figure_data_format_dataset():
    assert figure_data_format_dataset({"a": "b"}) == dict
    assert figure_data_format_dataset(pd.DataFrame([1, 2, 3, 4, 5], columns=["x"])) == pd.DataFrame
    assert (
        figure_data_format_dataset(
            dd.from_pandas(pd.DataFrame([1, 2, 3, 4, 5], columns=["x"]), npartitions=1).reset_index()
        )
        == dd.core.DataFrame
    )
    assert (
        figure_data_format_dataset(
            CacheableDataframe(df=pd.DataFrame([1, 2, 3, 4, 5], columns=["x"]), name="test", checksum="test123")
        )
        == pd.DataFrame
    )
    assert (
        figure_data_format_dataset(
            CacheableDataframe(
                df=dd.from_pandas(pd.DataFrame([1, 2, 3, 4, 5], columns=["x"]), npartitions=1).reset_index(),
                name="test",
                checksum="test123",
            )
        )
        == dd.core.DataFrame
    )


def test_hash_dict_numpy_types():
    d = {"float32": np.float32(1)}
    assert hash_dict(d) == b"uqtgWB"


def test_use_credentials():
    conf.clear()
    with use_credentials(None):
        assert len(conf) == 0

    s3_creds = {
        "s3": {
            "client_kwargs": {
                "endpoint_url": "http://localhost:9000",
                "aws_access_key_id": "test",
                "aws_secret_access_key": "test",
            }
        }
    }
    with use_credentials(s3_creds):
        assert len(conf) == 1
        assert conf == s3_creds

    assert len(conf) == 0


def test_numpy_encoder():
    # Test Python builtin data type encoding.
    assert json.dumps(None, cls=NumpyEncoder) == "null"
    assert json.dumps({}, cls=NumpyEncoder) == "{}"
    assert json.dumps(1, cls=NumpyEncoder) == "1"
    assert json.dumps(1.0, cls=NumpyEncoder) == "1.0"
    assert json.dumps("a", cls=NumpyEncoder) == '"a"'
    assert json.dumps([0, 1, 2, 3, 4], cls=NumpyEncoder) == "[0, 1, 2, 3, 4]"
    assert json.dumps((0, 1, 2, 3, 4), cls=NumpyEncoder) == "[0, 1, 2, 3, 4]"
    assert json.dumps({0, 1, 2, 3, 4}, cls=NumpyEncoder) == "[0, 1, 2, 3, 4]"
    assert json.dumps({"a": "b"}, cls=NumpyEncoder) == '{"a": "b"}'

    # Test numpy data type encoding
    for dtype in [np.byte, np.ubyte, np.short, np.ushort, np.int32, np.int64, np.uint, np.longlong, np.ulonglong]:
        x = np.arange(5, dtype=dtype)
        assert json.dumps(x, cls=NumpyEncoder) == "[0, 1, 2, 3, 4]"
        for i in x:
            assert json.dumps(i, cls=NumpyEncoder) == f"{i}"

    for dtype in [np.half, np.single, np.double, np.longdouble]:
        x = np.arange(5, dtype=dtype)
        assert json.dumps(x, cls=NumpyEncoder) == "[0.0, 1.0, 2.0, 3.0, 4.0]"
        for i in x:
            assert json.dumps(i, cls=NumpyEncoder) == f"{i}"


def test_dataset_synthesizer_output_feature_decoder():
    config = {
        "input_features": [{"name": "sentence", "type": "text"}],
        "output_features": [{"name": "product", "type": "category"}],
        "trainer": {"epochs": 5},
        "model_type": "ecd",
    }
    build_synthetic_dataset_df(dataset_size=100, config=config)
    LudwigModel(config=config, logging_level=logging.INFO)


@pytest.mark.parametrize(
    "dataset_1k_url",
    [
        private_param(["s3://ludwig-tests/datasets/synthetic_1k.csv"]),
        private_param(["s3://ludwig-tests/datasets/synthetic_1k.parquet"]),
    ],
)
@pytest.mark.parametrize("nrows", [None, 100])
def test_chunking(dataset_1k_url, nrows):
    reader_fn = {"csv": read_csv, "parquet": functools.partial(read_parquet, df_lib=PANDAS_DF)}

    format = figure_data_format_dataset(dataset_1k_url)

    assert reader_fn[format](dataset_1k_url, nrows=nrows).shape[0] == (nrows if nrows else 1000)


@pytest.mark.parametrize(
    "df_lib", [pytest.param(pd, id="pandas"), pytest.param(dd, marks=pytest.mark.distributed, id="dask")]
)
@pytest.mark.parametrize("nrows", [None, 10])
def test_read_html(df_lib, nrows):
    HTML_DOCUMENT = """
    <!DOCTYPE html>
    <html>
    <head><title>TITLE</title></head>
    <body>
    <table>
    <th><td>Col 1</td><td>Col 2</td></th>
    <tr><td>1</td><td>2</td></tt>
    </table>
    </body>
    </html>
    """

    kwargs = {}
    if not nrows:
        kwargs["nrows"] = nrows

    read_html(HTML_DOCUMENT, df_lib, **kwargs)


def test_sanitize_column_names():
    df = pd.DataFrame(
        {
            "col.one": [1, 2, 3, 4],
            "col(two)": [4, 5, 6, 7],
            "col[]:three": [7, 8, 9, 10],
            "col 'one' (new)": [1, 2, 3, 4],
        }
    )

    df = sanitize_column_names(df)

    assert list(df.columns) == ["col_one", "col_two_", "col___three", "col _one_ _new_"]
