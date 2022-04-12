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
import dask.dataframe as dd
import numpy as np
import pandas as pd
from fsspec.config import conf

from ludwig.data.cache.types import CacheableDataframe
from ludwig.utils.data_utils import (
    add_sequence_feature_column,
    figure_data_format_dataset,
    get_abs_path,
    hash_dict,
    use_credentials,
)


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
