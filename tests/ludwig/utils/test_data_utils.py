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
import pandas as pd

from ludwig.data.cache.types import CacheableDataframe
from ludwig.utils.data_utils import add_sequence_feature_column, figure_data_format_dataset, get_abs_path


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
