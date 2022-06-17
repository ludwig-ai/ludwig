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
import pandas as pd

from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split


def test_get_repeatable_train_val_test_split():

    # Test adding split with stratify
    df = pd.DataFrame(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 0],
            [5, 1],
            [6, 1],
            [7, 1],
            [8, 1],
            [9, 1],
            [10, 0],
            [11, 0],
            [12, 0],
            [13, 0],
            [14, 0],
            [15, 1],
            [16, 1],
            [17, 1],
            [18, 1],
            [19, 1],
        ],
        columns=["input", "target"],
    )
    split_df = get_repeatable_train_val_test_split(df, 'target', random_seed=42)
    assert split_df.equals(
        pd.DataFrame(
            [
                [7, 1, 0],
                [16, 1, 0],
                [5, 1, 0],
                [14, 0, 0],
                [19, 1, 0],
                [6, 1, 0],
                [11, 0, 0],
                [18, 1, 0],
                [1, 0, 0],
                [10, 0, 0],
                [2, 0, 0],
                [15, 1, 0],
                [0, 0, 0],
                [17, 1, 1],
                [12, 0, 1],
                [8, 1, 2],
                [4, 0, 2],
                [13, 0, 2],
                [3, 0, 2],
                [9, 1, 2],
            ],
            columns=["input", "target", "split"],
        )
    )

    # Test needing no change
    df = pd.DataFrame(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [5, 1, 0],
            [6, 1, 0],
            [7, 1, 0],
            [10, 0, 0],
            [11, 0, 0],
            [14, 0, 0],
            [15, 1, 0],
            [16, 1, 0],
            [18, 1, 0],
            [19, 1, 0],
            [12, 0, 1],
            [17, 1, 1],
            [3, 0, 2],
            [4, 0, 2],
            [8, 1, 2],
            [9, 1, 2],
            [13, 0, 2],
        ],
        columns=["input", "target", "split"],
    )
    split_df = get_repeatable_train_val_test_split(df, 'target', random_seed=42)
    assert split_df.equals(
        pd.DataFrame(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [5, 1, 0],
                [6, 1, 0],
                [7, 1, 0],
                [10, 0, 0],
                [11, 0, 0],
                [14, 0, 0],
                [15, 1, 0],
                [16, 1, 0],
                [18, 1, 0],
                [19, 1, 0],
                [12, 0, 1],
                [17, 1, 1],
                [3, 0, 2],
                [4, 0, 2],
                [8, 1, 2],
                [9, 1, 2],
                [13, 0, 2],
            ],
            columns=["input", "target", "split"],
        )
    )

    # Test adding only validation split
    df = pd.DataFrame(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [5, 1, 0],
            [6, 1, 0],
            [7, 1, 0],
            [10, 0, 0],
            [11, 0, 0],
            [14, 0, 0],
            [15, 1, 0],
            [16, 1, 0],
            [18, 1, 0],
            [19, 1, 0],
            [12, 0, 0],
            [17, 1, 0],
            [3, 0, 2],
            [4, 0, 2],
            [8, 1, 2],
            [9, 1, 2],
            [13, 0, 2],
        ],
        columns=["input", "target", "split"],
    )
    split_df = get_repeatable_train_val_test_split(df, 'target', random_seed=42)
    assert split_df.equals(
        pd.DataFrame(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [5, 1, 0],
                [6, 1, 0],
                [7, 1, 0],
                [10, 0, 0],
                [11, 0, 0],
                [14, 0, 0],
                [16, 1, 0],
                [19, 1, 0],
                [12, 0, 0],
                [17, 1, 0],
                [15, 1, 1],
                [18, 1, 1],
                [3, 0, 2],
                [4, 0, 2],
                [8, 1, 2],
                [9, 1, 2],
                [13, 0, 2],
            ],
            columns=["input", "target", "split"],
        )
    )

