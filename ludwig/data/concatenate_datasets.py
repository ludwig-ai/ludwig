#! /usr/bin/env python
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
import argparse
import logging

import numpy as np

from ludwig.backend import LOCAL_BACKEND
from ludwig.constants import SPLIT
from ludwig.utils.data_utils import read_csv

logger = logging.getLogger(__name__)


def concatenate_csv(train_csv, vali_csv, test_csv, output_csv):
    concatenated_df = concatenate_files(train_csv, vali_csv, test_csv, read_csv, LOCAL_BACKEND)

    logger.info("Saving concatenated dataset as csv..")
    concatenated_df.to_csv(output_csv, encoding="utf-8", index=False)
    logger.info("done")


def concatenate_files(train_fname, vali_fname, test_fname, read_fn, backend):
    df_lib = backend.df_engine.df_lib

    logger.info("Loading training file...")
    train_df = read_fn(train_fname, df_lib)
    logger.info("done")

    logger.info("Loading validation file..")
    vali_df = read_fn(vali_fname, df_lib) if vali_fname is not None else None
    logger.info("done")

    logger.info("Loading test file..")
    test_df = read_fn(test_fname, df_lib) if test_fname is not None else None
    logger.info("done")

    logger.info("Concatenating files..")
    concatenated_df = concatenate_df(train_df, vali_df, test_df, backend)
    logger.info("done")

    return concatenated_df


def concatenate_df(train_df, vali_df, test_df, backend):
    train_size = len(train_df)
    vali_size = len(vali_df) if vali_df is not None else 0

    concatenated_df = backend.df_engine.df_lib.concat(
        [df for df in [train_df, vali_df, test_df] if df is not None], ignore_index=True
    )

    def get_split(idx):
        if idx < train_size:
            return 0
        if idx < train_size + vali_size:
            return 1
        return 2

    concatenated_df[SPLIT] = concatenated_df.index.to_series().map(get_split).astype(np.int8)
    return concatenated_df


def concatenate_splits(train_df, vali_df, test_df, backend):
    def to_frame(df, split):
        if df is None:
            return None

        df = df.index.to_frame(name=SPLIT)
        df[SPLIT] = split
        return df

    dfs = [train_df, vali_df, test_df]
    dfs = [to_frame(df, split) for split, df in enumerate(dfs)]
    return backend.df_engine.df_lib.concat([df for df in dfs if df is not None])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate train validation and test set")

    parser.add_argument("-train", "--train_csv", help="CSV containing the training set")
    parser.add_argument("-vali", "--vali_csv", help="CSV containing the validation set")
    parser.add_argument("-test", "--test_csv", help="CSV containing the test set")

    parser.add_argument("-o", "--output_csv", help="output csv")
    args = parser.parse_args()

    concatenate_csv(args.train_csv, args.vali_csv, args.test_csv, args.output_csv)
