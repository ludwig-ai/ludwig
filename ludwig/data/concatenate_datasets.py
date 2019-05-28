#! /usr/bin/env python
# coding=utf-8
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
import pandas as pd

from ludwig.utils.data_utils import read_csv


logger = logging.getLogger(__name__)


def concatenate(train_csv, vali_csv, test_csv, output_csv):
    concatenated_df = concatenate_csv(train_csv, vali_csv, test_csv)

    logger.info('Saving concatenated csv..')
    concatenated_df.to_csv(output_csv, encoding='utf-8', index=False)
    logger.info('done')


def concatenate_csv(train_csv, vali_csv, test_csv):
    logger.info('Loading training csv...')
    train_df = read_csv(train_csv)
    logger.info('done')

    logger.info('Loading validation csv..')
    vali_df = read_csv(vali_csv) if vali_csv is not None else None
    logger.info('done')

    logger.info('Loading test csv..')
    test_df = read_csv(test_csv) if test_csv is not None else None
    logger.info('done')

    logger.info('Concatenating csvs..')
    concatenated_df = concatenate_df(train_df, vali_df, test_df)
    logger.info('done')

    return concatenated_df


def concatenate_df(train_df, vali_df, test_df):
    train_size = len(train_df)
    vali_size = len(vali_df) if vali_df is not None else 0
    test_size = len(test_df) if test_df is not None else 0
    concatenated_df = pd.concat([train_df, vali_df, test_df], ignore_index=True)
    split = np.array(
        [0] * train_size + [1] * vali_size + [2] * test_size,
        dtype=np.int8
    )
    concatenated_df = concatenated_df.assign(split=pd.Series(split).values)
    return concatenated_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Concatenate train validation and test set'
    )

    parser.add_argument(
        '-train',
        '--train_csv',
        help='CSV containing the training set'
    )
    parser.add_argument(
        '-vali',
        '--vali_csv',
        help='CSV containing the validation set'
    )
    parser.add_argument(
        '-test',
        '--test_csv',
        help='CSV containing the test set'
    )

    parser.add_argument('-o', '--output_csv', help='output csv')
    args = parser.parse_args()

    concatenate(args.train_csv, args.vali_csv, args.test_csv, args.output_csv)
