#! /usr/bin/env python
# Copyright (c) 2020 Uber Technologies, Inc.
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
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from ludwig.data.dataframe.base import DataFrameEngine
from ludwig.data.dataset.tfrecord import get_compression_ext, get_part_filename
from ludwig.utils.data_utils import save_json
from ludwig.utils.fs_utils import makedirs, upload_output_file


class PandasEngine(DataFrameEngine):
    def __init__(self, **kwargs):
        super().__init__()

    def df_like(self, df, proc_cols):
        # df argument unused for pandas, which can instantiate df directly
        return pd.DataFrame(proc_cols)

    def parallelize(self, data):
        return data

    def persist(self, data):
        return data

    def compute(self, data):
        return data

    def from_pandas(self, df):
        return df

    def map_objects(self, series, map_fn, meta=None):
        return series.map(map_fn)

    def apply_objects(self, df, apply_fn, meta=None):
        return df.apply(apply_fn, axis=1)

    def reduce_objects(self, series, reduce_fn):
        return reduce_fn(series)

    def to_parquet(self, df, path):
        df.to_parquet(path, engine="pyarrow")

    def to_tfrecord(self, df, path):
        compression_type = "GZIP"
        compression_ext = get_compression_ext(compression_type)

        makedirs(path, exist_ok=True)
        write_meta(df, path, compression_type)

        filename = os.path.join(path, get_part_filename(0, compression_ext))
        pandas_df_to_tfrecords(
            df, filename, compression_type=compression_type, compression_level=9
        )

    @property
    def array_lib(self):
        return np

    @property
    def df_lib(self):
        return pd

    @property
    def partitioned(self):
        return False

    def set_parallelism(self, parallelism):
        pass


PANDAS = PandasEngine()


# Pandas to TFRecord adapter.
# The code below is modified based on:
# https://github.com/schipiga/pandas-tfrecords/blob/master/pandas_tfrecords/to_tfrecords.py
def pandas_df_to_tfrecords(
    df, path, compression_type="GZIP", compression_level=9, columns=None
):
    schema = get_schema(df, columns)
    tfr_iters = get_tfrecords(df, schema)
    write_tfrecords(
        tfr_iters,
        path,
        compression_type=compression_type,
        compression_level=compression_level,
    )


def get_schema(df, columns=None):
    schema = {}
    if len(df) == 0:
        return schema

    for col, val in df.iloc[0].to_dict().items():
        if columns and col not in columns:
            continue

        if isinstance(val, (list, np.ndarray)):
            schema[col] = (
                lambda f: lambda x: tf.train.FeatureList(feature=[f(i) for i in x])
            )(_get_feature_func(val[0]))
        else:
            schema[col] = (lambda f: lambda x: f(x))(_get_feature_func(val))
    return schema


def write_meta(df, path, compression_type):
    meta = {"size": len(df.index), "compression_type": compression_type or ""}
    save_json(os.path.join(path, "meta.json"), meta)


def write_tfrecords(tfrecords, path, compression_type=None, compression_level=9):
    opts = {}
    if compression_type:
        opts["options"] = tf.io.TFRecordOptions(
            compression_type=compression_type,
            compression_level=compression_level,
        )
    with upload_output_file(path) as local_path:
        with tf.io.TFRecordWriter(local_path, **opts) as writer:
            for item in tfrecords:
                writer.write(item.SerializeToString())


def get_tfrecords(df, schema):
    for _, row in df.iterrows():
        features = {}
        feature_lists = {}

        for col, val in row.items():
            f = schema[col](val)

            if type(f) is tf.train.FeatureList:
                feature_lists[col] = f

            if type(f) is tf.train.Feature:
                features[col] = f

        context = tf.train.Features(feature=features)
        if feature_lists:
            ex = tf.train.SequenceExample(
                context=context,
                feature_lists=tf.train.FeatureLists(feature_list=feature_lists),
            )
        else:
            ex = tf.train.Example(features=context)
        yield ex


def _get_feature_func(val):
    if isinstance(val, (bytes, str)):
        return _bytes_feature

    if isinstance(val, (int, np.integer, np.bool_)):
        return _int64_feature

    if isinstance(val, (float, np.floating)):
        return _float_feature
    raise Exception(f"Unsupported type {type(val)!r}")


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    if isinstance(value, str):
        value = str.encode(value)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
