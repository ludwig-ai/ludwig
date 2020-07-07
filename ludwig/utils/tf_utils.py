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
import tensorflow as tf


def sequence_length_3D(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def sequence_length_2D(sequence):
    used = tf.sign(tf.abs(sequence))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


# Convert a dense matrix into a sparse matrix (for e.g. edit_distance)
def to_sparse(tensor, lengths, max_length):
    mask = tf.sequence_mask(lengths, max_length)
    indices = tf.cast(tf.where(tf.equal(mask, True)), tf.int64)
    values = tf.cast(tf.boolean_mask(tensor, mask), tf.int32)
    shape = tf.cast(tf.shape(tensor), tf.int64)
    return tf.SparseTensor(indices, values, shape)
