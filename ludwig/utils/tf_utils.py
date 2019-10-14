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


def get_tf_config(gpus=None, gpu_fraction=1, horovod=None,
                  allow_parallel_threads=True):
    intra_op_parallelism_threads = 2  # defult in tensorflow
    inter_op_parallelism_threads = 5  # defult in tensorflow
    if not allow_parallel_threads:
        # this is needed for reproducibility
        intra_op_parallelism_threads = 1
        inter_op_parallelism_threads = 1

    if gpus is not None:
        if gpu_fraction > 0 and gpu_fraction < 1:
            # this is the source of freezing in tensorflow 1.3.1
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=gpu_fraction,
                allow_growth=True)
        else:
            gpu_options = tf.GPUOptions(allow_growth=True)
            # allow_growth=True is needed for a weird behavior with CUDA 10
            # https://github.com/tensorflow/tensorflow/issues/24828
        if isinstance(gpus, int):
            gpus = [gpus]
        gpu_options.visible_device_list = ','.join(str(g) for g in gpus)
        tf_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            intra_op_parallelism_threads=intra_op_parallelism_threads,
            inter_op_parallelism_threads=inter_op_parallelism_threads,
            gpu_options=gpu_options
        )
    else:
        tf_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            intra_op_parallelism_threads=intra_op_parallelism_threads,
            inter_op_parallelism_threads=inter_op_parallelism_threads,
            gpu_options=tf.GPUOptions(allow_growth=True)
        )

    if horovod is not None:
        tf_config.gpu_options.visible_device_list = str(horovod.local_rank())

    return tf_config
