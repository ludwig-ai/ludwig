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
import multiprocessing
import warnings

import tensorflow as tf

_TF_INIT_PARAMS = None


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


def initialize_tensorflow(gpus=None,
                          gpu_memory_limit=None,
                          allow_parallel_threads=True,
                          horovod=None):
    use_horovod = horovod is not None
    param_tuple = (gpus, gpu_memory_limit, allow_parallel_threads, use_horovod)
    if _TF_INIT_PARAMS is not None:
        if _TF_INIT_PARAMS != param_tuple:
            warnings.warn(
                'TensorFlow has already been initialized. Changes to `gpus`, '
                '`gpu_memory_limit`, and `allow_parallel_threads` will be ignored. '
                'Start a new Python process to modify these values.')
        return

    # For reproducivility / determinism, set parallel threads to 1.
    # For performance, set to 0 to allow TensorFlow to select the best value automatically.
    tf.config.threading.set_intra_op_parallelism_threads(
        0 if allow_parallel_threads else 1)
    tf.config.threading.set_inter_op_parallelism_threads(
        0 if allow_parallel_threads else 1)

    gpu_devices = tf.config.list_physical_devices('GPU')
    if horovod is not None and gpus is None:
        if 0 < len(gpu_devices) < horovod.local_size():
            warnings.warn(
                f'Horovod: disabling GPU support! This host is running with '
                f'{horovod.local_size()} worker processes but only {len(gpu_devices)} '
                f'GPUs. To enable GPU training, reduce the number of worker processes '
                f'on this host to match the number of GPUs.')
            gpus = [-1]
        else:
            gpus = [horovod.local_rank()]

    if isinstance(gpus, int):
        gpus = [gpus]
    elif isinstance(gpus, str):
        gpus = gpus.strip()
        gpus = [int(g) for g in gpus.split(",")]

    if gpus and len(gpus) == 1 and gpus[0] == -1:
        # CUDA_VISIBLE_DEVICES syntax for disabling all GPUs
        tf.config.set_visible_devices([], 'GPU')
    else:
        # Allow memory growth and set memory limit. Regardless of whether we do this
        # before or after setting visible devices, TensorFlow will allocate a small
        # amount of memory per device.
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
            if gpu_memory_limit is not None:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=gpu_memory_limit)])

        # Set visible devices so GPU utilization is isolated
        # (no GPU contention between workers).
        if gpus and gpu_devices:
            local_devices = [gpu_devices[g] for g in gpus]
            tf.config.set_visible_devices(local_devices, 'GPU')

    _set_tf_init_params(param_tuple)


def _set_tf_init_params(params):
    global _TF_INIT_PARAMS
    _TF_INIT_PARAMS = params


def _get_tf_init_params():
    return _TF_INIT_PARAMS


def get_available_gpus_child_process(gpus_ids_queue):
    gpu_devices = tf.config.list_physical_devices('GPU')
    gpu_ids = [gpu.name.split(':')[-1] for gpu in gpu_devices]
    gpus_ids_queue.put(gpu_ids)


def get_available_gpus():
    ctx = multiprocessing.get_context('spawn')
    gpus_list_queue = ctx.Queue()
    proc_get_gpus = ctx.Process(
        target=get_available_gpus_child_process, args=(gpus_list_queue,))
    proc_get_gpus.start()
    proc_get_gpus.join()
    gpus_list = gpus_list_queue.get()
    return gpus_list


def get_available_gpus_cuda_string():
    gpus = get_available_gpus()
    if len(gpus) == 0:
        return None
    return ','.join(gpus)
