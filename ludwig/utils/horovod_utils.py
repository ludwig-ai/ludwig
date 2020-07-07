# coding=utf-8
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

import io
import os

import tensorflow as tf


def should_use_horovod(use_horovod):
    """Returns True if user did not specify explicitly and running with `horovodrun`."""
    if use_horovod is None:
        return has_horovodrun()
    return use_horovod


def has_horovodrun():
    """Returns True if running with `horovodrun` using Gloo or OpenMPI."""
    return 'OMPI_COMM_WORLD_RANK' in os.environ or 'HOROVOD_RANK' in os.environ


def allgather_object(obj):
    """
    Serializes and allgathers an object from all other processes.

    Arguments:
        obj: An object capable of being serialized without losing any context.

    Returns:
        The list of objects that were allgathered across all ranks.
    """
    import cloudpickle
    from horovod.tensorflow import allgather, size

    def load(byte_array):
        buf = io.BytesIO(byte_array.tobytes())
        return cloudpickle.load(buf)

    b = io.BytesIO()
    cloudpickle.dump(obj, b)

    t = tf.convert_to_tensor(bytearray(b.getvalue()), dtype=tf.uint8)
    sz = tf.convert_to_tensor([t.shape[0]], dtype=tf.int32)

    sizes = allgather(sz, name=type(obj).__name__ + '.sz').numpy()
    gathered = allgather(t, name=type(obj).__name__ + '.t').numpy()

    def select(i):
        start = sizes[i - 1] if i > 0 else 0
        end = start + sizes[i]
        return gathered[start:end]

    return [load(select(i)) for i in range(size())]
