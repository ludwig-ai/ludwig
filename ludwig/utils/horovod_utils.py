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

import cloudpickle

import tensorflow as tf

from horovod.tensorflow import allgather, size


def allgather_object(obj):
    """
    Serializes and allgathers an object from all other processes.

    Arguments:
        obj: An object capable of being serialized without losing any context.

    Returns:
        The list of objects that were allgathered across all ranks.
    """
    def load(t):
        buf = io.BytesIO(t.tobytes())
        return cloudpickle.load(buf)

    b = io.BytesIO()
    cloudpickle.dump(obj, b)
    t = tf.convert_to_tensor(bytearray(b.getvalue()), dtype=tf.uint8)
    gathered = allgather(t, name=type(obj).__name__).numpy()
    return [load(gathered[i]) for i in range(size())]
