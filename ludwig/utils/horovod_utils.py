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
import os
import time

try:
    import horovod.tensorflow

    _HVD = horovod.tensorflow
except (ModuleNotFoundError, ImportError):
    _HVD = None


def initialize_horovod():
    if not _HVD:
        raise ValueError("Horovod backend specified, "
                         "but cannot import `horovod.tensorflow`. "
                         "Install Horovod following the instructions at: "
                         "https://github.com/horovod/horovod")
    _HVD.init()
    return _HVD


def has_horovodrun():
    """Returns True if running with `horovodrun` using Gloo or OpenMPI."""
    return 'OMPI_COMM_WORLD_RANK' in os.environ or 'HOROVOD_RANK' in os.environ


def return_first(fn):
    """Wraps function so results are only returned by the first (coordinator) rank.

    The purpose of this function is to reduce network overhead.
    """
    def wrapped(*args, **kwargs):
        res = fn(*args, **kwargs)
        return res if _HVD.rank() == 0 else None
    return wrapped
