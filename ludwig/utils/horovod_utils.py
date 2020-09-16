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

ON_MASTER = True


def configure_horovod(use_horovod):
    hvd = None
    if should_use_horovod(use_horovod):
        _HVD.init()
        hvd = _HVD
    set_on_master(use_horovod)
    return hvd


def should_use_horovod(use_horovod):
    """Returns True if user did not specify explicitly and running with `horovodrun`."""
    if use_horovod is None:
        return has_horovodrun()
    return use_horovod


def has_horovodrun():
    """Returns True if running with `horovodrun` using Gloo or OpenMPI."""
    return 'OMPI_COMM_WORLD_RANK' in os.environ or 'HOROVOD_RANK' in os.environ


def broadcast_return(fn):
    """Returns the result of calling `fn` on master, broadcasted to all other ranks.

    Specifically, `fn` is only executed on master, but its result is returned by every
    rank by broadcasting the return value from master.
    """
    result = fn() if is_on_master() else None
    if _HVD:
        name = f'broadcast_return_{int(time.time())}'
        result = _HVD.broadcast_object(result, name=name)
    return result


def set_on_master(use_horovod):
    global ON_MASTER
    if should_use_horovod(use_horovod):
        try:
            _HVD.init()
            ON_MASTER = _HVD.rank() == 0
        except ImportError:
            raise ValueError("use_horovod parameter specified, "
                             "but cannot import horovod.tensorflow. "
                             "Install horovod following the instructions at: "
                             " https://github.com/horovod/horovod")
    else:
        ON_MASTER = True


def is_on_master():
    return ON_MASTER
