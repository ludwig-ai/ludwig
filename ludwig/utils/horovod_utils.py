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


def should_use_horovod(use_horovod):
    """Returns True if user did not specify explicitly and running with `horovodrun`."""
    if use_horovod is None:
        return has_horovodrun()
    return use_horovod


def has_horovodrun():
    """Returns True if running with `horovodrun` using Gloo or OpenMPI."""
    return 'OMPI_COMM_WORLD_RANK' in os.environ or 'HOROVOD_RANK' in os.environ
