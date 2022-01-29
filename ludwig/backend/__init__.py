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

import logging
import os

from ludwig.backend.base import Backend, LocalBackend
from ludwig.utils.horovod_utils import has_horovodrun

logger = logging.getLogger(__name__)

try:
    import ray as _ray
except Exception as e:
    logger.warning(f"import ray failed with exception: {e}")
    _ray = None


LOCAL_BACKEND = LocalBackend()

LOCAL = "local"
DASK = "dask"
HOROVOD = "horovod"
RAY = "ray"

ALL_BACKENDS = [LOCAL, DASK, HOROVOD, RAY]


def _has_ray():
    # Temporary workaround to prevent tests from automatically using the Ray backend. Taken from
    # https://stackoverflow.com/questions/25188119/test-if-code-is-executed-from-within-a-py-test-session
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False

    if _ray is None:
        return False

    if _ray.is_initialized():
        return True

    try:
        _ray.init("auto", ignore_reinit_error=True)
        return True
    except Exception as e:
        logger.error(f"ray.init() failed: {e}")
        return False


def get_local_backend(**kwargs):
    return LocalBackend(**kwargs)


def create_horovod_backend(**kwargs):
    from ludwig.backend.horovod import HorovodBackend

    return HorovodBackend(**kwargs)


def create_ray_backend(**kwargs):
    from ludwig.backend.ray import RayBackend

    return RayBackend(**kwargs)


backend_registry = {
    LOCAL: get_local_backend,
    HOROVOD: create_horovod_backend,
    RAY: create_ray_backend,
    None: get_local_backend,
}


def create_backend(type, **kwargs):
    if isinstance(type, Backend):
        return type

    if type is None and _has_ray():
        type = RAY
    elif type is None and has_horovodrun():
        type = HOROVOD

    return backend_registry[type](**kwargs)


def initialize_backend(backend):
    if isinstance(backend, dict):
        backend = create_backend(**backend)
    else:
        backend = create_backend(backend)
    backend.initialize()
    return backend
