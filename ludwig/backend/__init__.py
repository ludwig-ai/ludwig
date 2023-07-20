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

import contextlib
import logging
import os

from ludwig.api_annotations import DeveloperAPI
from ludwig.backend.base import Backend, LocalBackend
from ludwig.utils.horovod_utils import has_horovodrun

logger = logging.getLogger(__name__)


# TODO: remove LOCAL_BACKEND as a global constant, replace with singleton LocalBackend.shared_instance().
LOCAL_BACKEND = LocalBackend.shared_instance()


LOCAL = "local"
DASK = "dask"
HOROVOD = "horovod"
DEEPSPEED = "deepspeed"
RAY = "ray"

ALL_BACKENDS = [LOCAL, DASK, HOROVOD, DEEPSPEED, RAY]


def _has_ray():
    # Temporary workaround to prevent tests from automatically using the Ray backend. Taken from
    # https://stackoverflow.com/questions/25188119/test-if-code-is-executed-from-within-a-py-test-session
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False

    try:
        import ray
    except ImportError:
        return False

    if ray.is_initialized():
        return True

    try:
        ray.init("auto", ignore_reinit_error=True)
        return True
    except Exception:
        return False


def get_local_backend(**kwargs):
    return LocalBackend(**kwargs)


def create_horovod_backend(**kwargs):
    from ludwig.backend.horovod import HorovodBackend

    return HorovodBackend(**kwargs)


def create_deepspeed_backend(**kwargs):
    from ludwig.backend.deepspeed import DeepSpeedBackend

    return DeepSpeedBackend(**kwargs)


def create_ray_backend(**kwargs):
    from ludwig.backend.ray import RayBackend

    return RayBackend(**kwargs)


backend_registry = {
    LOCAL: get_local_backend,
    HOROVOD: create_horovod_backend,
    DEEPSPEED: create_deepspeed_backend,
    RAY: create_ray_backend,
    None: get_local_backend,
}


@DeveloperAPI
def create_backend(type, **kwargs):
    if isinstance(type, Backend):
        return type

    if type is None and _has_ray():
        type = RAY
    elif type is None and has_horovodrun():
        type = HOROVOD

    return backend_registry[type](**kwargs)


@DeveloperAPI
def initialize_backend(backend):
    if isinstance(backend, dict):
        backend = create_backend(**backend)
    else:
        backend = create_backend(backend)
    backend.initialize()
    return backend


@contextlib.contextmanager
def provision_preprocessing_workers(backend):
    if backend.BACKEND_TYPE == RAY:
        with backend.provision_preprocessing_workers():
            yield
    else:
        yield
