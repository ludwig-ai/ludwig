#! /usr/bin/env python
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

from ludwig.backend.base import Backend, LocalBackend
from ludwig.utils.horovod_utils import has_horovodrun


LOCAL_BACKEND = LocalBackend()

LOCAL = 'local'
DASK = 'dask'
HOROVOD = 'horovod'
RAY = 'ray'

ALL_BACKENDS = [LOCAL, DASK, HOROVOD, RAY]


def get_local_backend(**kwargs):
    return LocalBackend(**kwargs)


def create_dask_backend(**kwargs):
    from ludwig.backend.dask import DaskBackend
    return DaskBackend(**kwargs)


def create_horovod_backend(**kwargs):
    from ludwig.backend.horovod import HorovodBackend
    return HorovodBackend(**kwargs)


def create_ray_backend(**kwargs):
    from ludwig.backend.ray import RayBackend
    return RayBackend(**kwargs)


backend_registry = {
    LOCAL: get_local_backend,
    DASK: create_dask_backend,
    HOROVOD: create_horovod_backend,
    RAY: create_ray_backend,
    None: get_local_backend,
}


def create_backend(type, **kwargs):
    if isinstance(type, Backend):
        return type

    if type is None and has_horovodrun():
        type = HOROVOD

    return backend_registry[type](**kwargs)


def initialize_backend(backend):
    if isinstance(backend, dict):
        backend = create_backend(**backend)
    else:
        backend = create_backend(backend)
    backend.initialize()
    return backend
