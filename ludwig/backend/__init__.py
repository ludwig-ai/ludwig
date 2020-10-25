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

from ludwig.backend.base import Backend, CompositeBackend, LocalBackend
from ludwig.backend.ray import RayBackend


LOCAL_BACKEND = LocalBackend()


backend_registry = {
    'ray': RayBackend,
    'local': LocalBackend,
    None: LocalBackend,
}


def create_backend(name):
    return backend_registry[name]()
