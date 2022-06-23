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


def get_pandas_dataset_manager(**kwargs):
    from ludwig.data.dataset.pandas import PandasDatasetManager

    return PandasDatasetManager(**kwargs)


def get_ray_dataset_manager(**kwargs):
    from ludwig.data.dataset.ray import RayDatasetManager

    return RayDatasetManager(**kwargs)


dataset_registry = {
    "hdf5": get_pandas_dataset_manager,
    "ray": get_ray_dataset_manager,
    None: get_pandas_dataset_manager,
}


def create_dataset_manager(backend, cache_format, **kwargs):
    return dataset_registry[cache_format](backend=backend, **kwargs)
