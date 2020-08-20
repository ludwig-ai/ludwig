# -*- coding: utf-8 -*-
# Copyright (c) 2019 Uber Technologies, Inc.
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
import uuid

import pytest

from ludwig.utils.data_utils import replace_file_extension
from ludwig.utils.tf_utils import initialize_tensorflow


@pytest.fixture(scope="session", autouse=True)
def init_tensorflow_cpu(request):
    """Initialize tensorflow at the start of testing to only use CPUs.

    This fixture runs once before any tests, and ensures that the main process
    running the pytests does not claim any GPU resources.

    This is critical to avoid OOM errors when running subprocesses that need GPUs (e.g., hyperopt),
    as otherwise the main process will consume all the memory and cause the subprocesses to crash.

    Run most tests eagerly as the cost of graph construction can easily increase runtime by
    and order of magnitude for small tests. Tests that execute in subprocesses, and tests
    in `test_graph_execution.py` still run in graph mode.
    """
    import tensorflow as tf
    tf.config.experimental_run_functions_eagerly(True)
    initialize_tensorflow(gpus=-1)


@pytest.fixture()
def csv_filename():
    """
    This methods returns a random filename for the tests to use for generating
    temporary data. After the data is used, all the temporary data is deleted.
    :return: None
    """
    csv_filename = uuid.uuid4().hex[:10].upper() + '.csv'
    yield csv_filename

    delete_temporary_data(csv_filename)


@pytest.fixture()
def yaml_filename():
    """
    This methods returns a random filename for the tests to use for generating
    a model definition file. After the test runs, this file will be deleted
    :return: None
    """
    yaml_filename = 'model_def_' + uuid.uuid4().hex[:10].upper() + '.yaml'
    yield yaml_filename

    if os.path.exists(yaml_filename):
        os.remove(yaml_filename)


def delete_temporary_data(csv_path):
    """
    Helper method to delete temporary data created for running tests. Deletes
    the csv and hdf5/json data (if any)
    :param csv_path: path to the csv data file
    :return: None
    """
    if os.path.isfile(csv_path):
        os.remove(csv_path)

    json_path = replace_file_extension(csv_path, 'json')
    if os.path.isfile(json_path):
        os.remove(json_path)

    hdf5_path = replace_file_extension(csv_path, 'hdf5')
    if os.path.isfile(hdf5_path):
        os.remove(hdf5_path)
