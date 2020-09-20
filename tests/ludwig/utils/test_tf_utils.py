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
import contextlib
from unittest.mock import Mock, patch

from ludwig.utils.tf_utils import initialize_tensorflow, _get_tf_init_params, \
    _set_tf_init_params


@contextlib.contextmanager
def clean_params():
    prev = _get_tf_init_params()
    try:
        _set_tf_init_params(None)
        yield
    finally:
        _set_tf_init_params(prev)


@patch('ludwig.utils.tf_utils.tf.config')
def test_initialize_tensorflow_only_once(mock_tf_config):
    mock_tf_config.list_physical_devices.return_value = ['gpu0', 'gpu1',
                                                         'gpu2', 'gpu3']
    with clean_params():
        # During first time initialization, set TensorFlow parallelism
        initialize_tensorflow()
        mock_tf_config.threading.set_intra_op_parallelism_threads.assert_called_once()
        mock_tf_config.threading.set_inter_op_parallelism_threads.assert_called_once()

        # Reset call counts on all threading calls
        mock_tf_config.threading.reset_mock()

        # In the second call to initialization, avoid calling these methods again, as TensorFlow
        # will raise an exception
        initialize_tensorflow()
        mock_tf_config.threading.set_intra_op_parallelism_threads.assert_not_called()
        mock_tf_config.threading.set_inter_op_parallelism_threads.assert_not_called()

    # No GPUs were specified, so this should not have been called even once
    mock_tf_config.set_visible_devices.assert_not_called()


@patch('ludwig.utils.tf_utils.tf.config')
def test_initialize_tensorflow_with_gpu_list(mock_tf_config):
    # For test purposes, these devices can be anything, we just need to be able to uniquely
    # identify them.
    mock_tf_config.list_physical_devices.return_value = ['gpu0', 'gpu1',
                                                         'gpu2', 'gpu3']
    with clean_params():
        initialize_tensorflow(gpus=[1, 2])
    mock_tf_config.set_visible_devices.assert_called_with(['gpu1', 'gpu2'],
                                                          'GPU')


@patch('ludwig.utils.tf_utils.tf.config')
def test_initialize_tensorflow_with_gpu_string(mock_tf_config):
    mock_tf_config.list_physical_devices.return_value = ['gpu0', 'gpu1',
                                                         'gpu2', 'gpu3']
    with clean_params():
        initialize_tensorflow(gpus='1,2')
    mock_tf_config.set_visible_devices.assert_called_with(['gpu1', 'gpu2'],
                                                          'GPU')


@patch('ludwig.utils.tf_utils.tf.config')
def test_initialize_tensorflow_with_gpu_int(mock_tf_config):
    mock_tf_config.list_physical_devices.return_value = ['gpu0', 'gpu1',
                                                         'gpu2', 'gpu3']
    with clean_params():
        initialize_tensorflow(gpus=1)
    mock_tf_config.set_visible_devices.assert_called_with(['gpu1'], 'GPU')


@patch('ludwig.utils.tf_utils.tf.config')
def test_initialize_tensorflow_without_gpu(mock_tf_config):
    mock_tf_config.list_physical_devices.return_value = ['gpu0', 'gpu1',
                                                         'gpu2', 'gpu3']
    with clean_params():
        initialize_tensorflow(gpus=-1)
    mock_tf_config.set_visible_devices.assert_called_with([], 'GPU')


@patch('ludwig.utils.tf_utils.tf.config')
def test_initialize_tensorflow_with_horovod(mock_tf_config):
    mock_tf_config.list_physical_devices.return_value = ['gpu0', 'gpu1',
                                                         'gpu2', 'gpu3']

    mock_hvd = Mock()
    mock_hvd.local_rank.return_value = 1
    mock_hvd.local_size.return_value = 4

    with clean_params():
        initialize_tensorflow(horovod=mock_hvd)

    mock_tf_config.set_visible_devices.assert_called_with(['gpu1'], 'GPU')


@patch('ludwig.utils.tf_utils.warnings')
@patch('ludwig.utils.tf_utils.tf.config')
def test_initialize_tensorflow_with_horovod_bad_local_rank(mock_tf_config,
                                                           mock_warnings):
    """In this scenario, the local_size 5 is out of the bounds of the GPU indices."""
    mock_tf_config.list_physical_devices.return_value = ['gpu0', 'gpu1',
                                                         'gpu2', 'gpu3']

    mock_hvd = Mock()
    mock_hvd.local_rank.return_value = 1
    mock_hvd.local_size.return_value = 5

    with clean_params():
        initialize_tensorflow(horovod=mock_hvd)

    mock_tf_config.set_visible_devices.assert_called_with([], 'GPU')
    mock_warnings.warn.assert_called()


@patch('ludwig.utils.tf_utils.tf.config')
def test_initialize_tensorflow_with_horovod_explicit_gpus(mock_tf_config):
    mock_tf_config.list_physical_devices.return_value = ['gpu0', 'gpu1',
                                                         'gpu2', 'gpu3']

    mock_hvd = Mock()
    mock_hvd.local_rank.return_value = 1
    mock_hvd.local_size.return_value = 4

    with clean_params():
        initialize_tensorflow(gpus='-1', horovod=mock_hvd)

    mock_tf_config.set_visible_devices.assert_called_with([], 'GPU')
