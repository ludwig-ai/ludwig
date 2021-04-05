# -*- coding: utf-8 -*-
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
import json
import os
import platform
import shlex
import subprocess
import sys

import pytest
import tensorflow as tf

try:
    from horovod.common.util import nccl_built
except ImportError:
    HOROVOD_AVAILABLE = False
else:
    HOROVOD_AVAILABLE = True

from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import sequence_feature
from tests.integration_tests.utils import ENCODERS

# This script will run the actual test model training in parallel
TEST_SCRIPT = os.path.join(os.path.dirname(__file__), 'scripts',
                           'run_train_horovod.py')


def _nccl_available():
    if not HOROVOD_AVAILABLE:
        return False

    try:
        return nccl_built()
    except AttributeError:
        return False
    except RuntimeError:
        return False


def _run_horovod(csv_filename, **ludwig_kwargs):
    """Execute the training script across multiple workers in parallel."""
    input_features, output_features, rel_path = _prepare_data(csv_filename)
    cmdline = [
        'horovodrun',
        '-np', '2',
        sys.executable, TEST_SCRIPT,
        '--rel-path', rel_path,
        '--input-features', shlex.quote(json.dumps(input_features)),
        '--output-features', shlex.quote(json.dumps(output_features)),
        '--ludwig-kwargs', shlex.quote(json.dumps(ludwig_kwargs))
    ]
    exit_code = subprocess.call(' '.join(cmdline), shell=True,
                                env=os.environ.copy())
    assert exit_code == 0


def _prepare_data(csv_filename):
    # Single sequence input, single category output
    input_features = [sequence_feature(reduce_output='sum')]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    input_features[0]['encoder'] = ENCODERS[0]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    return input_features, output_features, rel_path


@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Horovod is not supported on Windows")
@pytest.mark.distributed
def test_horovod_implicit(csv_filename):
    """Test Horovod running without `backend='horovod'`."""
    _run_horovod(csv_filename)


@pytest.mark.skipif(platform.system() == "Windows",
                    reason="Horovod is not supported on Windows")
@pytest.mark.skipif(not _nccl_available(),
                    reason="test requires Horovod with NCCL support")
@pytest.mark.skipif(not tf.test.is_gpu_available(cuda_only=True),
                    reason="test requires multi-GPU machine")
@pytest.mark.distributed
def test_horovod_gpu_memory_limit(csv_filename):
    """Test Horovod with explicit GPU memory limit set."""
    ludwig_kwargs = dict(
        gpu_memory_limit=128
    )
    _run_horovod(csv_filename, **ludwig_kwargs)
