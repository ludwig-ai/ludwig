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

import os
import subprocess
import sys

import pytest

try:
    from horovod.run.runner import run as horovodrun

    USE_HOROVOD = True
except ImportError:
    USE_HOROVOD = False

# This script will run the actual test model training in parallel
TEST_SCRIPT = os.path.join(os.path.dirname(__file__), 'scripts',
                           'run_horovod_utils.py')


def _run_horovod(test_name):
    """Execute the training script across multiple workers in parallel."""
    cmdline = [
        'horovodrun',
        '-np', '2',
        sys.executable, TEST_SCRIPT,
        '--test-name', test_name,
    ]
    exit_code = subprocess.call(' '.join(cmdline), shell=True,
                                env=os.environ.copy())
    assert exit_code == 0


@pytest.mark.skipif(not USE_HOROVOD, reason='Horovod is not available')
def test_allgather_object():
    _run_horovod('test_allgather_object')
