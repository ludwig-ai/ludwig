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

import pytest

from ludwig.utils.horovod_utils import allgather_object

try:
    from horovod.run.runner import run as horovodrun

    USE_HOROVOD = True
except ImportError:
    USE_HOROVOD = False


@pytest.mark.skipif(not USE_HOROVOD, reason='Horovod is not available')
def test_allgather_object():
    def fn():
        import horovod.tensorflow as hvd
        hvd.init()
        d = {'metric_val_1': hvd.rank()}
        if hvd.rank() == 1:
            d['metric_val_2'] = 42
        return allgather_object(d)

    results = horovodrun(fn, np=2)
    assert len(results) == 2
    assert results[0] == results[1]
    assert results[0] == [
        {'metric_val_1': 0},
        {'metric_val_1': 1, 'metric_val_2': 42}
    ]
