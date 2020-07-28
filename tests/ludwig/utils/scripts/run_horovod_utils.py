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

import argparse
import os
import sys

import horovod.tensorflow as hvd

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.join(PATH_HERE, '..', '..', '..')
sys.path.insert(0, os.path.abspath(PATH_ROOT))

from ludwig.utils.horovod_utils import allgather_object

parser = argparse.ArgumentParser()
parser.add_argument('--test-name', required=True)


def test_allgather_object():
    hvd.init()
    d = {'metric_val_1': hvd.rank()}
    if hvd.rank() == 1:
        d['metric_val_2'] = 42

    results = allgather_object(d)

    assert len(results) == 2
    assert results == [
        {'metric_val_1': 0},
        {'metric_val_1': 1, 'metric_val_2': 42}
    ]


if __name__ == "__main__":
    args = parser.parse_args()
    globals()[args.test_name]()
