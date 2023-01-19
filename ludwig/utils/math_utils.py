#! /usr/bin/env python
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
import math
from typing import List

import numpy as np


def softmax(x, temperature=1.0):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()


def int_type(number):
    if number <= np.iinfo(np.int8).max:
        return np.int8
    elif number <= np.iinfo(np.int16).max:
        return np.int16
    elif number <= np.iinfo(np.int32).max:
        return np.int32
    else:  # if number <= np.iinfo(np.int64).max:
        return np.int64


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def round2precision(val, precision: int = 0, which: str = ""):
    assert precision >= 0
    val *= 10**precision
    round_callback = round
    if which.lower() == "up":
        round_callback = math.ceil
    if which.lower() == "down":
        round_callback = math.floor
    return "{1:.{0}f}".format(precision, round_callback(val) / 10**precision)


def cumsum(x: List[int]) -> List[int]:
    results = []
    j = 0
    for i in range(0, len(x)):
        j += x[i]
        results.append(j)
    return results
