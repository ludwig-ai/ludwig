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
import numpy as np
import pytest

from ludwig.utils.data_utils import get_abs_path
from ludwig.utils.image_utils import num_channels_in_image

image_2d = np.random.randint(0, 1, (10, 10))
image_3d = np.random.randint(0, 1, (10, 10, 3))


def test_num_channels_in_image():
    assert num_channels_in_image(image_2d) == 1
    assert num_channels_in_image(image_3d) == 3

    with pytest.raises(ValueError):
        num_channels_in_image(np.arange(5))
        num_channels_in_image(None)


def test_get_abs_path():
    assert get_abs_path('a', 'b.jpg') == 'a/b.jpg'
    assert get_abs_path(None, 'b.jpg') == 'b.jpg'
