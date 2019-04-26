#! /usr/bin/env python
# coding=utf-8
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
from math import floor, ceil

import os
import numpy as np
from skimage import img_as_ubyte
from skimage.transform import resize

from ludwig.constants import CROP_OR_PAD, INTERPOLATE


def pad(img, size, axis):
    old_size = img.shape[axis]
    pad_size = float(size - old_size) / 2
    pads = [(0, 0), (0, 0), (0, 0)]
    pads[axis] = (floor(pad_size), ceil(pad_size))
    return np.pad(img, pads, 'edge')


def crop(img, size, axis):
    y_min = 0
    y_max = img.shape[0]
    x_min = 0
    x_max = img.shape[1]
    if axis == 0:
        y_min = int(float(y_max - size) / 2)
        y_max = y_min + size
    else:
        x_min = int(float(x_max - size) / 2)
        x_max = x_min + size

    return img[y_min: y_max, x_min: x_max, :]


def crop_or_pad(img, new_size_tuple):
    for axis in range(2):
        if new_size_tuple[axis] != img.shape[axis]:
            if new_size_tuple[axis] > img.shape[axis]:
                img = pad(img, new_size_tuple[axis], axis)
            else:
                img = crop(img, new_size_tuple[axis], axis)
    return img


def resize_image(img, new_size_typle, resize_method):
    if tuple(img.shape[:2]) != new_size_typle:
        if resize_method == CROP_OR_PAD:
            return crop_or_pad(img, new_size_typle)
        elif resize_method == INTERPOLATE:
            return img_as_ubyte(resize(img, new_size_typle))
        raise ValueError(
            'Invalid image resize method: {}'.format(resize_method))
    return img


def get_abs_path(data_csv_path, image_path):
    if data_csv_path is not None:
        return os.path.join(data_csv_path, image_path)
    else:
        return image_path


def num_channels_in_image(img):
    if img is None or img.ndim < 2:
        raise ValueError('Invalid image data')

    if img.ndim == 2:
        return 1
    else:
        return img.shape[2]
