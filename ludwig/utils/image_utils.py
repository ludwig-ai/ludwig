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
import logging
import os
import sys
from math import ceil, floor

import numpy as np
# TODO(shreya): Import guard?
import torch

from ludwig.constants import CROP_OR_PAD, INTERPOLATE
from ludwig.utils.data_utils import get_abs_path
from ludwig.utils.fs_utils import open_file, is_http

logger = logging.getLogger(__name__)


def get_image_from_path(src_path, img_entry, ret_bytes=False):
    """
    skimage.io.imread() can read filenames or urls
    imghdr.what() can read filenames or bytes
    """
    if not isinstance(img_entry, str):
        return img_entry
    if is_http(img_entry):
        if ret_bytes:
            import requests
            return requests.get(img_entry, stream=True).raw.read()
        return img_entry
    if src_path or os.path.isabs(img_entry):
        return get_abs_path(src_path, img_entry)
    with open_file(img_entry, 'rb') as f:
        if ret_bytes:
            return f.read()
        return f


def is_image(src_path, img_entry):
    if not isinstance(img_entry, str):
        return False
    try:
        import imghdr

        img = get_image_from_path(src_path, img_entry, True)
        if isinstance(img, bytes):
            return imghdr.what(None, img) is not None
        return imghdr.what(img) is not None
    except:
        return False


def read_image(img) -> torch.Tensor:
    # TODO(shreya): Confirm that it's ok to switch image reader to support NCHW
    # TODO(shreya): Confirm that it's ok to read all images as RGB
    try:
        from torchvision.io import read_image, ImageReadMode
    except ImportError:
        logger.error(
            ' scikit-image is not installed. '
            'In order to install all image feature dependencies run '
            'pip install ludwig[image]'
        )
        sys.exit(-1)
    if isinstance(img, str):
        return read_image(img, mode=ImageReadMode.RGB)
    return img


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
    try:
        from skimage import img_as_ubyte
        from skimage.transform import resize
    except ImportError:
        logger.error(
            ' scikit-image is not installed. '
            'In order to install all image feature dependencies run '
            'pip install ludwig[image]'
        )
        sys.exit(-1)

    if tuple(img.shape[:2]) != new_size_typle:
        if resize_method == CROP_OR_PAD:
            return crop_or_pad(img, new_size_typle)
        elif resize_method == INTERPOLATE:
            return img_as_ubyte(resize(img, new_size_typle))
        raise ValueError(
            'Invalid image resize method: {}'.format(resize_method))
    return img


def greyscale(img):
    try:
        from skimage import img_as_ubyte
        from skimage.color import rgb2gray
    except ImportError:
        logger.error(
            ' scikit-image is not installed. '
            'In order to install all image feature dependencies run '
            'pip install ludwig[image]'
        )
        sys.exit(-1)

    return np.expand_dims(img_as_ubyte(rgb2gray(img)), axis=2)


def num_channels_in_image(img):
    if img is None or img.ndim < 2:
        raise ValueError('Invalid image data')

    if img.ndim == 2:
        return 1
    else:
        return img.shape[0]
