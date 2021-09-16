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
from collections.abc import Iterable
from typing import BinaryIO, Optional, TextIO, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as F

from ludwig.constants import CROP_OR_PAD, INTERPOLATE
from ludwig.utils.data_utils import get_abs_path
from ludwig.utils.fs_utils import open_file, is_http

logger = logging.getLogger(__name__)


# TODO(shreya): Confirm output type.
def get_image_from_path(
    src_path: Union[str, torch.Tensor],
    img_entry: Union[str, bytes],
    ret_bytes: bool = False
) -> Union[BinaryIO, TextIO, bytes]:
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


def is_image(src_path: str, img_entry: Union[bytes, str]) -> bool:
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


def read_image(img: Union[str, torch.Tensor], num_channels: Optional[int] = None) -> torch.Tensor:
    """ Returns a tensor with CHW format.
    
    If num_channels is not provided, he image is read in unchanged format.
    """

    try:
        from torchvision.io import read_image, ImageReadMode
    except ImportError:
        logger.error(
            ' torchvision is not installed. '
            'In order to install all image feature dependencies run '
            'pip install ludwig[image]'
        )
        sys.exit(-1)
    if isinstance(img, str):
        if num_channels == 1:
            return read_image(img, mode=ImageReadMode.GRAY)
        elif num_channels == 2:
            return read_image(img, mode=ImageReadMode.GRAY_ALPHA)
        elif num_channels == 3:
            return read_image(img, mode=ImageReadMode.RGB)
        elif num_channels == 4:
            return read_image(img, mode=ImageReadMode.RGB_ALPHA)
        else:
            raise ValueError(f'Invalid num_channels={num_channels}, value must be one of 1, 2, 3, 4.')
    return img


def pad(
        img: torch.Tensor,
        size: Union[int, Tuple[int]],
) -> torch.Tensor:
    old_size = np.array(img.shape[1:])
    pad_size = (to_np_tuple(size) - old_size) / 2
    padding = np.concatenate((np.floor(pad_size), np.ceil(pad_size)))
    padding[padding < 0] = 0
    padding = tuple(padding.astype(int).tolist())
    return F.pad(img, padding=padding, padding_mode='edge')


def crop(
        img: torch.Tensor,
        size: Union[int, Tuple[int]],
) -> torch.Tensor:
    return F.center_crop(img, output_size=size)


def crop_or_pad(
        img: torch.Tensor,
        new_size: Union[int, Tuple[int]]
):
    new_size = to_np_tuple(new_size)
    if new_size.tolist() == list(img.shape[1:]):
        return img
    img = pad(img, new_size)
    img = crop(img, new_size)
    return img


def resize_image(
        img: torch.Tensor,
        new_size: Union[int, Tuple[int]],
        resize_method: str
):
    try:
        import torchvision.transforms.functional as F
    except ImportError:
        logger.error(
            'torchvision is not installed. '
            'In order to install all image feature dependencies run '
            'pip install ludwig[image]'
        )
        sys.exit(-1)

    new_size = to_np_tuple(new_size)
    if list(img.shape[:1]) != new_size.tolist():
        if resize_method == CROP_OR_PAD:
            return crop_or_pad(img, new_size.tolist())
        elif resize_method == INTERPOLATE:
            return F.resize(img, new_size.tolist())
        raise ValueError(
            'Invalid image resize method: {}'.format(resize_method))
    return img


def grayscale(img: torch.Tensor) -> torch.Tensor:
    try:
        import torchvision.transforms.functional as F
    except ImportError:
        logger.error(
            'torchvision is not installed. '
            'In order to install all image feature dependencies run '
            'pip install ludwig[image]'
        )
        sys.exit(-1)

    return F.rgb_to_grayscale(img)


def num_channels_in_image(img: torch.Tensor):
    if img is None or img.ndim < 2:
        raise ValueError('Invalid image data')

    if img.ndim == 2:
        return 1
    else:
        return img.shape[0]


def to_np_tuple(prop: Union[int, Iterable]) -> np.ndarray:
    """ Creates a np array of length 2 from a Conv2D property.

    E.g., stride=(2, 3) gets converted into np.array([2, 3]), where the
    height_stride = 2 and width_stride = 3. stride=2 gets converted into
    np.array([2, 2]).
    """
    if type(prop) == int:
        return np.ones(2).astype(int) * prop
    elif isinstance(prop, Iterable) and len(prop) == 2:
        return np.array(list(prop)).astype(int)
    elif type(prop) == np.ndarray and prop.size == 2:
        return prop.astype(int)
    else:
        raise TypeError(f'prop must be int or tuple of length 2, but is {prop}.')


def get_img_output_shape(
    img_height: int,
    img_width: int,
    kernel_size: Union[int, Tuple[int]],
    stride: Union[int, Tuple[int]],
    padding: Union[int, Tuple[int], str],
    dilation: Union[int, Tuple[int]],
) -> Tuple[int]:
    """ Returns the height and width of an image after a 2D img op.

    Currently supported for Conv2D, MaxPool2D and AvgPool2d ops.
    """

    if padding == 'same':
        return (img_height, img_width)
    elif padding == 'valid':
        padding = np.zeros(2)
    else:
        padding = to_np_tuple(padding)

    kernel_size = to_np_tuple(kernel_size)
    stride = to_np_tuple(stride)
    dilation = to_np_tuple(dilation)
    shape = np.array([img_height, img_width])

    out_shape = np.floor(
        ((shape + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
    )

    return tuple(out_shape.astype(int))
