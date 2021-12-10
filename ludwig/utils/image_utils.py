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
import functools
import logging
import os
import sys
from collections.abc import Iterable
from io import BytesIO
from typing import BinaryIO, List, Optional, TextIO, Tuple, Union
from urllib.error import HTTPError

import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.io import decode_image

from ludwig.constants import CROP_OR_PAD, INTERPOLATE
from ludwig.utils.data_utils import get_abs_path
from ludwig.utils.fs_utils import is_http, open_file, upgrade_http

logger = logging.getLogger(__name__)


def get_gray_default_image(height: int, width: int, num_channels: int) -> np.ndarray:
    return np.full((height, width, num_channels), 128, dtype=np.uint8)


def get_average_image(image_lst: List[np.ndarray]) -> np.array:
    return np.mean([x for x in image_lst if x is not None], axis=(0))


@functools.lru_cache(maxsize=32)
def get_image_from_http_bytes(img_entry) -> BytesIO:
    import requests

    data = requests.get(img_entry, stream=True)
    if data.status_code == 404:
        upgraded = upgrade_http(img_entry)
        if upgraded:
            logger.info(f"reading image url {img_entry} failed. upgrading to https and retrying")
            return get_image_from_http_bytes(upgraded)
        else:
            raise requests.exceptions.HTTPError(f"reading image url {img_entry} failed and cannot be upgraded to https")
    return BytesIO(data.raw.read())


def get_image_from_path(
    src_path: Union[str, torch.Tensor], img_entry: Union[str, bytes], ret_bytes: bool = False
) -> Union[BytesIO, BinaryIO, TextIO, bytes, str]:
    if not isinstance(img_entry, str):
        return img_entry
    if is_http(img_entry):
        if ret_bytes:
            # Returns BytesIO.
            return get_image_from_http_bytes(img_entry)
        return img_entry
    if src_path or os.path.isabs(img_entry):
        return get_abs_path(src_path, img_entry)
    with open_file(img_entry, "rb") as f:
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
    except Exception as e:
        logger.warning(f"While assessing potential image in is_image(), encountered exception: {e}")
        return False


# For image inference, want to bias towards both readable images, but also account for unreadable (i.e. expired) urls
# with image extensions
def is_image_score(src_path, img_entry):
    if is_image(src_path, img_entry):
        return 1
    elif isinstance(img_entry, str) and img_entry.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")):
        return 0.5
    return 0


@functools.lru_cache(maxsize=32)
def read_image(img: Union[str, bytes, BytesIO, torch.Tensor], num_channels: Optional[int] = None) -> torch.Tensor:
    """Returns a tensor with CHW format.

    If num_channels is not provided, the image is read in unchanged format. Returns None if the image could not be read.
    """
    if isinstance(img, torch.Tensor):
        return img
    if isinstance(img, str):
        return read_image_from_str(img, num_channels)
    if isinstance(img, bytes):
        with BytesIO(img) as buffer:
            buffer_view = buffer.getbuffer()
            image_tensor = decode_image(torch.frombuffer(buffer_view, dtype=torch.uint8))
            del buffer_view
            return image_tensor
    if isinstance(img, BytesIO):
        buffer_view = img.getbuffer()
        try:
            image_tensor = decode_image(torch.frombuffer(buffer_view, dtype=torch.uint8))
            del buffer_view
            return image_tensor
        except RuntimeError as e:
            logger.warning(f"Encountered torchvision error while reading {img}: {e}")
    logger.warning(f"Could not read image {img}, unsupported type {type(img)}")


@functools.lru_cache(maxsize=32)
def read_image_from_str(img: str, num_channels: Optional[int] = None) -> torch.Tensor:
    try:
        from torchvision.io import ImageReadMode, read_image
    except ImportError:
        logger.error(
            " torchvision is not installed. "
            "In order to install all image feature dependencies run "
            "pip install ludwig[image]"
        )
        sys.exit(-1)

    try:
        if num_channels == 1:
            return read_image(img, mode=ImageReadMode.GRAY)
        elif num_channels == 2:
            return read_image(img, mode=ImageReadMode.GRAY_ALPHA)
        elif num_channels == 3:
            return read_image(img, mode=ImageReadMode.RGB)
        elif num_channels == 4:
            return read_image(img, mode=ImageReadMode.RGB_ALPHA)
        else:
            return read_image(img)
    except HTTPError as e:
        upgraded = upgrade_http(img)
        if upgraded:
            logger.info(f"reading image url {img} failed due to {e}. upgrading to https and retrying")
            return read_image(upgraded)
        logger.info(f"reading image url {img} failed due to {e} and cannot be upgraded to https")
        return None
    except Exception as e:
        logger.info(f"reading image url {img} failed with error: ", e)


def pad(
    img: torch.Tensor,
    size: Union[int, Tuple[int]],
) -> torch.Tensor:
    old_size = np.array(img.shape[1:])
    pad_size = (to_np_tuple(size) - old_size) / 2
    padding = np.concatenate((np.floor(pad_size), np.ceil(pad_size)))
    padding[padding < 0] = 0
    padding = tuple(padding.astype(int).tolist())
    return F.pad(img, padding=padding, padding_mode="edge")


def crop(
    img: torch.Tensor,
    size: Union[int, Tuple[int]],
) -> torch.Tensor:
    return F.center_crop(img, output_size=size)


def crop_or_pad(img: torch.Tensor, new_size: Union[int, Tuple[int]]):
    new_size = to_np_tuple(new_size)
    if new_size.tolist() == list(img.shape[1:]):
        return img
    img = pad(img, new_size)
    img = crop(img, new_size)
    return img


def resize_image(img: torch.Tensor, new_size: Union[int, Tuple[int, int]], resize_method: str) -> torch.Tensor:
    try:
        import torchvision.transforms.functional as F
    except ImportError:
        logger.error(
            "torchvision is not installed. "
            "In order to install all image feature dependencies run "
            "pip install ludwig[image]"
        )
        sys.exit(-1)

    new_size = to_np_tuple(new_size)
    if list(img.shape[:1]) != new_size.tolist():
        if resize_method == CROP_OR_PAD:
            return crop_or_pad(img, new_size.tolist())
        elif resize_method == INTERPOLATE:
            return F.resize(img, new_size.tolist())
        raise ValueError(f"Invalid image resize method: {resize_method}")
    return img


def grayscale(img: torch.Tensor) -> torch.Tensor:
    try:
        import torchvision.transforms.functional as F
    except ImportError:
        logger.error(
            "torchvision is not installed. "
            "In order to install all image feature dependencies run "
            "pip install ludwig[image]"
        )
        sys.exit(-1)

    return F.rgb_to_grayscale(img)


def num_channels_in_image(img: torch.Tensor):
    if img is None or img.ndim < 2:
        raise ValueError("Invalid image data")

    if img.ndim == 2:
        return 1
    else:
        return img.shape[0]


def to_np_tuple(prop: Union[int, Iterable]) -> np.ndarray:
    """Creates a np array of length 2 from a Conv2D property.

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
        raise TypeError(f"prop must be int or iterable of length 2, but is {prop}.")


def get_img_output_shape(
    img_height: int,
    img_width: int,
    kernel_size: Union[int, Tuple[int]],
    stride: Union[int, Tuple[int]],
    padding: Union[int, Tuple[int], str],
    dilation: Union[int, Tuple[int]],
) -> Tuple[int]:
    """Returns the height and width of an image after a 2D img op.

    Currently supported for Conv2D, MaxPool2D and AvgPool2d ops.
    """

    if padding == "same":
        return (img_height, img_width)
    elif padding == "valid":
        padding = np.zeros(2)
    else:
        padding = to_np_tuple(padding)

    kernel_size = to_np_tuple(kernel_size)
    stride = to_np_tuple(stride)
    dilation = to_np_tuple(dilation)
    shape = np.array([img_height, img_width])

    out_shape = np.floor(((shape + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)

    return tuple(out_shape.astype(int))
