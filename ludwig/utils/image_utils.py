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
import logging
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.io import decode_image, ImageReadMode
from torchvision.models._api import WeightsEnum

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import CROP_OR_PAD, INTERPOLATE
from ludwig.encoders.base import Encoder
from ludwig.utils.data_utils import get_abs_path
from ludwig.utils.fs_utils import get_bytes_obj_from_path
from ludwig.utils.registry import Registry


@dataclass
class TVModelVariant:
    # Model variant identifier
    variant_id: Union[str, int]

    # TorchVision function to create model class
    create_model_function: Callable

    # Torchvision class for model weights
    model_weights: WeightsEnum


logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")


@DeveloperAPI
class ResizeChannels(torch.nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, imgs: torch.Tensor):
        original_imgs_shape = imgs.shape
        if len(original_imgs_shape) == 3:  # if shape is (C, H, W), add batch dimension
            imgs = imgs.unsqueeze(0)

        channels = imgs.shape[1]
        if channels > self.num_channels:
            # take the first `self.num_channels` channels
            imgs = imgs[:, : self.num_channels, :, :]
        elif channels < self.num_channels:
            # repeat and use the first `self.num_channels` channels
            imgs = imgs.repeat(1, (self.num_channels // channels) + 1, 1, 1)[:, : self.num_channels, :, :]

        if len(original_imgs_shape) == 3:  # if shape was (C, H, W), remove batch dimension
            return imgs[0]
        return imgs


@DeveloperAPI
def get_gray_default_image(num_channels: int, height: int, width: int) -> np.ndarray:
    return np.full((num_channels, height, width), 128, dtype=np.float32)


@DeveloperAPI
def get_average_image(image_lst: List[np.ndarray]) -> np.array:
    return np.mean([x for x in image_lst if x is not None], axis=(0), dtype=np.float32)


@DeveloperAPI
def is_image(src_path: str, img_entry: Union[bytes, str], column: str) -> bool:
    if not isinstance(img_entry, str):
        return False
    try:
        import imghdr

        path = get_abs_path(src_path, img_entry)
        bytes_obj = get_bytes_obj_from_path(path)
        if isinstance(bytes_obj, bytes):
            return imghdr.what(None, bytes_obj) is not None
        return imghdr.what(bytes_obj) is not None
    except AttributeError:
        # An AttributeError is raised when an image doesn't exist in the dataset, and we want to silence those errors.
        return False
    except Exception as e:
        logger.warning(f"While assessing potential image in is_image() for column {column}, encountered exception: {e}")
        return False


@DeveloperAPI
def is_image_score(src_path, img_entry, column: str):
    """Used for AutoML For image inference, want to bias towards both readable images, but also account for
    unreadable (i.e. expired) urls with image extensions."""
    if is_image(src_path, img_entry, column):
        return 1
    elif isinstance(img_entry, str) and img_entry.lower().endswith(IMAGE_EXTENSIONS):
        return 0.5
    return 0


@DeveloperAPI
def is_torchvision_encoder(encoder_obj: Encoder) -> bool:
    # TODO(travis): do this through an interface rather than conditional logic
    from ludwig.encoders.image.torchvision import TVBaseEncoder

    return isinstance(encoder_obj, TVBaseEncoder)


@DeveloperAPI
def get_image_read_mode_from_num_channels(num_channels: int) -> ImageReadMode:
    """Returns the torchvision.io.ImageReadMode corresponding to the number of channels.

    If num_channels is not recognized, returns ImageReadMode.UNCHANGED.
    """
    mode = ImageReadMode.UNCHANGED
    if num_channels == 1:
        mode = ImageReadMode.GRAY
    elif num_channels == 2:
        mode = ImageReadMode.GRAY_ALPHA
    elif num_channels == 3:
        mode = ImageReadMode.RGB
    elif num_channels == 4:
        mode = ImageReadMode.RGB_ALPHA
    return mode


@DeveloperAPI
def read_image_from_path(
    path: str, num_channels: Optional[int] = None, return_num_bytes=False
) -> Union[Optional[torch.Tensor], Tuple[Optional[torch.Tensor], int]]:
    """Reads image from path.

    Useful for reading from a small number of paths. For more intensive reads, use backend.read_binary_files instead. If
    `return_num_bytes` is True, returns a tuple of (image, num_bytes).
    """
    bytes_obj = get_bytes_obj_from_path(path)
    image = read_image_from_bytes_obj(bytes_obj, num_channels)
    if return_num_bytes:
        if bytes_obj is not None:
            num_bytes = len(bytes_obj)
        else:
            num_bytes = None
        return image, num_bytes
    else:
        return image


@DeveloperAPI
def read_image_from_bytes_obj(
    bytes_obj: Optional[bytes] = None, num_channels: Optional[int] = None
) -> Optional[torch.Tensor]:
    """Tries to read image as a tensor from the path.

    If the path is not decodable as a PNG, attempts to read as a numpy file. If neither of these work, returns None.
    """
    mode = get_image_read_mode_from_num_channels(num_channels)

    image = read_image_as_png(bytes_obj, mode)
    if image is None:
        image = read_image_as_numpy(bytes_obj)
    if image is None:
        warnings.warn("Unable to read image from bytes object.")
    return image


@DeveloperAPI
def read_image_as_png(
    bytes_obj: Optional[bytes] = None, mode: ImageReadMode = ImageReadMode.UNCHANGED
) -> Optional[torch.Tensor]:
    """Reads image from bytes object from a PNG file."""
    try:
        with BytesIO(bytes_obj) as buffer:
            buffer_view = buffer.getbuffer()
            if len(buffer_view) == 0:
                del buffer_view
                raise Exception("Bytes object is empty. This could be due to a failed load from storage.")
            image = decode_image(torch.frombuffer(buffer_view, dtype=torch.uint8), mode=mode)
            del buffer_view
            return image
    except Exception as e:
        warnings.warn(f"Failed to read image from PNG file. Original exception: {e}")
        return None


@DeveloperAPI
def read_image_as_numpy(bytes_obj: Optional[bytes] = None) -> Optional[torch.Tensor]:
    """Reads image from bytes object from a numpy file."""
    try:
        with BytesIO(bytes_obj) as buffer:
            image = np.load(buffer)
            return torch.from_numpy(image)
    except Exception as e:
        warnings.warn(f"Failed to read image from numpy file. Original exception: {e}")
        return None


@DeveloperAPI
def pad(
    img: torch.Tensor,
    new_size: Union[int, Tuple[int, int]],
) -> torch.Tensor:
    """torchscript-compatible implementation of pad.

    Args:
        img (torch.Tensor): image with shape [..., height, width] to pad
        new_size (Union[int, Tuple[int, int]]): size to pad to. If int, resizes to square image of that size.

    Returns:
        torch.Tensor: padded image of size [..., size[0], size[1]] or [..., size, size] if size is int.
    """
    new_size = to_tuple(new_size)
    old_size = img.shape[-2:]
    pad_size = (torch.tensor(new_size) - torch.tensor(old_size)) / 2
    padding = torch.cat((torch.floor(pad_size), torch.ceil(pad_size)))
    padding[padding < 0] = 0
    padding = [int(x) for x in padding]
    return F.pad(img, padding=padding, padding_mode="edge")


@DeveloperAPI
def crop(
    img: torch.Tensor,
    new_size: Union[int, Tuple[int, int]],
) -> torch.Tensor:
    """torchscript-compatible implementation of crop.

    Args:
        img (torch.Tensor): image with shape [..., height, width] to crop
        size (Union[int, Tuple[int, int]]): size to crop to. If int, crops to square image of that size.

    Returns:
        torch.Tensor: cropped image of size [..., size[0], size[1]] or [..., size, size] if size is int.
    """
    new_size = to_tuple(new_size)
    return F.center_crop(img, output_size=new_size)


@DeveloperAPI
def crop_or_pad(img: torch.Tensor, new_size: Union[int, Tuple[int, int]]):
    """torchscript-compatible implementation of resize using constants.CROP_OR_PAD.

    Args:
        img (torch.Tensor): image with shape [..., height, width] to resize
        new_size (Union[int, Tuple[int, int]]): size to resize to. If int, resizes to square image of that size.

    Returns:
        torch.Tensor: resized image of size [..., size[0], size[1]] or [..., size, size] if size is int.
    """
    new_size = to_tuple(new_size)
    if list(new_size) == list(img.shape[-2:]):
        return img
    img = pad(img, new_size)
    img = crop(img, new_size)
    return img


@DeveloperAPI
def resize_image(
    img: torch.Tensor,
    new_size: Union[int, Tuple[int, int]],
    resize_method: str,
    crop_or_pad_constant: str = CROP_OR_PAD,
    interpolate_constant: str = INTERPOLATE,
) -> torch.Tensor:
    """torchscript-compatible implementation of resize.

    Args:
        img (torch.Tensor): image with shape [..., height, width] to resize
        new_size (Union[int, Tuple[int, int]]): size to resize to. If int, resizes to square image of that size.
        resize_method (str): method to use for resizing. Either constants.CROP_OR_PAD or constants.INTERPOLATE.

    Returns:
        torch.Tensor: resized image of size [..., size[0], size[1]] or [..., size, size] if size is int.
    """
    new_size = to_tuple(new_size)
    if list(img.shape[-2:]) != list(new_size):
        if resize_method == crop_or_pad_constant:
            return crop_or_pad(img, new_size)
        elif resize_method == interpolate_constant:
            return F.resize(img, new_size)
        raise ValueError(f"Invalid image resize method: {resize_method}")
    return img


@DeveloperAPI
def grayscale(img: torch.Tensor) -> torch.Tensor:
    """Grayscales RGB image."""
    return F.rgb_to_grayscale(img)


@DeveloperAPI
def num_channels_in_image(img: torch.Tensor):
    """Returns number of channels in image."""
    if img is None or img.ndim < 2:
        raise ValueError("Invalid image data")

    if img.ndim == 2:
        return 1
    else:
        return img.shape[0]


@DeveloperAPI
def to_tuple(v: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Converts int or tuple to tuple of ints."""
    if torch.jit.isinstance(v, int):
        return v, v
    else:
        return v


@DeveloperAPI
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


@DeveloperAPI
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


torchvision_model_registry = Registry()


def register_torchvision_model_variants(variants: List[TVModelVariant]):
    def wrap(cls):
        # prime with empty placeholder
        torchvision_model_registry[cls.torchvision_model_type] = {}

        # register each variant
        for variant in variants:
            torchvision_model_registry[cls.torchvision_model_type][variant.variant_id] = variant
        return cls

    return wrap
