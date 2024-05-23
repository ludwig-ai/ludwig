#! /usr/bin/env python
# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
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
import tifffile
import torch
import torchvision.transforms.functional as F
from torchvision.io import decode_image, ImageReadMode
from torchvision.models._api import WeightsEnum

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import CROP_OR_PAD, IMAGE_MAX_CLASSES, INTERPOLATE
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

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif")


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
def is_bytes_image(bytes_obj) -> bool:
    import imghdr

    return imghdr.what(None, bytes_obj) is not None


@DeveloperAPI
def is_image_score(path):
    return int(isinstance(path, str) and path.lower().endswith(IMAGE_EXTENSIONS))


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
    if bytes_obj is None:
        return None
    mode = get_image_read_mode_from_num_channels(num_channels)

    image = read_image_as_png(bytes_obj, mode)
    if image is None:
        image = read_image_as_numpy(bytes_obj)
    if image is None:
        image = read_image_as_tif(bytes_obj)
    if image is None:
        warnings.warn("Unable to read image from bytes object.")
    return image


@DeveloperAPI
def read_image_as_png(bytes_obj: bytes, mode: ImageReadMode = ImageReadMode.UNCHANGED) -> Optional[torch.Tensor]:
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
def read_image_as_numpy(bytes_obj: bytes) -> Optional[torch.Tensor]:
    """Reads image from bytes object from a numpy file."""
    try:
        with BytesIO(bytes_obj) as buffer:
            image = np.load(buffer)
            return torch.from_numpy(image)
    except Exception as e:
        warnings.warn(f"Failed to read image from numpy file. Original exception: {e}")
        return None


@DeveloperAPI
def read_image_as_tif(bytes_obj: bytes) -> Optional[torch.Tensor]:
    """Reads image from bytes object from a tif file."""
    try:
        with BytesIO(bytes_obj) as buffer:
            image = tifffile.imread(buffer)
            if image.dtype == np.uint16:
                image = image.astype(np.int32)
            image = torch.from_numpy(image)
            if len(image.shape) == 2:
                image = torch.unsqueeze(image, dim=0)
            return image
    except Exception as e:
        warnings.warn(f"Failed to read image from tif file. Original exception: {e}")
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
def get_unique_channels(
    image_sample: List[torch.Tensor],
    num_channels: int,
    num_classes: int = None,
) -> torch.Tensor:
    """Returns a tensor of unique channel values from a list of images.
    Args:
        image_sample: A list of images of dimensions [C x H x W] or [H x W], where C is the channel dimension
        num_channels: The expected number of channels
        num_classes: The expected number of classes or None

    Return:
        channel_class_map: A tensor mapping channel values to classes, where dim=0 is the class.
    """
    n_images = 0
    no_new_class = 0
    channel_class_map = None
    for img in image_sample:
        if img.ndim < 2:
            raise ValueError("Invalid image dimensions {img.ndim}")
        if img.ndim == 2:
            img = img.unsqueeze(0)
        if num_channels == 1 and num_channels_in_image(img) != 1:
            img = grayscale(img)
        if num_classes == 2 and num_channels_in_image(img) == 1:
            img = img.type(torch.float32) / 255
            img = img.round() * 255
            img = img.type(torch.uint8)

        img = img.flatten(1, 2)
        img = img.permute(1, 0)
        uniq_chans = img.unique(dim=0)

        if channel_class_map is None:
            channel_class_map = uniq_chans
        else:
            channel_class_map = torch.concat((channel_class_map, uniq_chans)).unique(dim=0)
        if channel_class_map.shape[0] > IMAGE_MAX_CLASSES:
            raise ValueError(
                f"Images inferred num classes {channel_class_map.shape[0]} exceeds " f"max classes {IMAGE_MAX_CLASSES}."
            )

        n_images += 1
        if n_images % 25 == 0:
            logger.info(f"Processed the first {n_images} images inferring {channel_class_map.shape[0]} classes...")

        if channel_class_map.shape[0] == uniq_chans.shape[0]:
            no_new_class += 1
            if no_new_class >= 4 and channel_class_map.shape[0] == num_classes:
                break  # early loop exit
        else:
            no_new_class = 0

    logger.info(f"Inferred {channel_class_map.shape[0]} classes from the first {n_images} images.")
    return channel_class_map.type(torch.uint8)


@DeveloperAPI
def get_class_mask_from_image(
    channel_class_map: torch.Tensor,
    img: torch.Tensor,
) -> torch.Tensor:
    """Returns a masked image where each mask value is the channel class of the input.
    Args:
        channel_class_map: A tensor mapping channel values to classes, where dim=0 is the class.
        img: An input image of dimensions [C x H x W] or [H x W], where C is the channel dimension

    Return:
        [mask] A masked image of dimensions [H x W] where each value is the channel class of the input
    """
    num_classes = channel_class_map.shape[0]
    mask = torch.full((img.shape[-2], img.shape[-1]), num_classes, dtype=torch.uint8)
    if img.ndim == 2:
        img = img.unsqueeze(0)
    if num_classes == 2 and num_channels_in_image(img) == 1:
        img = img.type(torch.float32) / 255
        img = img.round() * 255
        img = img.type(torch.uint8)
    img = img.permute(1, 2, 0)
    for nclass, value in enumerate(channel_class_map):
        mask[(img == value).all(-1)] = nclass

    if torch.any(mask.ge(num_classes)):
        raise ValueError(
            f"Image channel could not be mapped to a class because an unknown channel value was detected. "
            f"{num_classes} classes were inferred from the first set of images. This image has a channel "
            f"value that was not previously seen in the first set of images. Check preprocessing parameters "
            f"for image resizing, num channels, num classes and num samples. Image resizing may affect "
            f"channel values. "
        )

    return mask


@DeveloperAPI
def get_image_from_class_mask(
    channel_class_map: torch.Tensor,
    mask: np.ndarray,
) -> np.ndarray:
    """Returns an image with channel values determined from a corresponding mask.
    Args:
        channel_class_map: An tensor mapping channel values to classes, where dim=0 is the class.
        mask: A masked image of dimensions [H x W] where each value is the channel class of the final image

    Return:
        [img] An image of dimensions [C x H x W], where C is the channel dimension
    """
    mask = torch.from_numpy(mask)
    img = torch.zeros(channel_class_map.shape[1], mask.shape[-2], mask.shape[-1], dtype=torch.uint8)
    img = img.permute(1, 2, 0)
    mask = mask.unsqueeze(0)
    mask = mask.permute(1, 2, 0)
    for nclass, value in enumerate(channel_class_map):
        img[(mask == nclass).all(-1)] = value
    img = img.permute(2, 0, 1)

    return img.numpy()


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
    if type(prop) is int:
        return np.ones(2).astype(int) * prop
    elif isinstance(prop, Iterable) and len(prop) == 2:
        return np.array(list(prop)).astype(int)
    elif type(prop) is np.ndarray and prop.size == 2:
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
