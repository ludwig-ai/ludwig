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
from typing import Callable, List

import pytest
import torch
import torchvision.transforms.functional as F

from ludwig.utils.image_utils import (
    crop,
    crop_or_pad,
    get_class_mask_from_image,
    get_image_from_class_mask,
    get_unique_channels,
    grayscale,
    is_image_score,
    num_channels_in_image,
    pad,
    read_image_as_tif,
    resize_image,
    ResizeChannels,
)


@pytest.mark.parametrize("pad_fn", [pad, torch.jit.script(pad)])
@pytest.mark.parametrize(
    "img,size,padded_img",
    [
        (
            torch.arange(12, dtype=torch.int).reshape(3, 2, 2),
            4,
            torch.Tensor(
                [
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    5,
                    5,
                    6,
                    6,
                    7,
                    7,
                    6,
                    6,
                    7,
                    7,
                    8,
                    8,
                    9,
                    9,
                    8,
                    8,
                    9,
                    9,
                    10,
                    10,
                    11,
                    11,
                    10,
                    10,
                    11,
                    11,
                ]
            )
            .type(torch.int)
            .reshape(3, 4, 4),
        )
    ],
)
def test_pad(pad_fn: Callable, img: torch.Tensor, size: int, padded_img: torch.Tensor):
    output_img = pad_fn(img, size)
    assert torch.equal(output_img, padded_img)


@pytest.mark.parametrize("crop_fn", [crop, torch.jit.script(crop)])
@pytest.mark.parametrize(
    "img,size,cropped_img",
    [
        (
            torch.arange(27, dtype=torch.int).reshape(3, 3, 3),
            2,
            torch.Tensor([0, 1, 3, 4, 9, 10, 12, 13, 18, 19, 21, 22]).type(torch.int).reshape(3, 2, 2),
        )
    ],
)
def test_crop(crop_fn: Callable, img: torch.Tensor, size: int, cropped_img: torch.Tensor):
    output_img = crop_fn(img, size)
    assert torch.equal(output_img, cropped_img)


@pytest.mark.parametrize("crop_or_pad_fn", [crop_or_pad, torch.jit.script(crop_or_pad)])
@pytest.mark.parametrize(
    "img,new_size,expected_img",
    [
        (
            torch.arange(12, dtype=torch.int).reshape(3, 2, 2),
            4,
            torch.Tensor(
                [
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    4,
                    4,
                    5,
                    5,
                    6,
                    6,
                    7,
                    7,
                    6,
                    6,
                    7,
                    7,
                    8,
                    8,
                    9,
                    9,
                    8,
                    8,
                    9,
                    9,
                    10,
                    10,
                    11,
                    11,
                    10,
                    10,
                    11,
                    11,
                ]
            )
            .type(torch.int)
            .reshape(3, 4, 4),
        ),
        (
            torch.arange(27, dtype=torch.int).reshape(3, 3, 3),
            2,
            torch.Tensor([0, 1, 3, 4, 9, 10, 12, 13, 18, 19, 21, 22]).type(torch.int).reshape(3, 2, 2),
        ),
    ],
)
def test_crop_or_pad(crop_or_pad_fn: Callable, img: torch.Tensor, new_size: int, expected_img: torch.Tensor):
    output_image = crop_or_pad_fn(img, new_size)
    assert torch.equal(output_image, expected_img)


@pytest.mark.parametrize("resize_image_fn", [resize_image, torch.jit.script(resize_image)])
@pytest.mark.parametrize(
    "img,new_size,resize_method",
    [
        (
            torch.arange(27, dtype=torch.int).reshape(3, 3, 3),
            2,
            "crop_or_pad",
        ),
        (
            torch.arange(27, dtype=torch.int).reshape(3, 3, 3),
            2,
            "interpolate",
        ),
    ],
)
def test_resize_image(resize_image_fn: Callable, img: torch.Tensor, new_size: int, resize_method: str):
    # Get the expected output from the underlying function
    if resize_method == "crop_or_pad":
        expected_img = crop_or_pad(img, new_size)
    else:
        expected_img = F.resize(img, new_size)

    output_img = resize_image_fn(img, new_size, resize_method)

    # Test that resize_image is equivalent to the underlying function output
    assert torch.equal(output_img, expected_img)


@pytest.mark.parametrize("grayscale_fn", [grayscale, torch.jit.script(grayscale)])
@pytest.mark.parametrize(
    "input_img,grayscale_img",
    [(torch.arange(12).reshape(3, 2, 2).type(torch.int), torch.Tensor([[[3, 4], [5, 6]]]).type(torch.int))],
)
def test_grayscale(grayscale_fn: Callable, input_img: torch.Tensor, grayscale_img: torch.Tensor):
    output_img = grayscale_fn(input_img)
    assert torch.equal(output_img, grayscale_img)


def test_num_channels_in_image():
    image_2d = torch.randint(0, 1, (10, 10))
    image_3d = torch.randint(0, 1, (3, 10, 10))
    assert num_channels_in_image(image_2d) == 1
    assert num_channels_in_image(image_3d) == 3

    with pytest.raises(ValueError):
        num_channels_in_image(torch.rand(5))
        num_channels_in_image(None)


@pytest.mark.parametrize("image_shape", [(1, 10, 10), (3, 10, 10), (5, 10, 10)])
@pytest.mark.parametrize("num_channels_expected", [1, 2, 3, 4])
def test_ResizeChannels_module(image_shape, num_channels_expected):
    image = torch.randint(0, 1, image_shape)
    fn = ResizeChannels(num_channels_expected)
    assert fn(image).shape == tuple([num_channels_expected] + list(image_shape[1:]))


@pytest.mark.parametrize("image_shape", [(2, 1, 10, 10), (2, 3, 10, 10), (2, 5, 10, 10)])
@pytest.mark.parametrize("num_channels_expected", [1, 2, 3, 4])
def test_ResizeChannels_module_with_batch_dim(image_shape, num_channels_expected):
    image = torch.randint(0, 1, image_shape)
    fn = ResizeChannels(num_channels_expected)
    assert fn(image).shape == tuple([image_shape[0], num_channels_expected] + list(image_shape[2:]))


def test_read_image_as_tif():
    img_bytes = b"II*\x00\x0c\x00\x00\x00\x05 \x8c\xe5\x10\x00\x00\x01\x03\x00\x01\x00\x00\x00\x02\x00\x00\x00\x01\x01\x03\x00\x01\x00\x00\x00\x02\x00\x00\x00\x02\x01\x03\x00\x01\x00\x00\x00\x08\x00\x00\x00\x03\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\x06\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\x11\x01\x04\x00\x01\x00\x00\x00\x08\x00\x00\x00\x12\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\x15\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\x16\x01\x03\x00\x01\x00\x00\x00\x80\x00\x00\x00\x17\x01\x04\x00\x01\x00\x00\x00\x04\x00\x00\x00\x1a\x01\x05\x00\x01\x00\x00\x00\xd2\x00\x00\x00\x1b\x01\x05\x00\x01\x00\x00\x00\xda\x00\x00\x00\x1c\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\x1d\x01\x02\x00\x07\x00\x00\x00\xe2\x00\x00\x00(\x01\x03\x00\x01\x00\x00\x00\x02\x00\x00\x00S\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00H\x00\x00\x00\x01\x00\x00\x00H\x00\x00\x00\x01\x00\x00\x004.tiff\x00"  # noqa: E501
    tensor = read_image_as_tif(img_bytes)
    assert tensor is not None
    assert tensor.equal(torch.tensor([[[5, 32], [140, 229]]], dtype=torch.uint8))


@pytest.mark.parametrize(
    "extension, score",
    [
        ("data.png", 1),
        ("/home/peter/data.jpg", 1),
        ("./data/file.jpeg", 1),
        ("new.tiff", 1),
        ("b.tif", 1),
        (".bmp", 1),
        ("a.gif", 1),
        ("b.tif", 1),
        ("audio.wav", 0),
        (".png/video.mp4", 0),
    ],
)
def test_is_image_score(extension: str, score: int):
    assert is_image_score(extension) == score


@pytest.mark.parametrize(
    "img_list,num_channels,num_classes,expected_class_map",
    [
        (
            [
                torch.Tensor([0, 0, 8, 8, 120, 120, 180, 180, 230, 230, 255, 255]).type(torch.uint8).reshape(3, 2, 2),
                torch.Tensor([1, 2, 3, 4, 131, 132, 133, 134, 241, 242, 243, 244]).type(torch.uint8).reshape(3, 2, 2),
            ],
            3,
            None,
            torch.Tensor(
                [[0, 120, 230], [8, 180, 255], [1, 131, 241], [2, 132, 242], [3, 133, 243], [4, 134, 244]]
            ).type(torch.uint8),
        ),
        (
            [
                torch.Tensor([0, 255, 255, 0, 255, 255, 255, 0, 0, 0, 255, 255, 0, 255, 255])
                .type(torch.uint8)
                .reshape(1, 3, 5),
            ],
            1,
            None,
            torch.Tensor([[0], [255]]).type(torch.uint8),
        ),
        (
            [
                torch.Tensor([0, 31, 17, 185, 192, 173, 55, 76, 24, 128, 255, 238]).type(torch.uint8).reshape(3, 4),
            ],
            1,
            2,
            torch.Tensor([[0], [255]]).type(torch.uint8),
        ),
    ],
)
def test_unique_channels(
    img_list: List[torch.Tensor], num_channels: int, num_classes: int, expected_class_map: torch.Tensor
):
    channel_class_map = get_unique_channels(img_list, num_channels, num_classes)

    channel_class_map, _ = channel_class_map.sort(dim=0)
    expected_class_map, _ = expected_class_map.sort(dim=0)
    assert torch.equal(channel_class_map, expected_class_map)


@pytest.mark.parametrize(
    "img,channel_class_map,expected_mask",
    [
        (
            torch.Tensor([1, 2, 3, 4, 131, 132, 133, 134, 241, 242, 243, 244]).type(torch.uint8).reshape(3, 2, 2),
            torch.Tensor(
                [[0, 120, 230], [8, 180, 255], [1, 131, 241], [2, 132, 242], [3, 133, 243], [4, 134, 244]]
            ).type(torch.uint8),
            torch.Tensor([2, 3, 4, 5]).type(torch.uint8).reshape(2, 2),
        ),
        (
            torch.Tensor([0, 255, 255, 0, 255, 255, 255, 0, 0, 0, 255, 255, 0, 255, 255])
            .type(torch.uint8)
            .reshape(1, 3, 5),
            torch.Tensor([[0], [255]]).type(torch.uint8),
            torch.Tensor([0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1]).type(torch.uint8).reshape(3, 5),
        ),
        (
            torch.Tensor([0, 31, 17, 185, 192, 173, 55, 76, 24, 128, 255, 238]).type(torch.uint8).reshape(3, 4),
            torch.Tensor([[0], [255]]).type(torch.uint8),
            torch.Tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]).type(torch.uint8).reshape(3, 4),
        ),
    ],
)
def test_class_mask_from_image(img: torch.Tensor, channel_class_map: torch.Tensor, expected_mask: torch.Tensor):
    mask = get_class_mask_from_image(channel_class_map, img)
    assert torch.equal(mask, expected_mask)


@pytest.mark.parametrize(
    "mask,channel_class_map,expected_img",
    [
        (
            torch.Tensor([0, 0, 1, 1]).type(torch.uint8).reshape(2, 2),
            torch.Tensor(
                [[0, 120, 230], [8, 180, 255], [1, 131, 241], [2, 132, 242], [3, 133, 243], [4, 134, 244]]
            ).type(torch.uint8),
            torch.Tensor([0, 0, 8, 8, 120, 120, 180, 180, 230, 230, 255, 255]).type(torch.uint8).reshape(3, 2, 2),
        ),
        (
            torch.Tensor([2, 3, 4, 5]).type(torch.uint8).reshape(2, 2),
            torch.Tensor(
                [[0, 120, 230], [8, 180, 255], [1, 131, 241], [2, 132, 242], [3, 133, 243], [4, 134, 244]]
            ).type(torch.uint8),
            torch.Tensor([1, 2, 3, 4, 131, 132, 133, 134, 241, 242, 243, 244]).type(torch.uint8).reshape(3, 2, 2),
        ),
        (
            torch.Tensor([0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1]).type(torch.uint8).reshape(3, 5),
            torch.Tensor([[0], [255]]).type(torch.uint8),
            torch.Tensor([0, 255, 255, 0, 255, 255, 255, 0, 0, 0, 255, 255, 0, 255, 255])
            .type(torch.uint8)
            .reshape(1, 3, 5),
        ),
    ],
)
def test_image_from_class_mask(mask: torch.Tensor, channel_class_map: torch.Tensor, expected_img: torch.Tensor):
    img = get_image_from_class_mask(channel_class_map, mask.numpy())
    assert torch.equal(torch.from_numpy(img), expected_img)
