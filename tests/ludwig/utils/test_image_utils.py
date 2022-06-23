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
from typing import Callable

import pytest
import torch

from ludwig.utils.image_utils import crop, crop_or_pad, grayscale, num_channels_in_image, pad, resize_image


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
    "img,new_size,resize_method,expected_img",
    [
        (
            torch.arange(27, dtype=torch.int).reshape(3, 3, 3),
            2,
            "crop_or_pad",
            torch.Tensor([0, 1, 3, 4, 9, 10, 12, 13, 18, 19, 21, 22]).type(torch.int).reshape(3, 2, 2),
        ),
        (
            torch.arange(27, dtype=torch.int).reshape(3, 3, 3),
            2,
            "interpolate",
            torch.Tensor([1, 2, 6, 7, 10, 12, 14, 16, 19, 20, 24, 25]).type(torch.int).reshape(3, 2, 2),
        ),
    ],
)
def test_resize_image(
    resize_image_fn: Callable, img: torch.Tensor, new_size: int, resize_method: str, expected_img: torch.Tensor
):
    output_img = resize_image_fn(img, new_size, resize_method)
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
