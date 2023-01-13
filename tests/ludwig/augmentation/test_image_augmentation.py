import pytest
import torch

from ludwig.constants import IMAGE
from ludwig.features.image_feature import get_augmentation_op


@pytest.fixture(scope="module")
def test_image():
    # return random normal image of size 2x3x32x32 [batch_size, channels, height, width]
    return torch.randn(2, 3, 32, 32)


@pytest.mark.parametrize(
    "augmentation_type, augmentation_params",
    [
        ("random_horizontal_flip", {}),
        ("random_vertical_flip", {}),
        ("random_rotate", {"degree": 45}),
        ("random_blur", {"kernel_size": 9}),
        ("random_blur", {"kernel_size": 15}),
        ("random_contrast", {"min_contrast": 0.5, "max_contrast": 1.5}),
        ("random_brightness", {"min_brightness": 0.5, "max_brightness": 1.5}),
    ],
)
def test_image_augmentation(test_image, augmentation_type, augmentation_params):
    augmentation_op_cls = get_augmentation_op(IMAGE, augmentation_type)
    augmentation_op = augmentation_op_cls(**augmentation_params)
    augmentation_op(test_image)
