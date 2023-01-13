import pytest
import torch

from ludwig.features.image_feature import AugmentationPipeline


# define fixture for test  image augmentation
@pytest.fixture(scope="module")
def test_image():
    # return random normal batch of images of size 2x3x32x32 [batch_size, channels, height, width]
    return torch.randn(2, 3, 32, 32)


@pytest.mark.parametrize(
    "augmentation_pipeline_ops",
    [
        [],
        [{"type": "random_horizontal_flip"}],
        [
            {"type": "random_vertical_flip"},
            {"type": "random_rotate", "degree": 45},
        ],
        [
            {"type": "random_horizontal_flip"},
            {"type": "random_vertical_flip"},
            {"type": "random_rotate", "degree": 45},
            {"type": "random_brightness"},
            {"type": "random_blur", "kernel_size": 9},
            {"type": "random_contrast"},
        ],
    ],
)
# test image augmentation pipeline
def test_augmentation_pipeline(test_image, augmentation_pipeline_ops):
    # define augmentation pipeline
    augmentation_pipeline = AugmentationPipeline(augmentation_pipeline_ops)
    # apply augmentation pipeline to batch of test images
    augmentation_pipeline(test_image)
