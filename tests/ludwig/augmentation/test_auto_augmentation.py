import pytest
import torch

from ludwig.constants import IMAGE
from ludwig.features.image_feature import get_augmentation_op
from ludwig.schema.features.augmentation.utils import get_augmentation_cls


@pytest.fixture(scope="module")
def test_image():
    return torch.randn(5, 3, 256, 256)


@pytest.mark.parametrize(
    "augmentation_type, augmentation_params",
    [
        ("auto_augmentation", {"method": "trivial_augment"}),
        ("auto_augmentation", {"method": "auto_augment"}),
        ("auto_augmentation", {"method": "rand_augment"}),
    ],
)
def test_auto_augmentation(test_image, augmentation_type, augmentation_params):
    aug_config = get_augmentation_cls(IMAGE, augmentation_type).from_dict(augmentation_params)
    augmentation_op_cls = get_augmentation_op(IMAGE, augmentation_type)
    augmentation_op = augmentation_op_cls(aug_config)
    augmented_image = augmentation_op(test_image)
    assert augmented_image.shape == (5, 3, 256, 256)
