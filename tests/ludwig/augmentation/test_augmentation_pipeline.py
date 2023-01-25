import copy
import logging
import os
import tempfile

import pytest
import torch

from ludwig.api import LudwigModel
from ludwig.constants import IMAGENET1K
from ludwig.data.dataset_synthesizer import cli_synthesize_dataset
from ludwig.features.image_feature import ImageAugmentation


# define fixture for test  image augmentation
@pytest.fixture(scope="module")
def test_image():
    # return random normal batch of images of size 2x3x32x32 [batch_size, channels, height, width]
    return torch.randn(2, 3, 32, 32)


# create training data for model training with augmentation pipeline
@pytest.fixture(scope="module")
def train_data_rgb():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # setup basic data description for training
        output_features = [
            {
                "name": "binary_output_feature",
                "type": "binary",
            },
        ]
        input_features = [
            {
                "name": "my_image",
                "type": "image",
            },
        ]

        # add parameters to generate images
        input_features[0].update(
            {
                "destination_folder": os.path.join(os.getcwd(), os.path.join(tmp_dir, "images")),
                "preprocessing": {"height": 350, "width": 350, "num_channels": 3},
            }
        )
        feature_list = input_features + output_features

        # create synthetic data
        data_dir = os.path.join(tmp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        train_fp = os.path.join(tmp_dir, "train.csv")

        cli_synthesize_dataset(16, feature_list, train_fp)

        # remove unneeded data generation parameters
        input_features[0].pop("destination_folder")

        # return training data for testing
        yield train_fp, input_features, output_features


# create training data for model training with augmentation pipeline
@pytest.fixture(scope="module")
def train_data_gray_scale():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # setup basic data description for training
        output_features = [
            {
                "name": "binary_output_feature",
                "type": "binary",
            },
        ]
        input_features = [
            {
                "name": "my_image",
                "type": "image",
            },
        ]

        # add parameters to generate images
        input_features[0].update(
            {
                "destination_folder": os.path.join(os.getcwd(), os.path.join(tmp_dir, "images")),
                "preprocessing": {"height": 350, "width": 350, "num_channels": 1},
            }
        )
        feature_list = input_features + output_features

        # create synthetic data
        data_dir = os.path.join(tmp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        train_fp = os.path.join(tmp_dir, "train.csv")

        cli_synthesize_dataset(16, feature_list, train_fp)

        # remove unneeded data generation parameters
        input_features[0].pop("destination_folder")

        # return training data for testing
        yield train_fp, input_features, output_features


# common function to run model training with augmentation pipeline
def run_augmentation_training(
    train_data,
    backend,
    encoder,
    preprocessing,
    augmentation_pipeline_ops,
):
    # unpack training data
    train_fp, input_features, output_features = train_data

    # add encoder and preprocessing specification to input feature
    input_features[0].update(
        {
            "encoder": encoder,
            "preprocessing": preprocessing,
        }
    )

    # add augmentation pipeline to input feature if specified
    test_input_features = copy.deepcopy(input_features)
    if augmentation_pipeline_ops:
        test_input_features[0].update({"augmentation": augmentation_pipeline_ops})

    config = {
        "input_features": test_input_features,
        "output_features": output_features,
        "trainer": {
            "epochs": 2,
            "batch_size": 8,
        },
        "backend": {
            "type": backend,
        },
    }

    model = LudwigModel(config, logging_level=logging.INFO)
    model.experiment(
        dataset=train_fp,
        skip_save_processed_input=True,
        skip_save_model=True,
    )


@pytest.mark.parametrize(
    "augmentation_pipeline_ops",
    [
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
def test_image_augmentation(test_image, augmentation_pipeline_ops):
    # define augmentation pipeline
    augmentation_pipeline = ImageAugmentation(augmentation_pipeline_ops)
    # apply augmentation pipeline to batch of test images
    augmentation_pipeline(test_image)


AUGMENTATION_PIPELINE_OPS = [
    False,
    True,
    [{"type": "random_blur"}, {"type": "random_rotate"}],
]

IMAGE_ENCODER = [
    {"type": "stacked_cnn"},
    {"type": "alexnet", "model_cache_dir": os.path.join(os.getcwd(), "tv_cache")},
]

IMAGE_PREPROCESSING = [
    {
        "standardize_image": None,
        "width": 300,
        "height": 300,
    },
    {
        "standardize_image": IMAGENET1K,
        "width": 300,
        "height": 300,
    },
]


@pytest.mark.parametrize("augmentation_pipeline_ops", AUGMENTATION_PIPELINE_OPS)
@pytest.mark.parametrize("encoder", IMAGE_ENCODER)
@pytest.mark.parametrize("preprocessing", IMAGE_PREPROCESSING)
def test_local_model_training_with_augmentation_pipeline(
    train_data_rgb,
    encoder,
    preprocessing,
    augmentation_pipeline_ops,
):
    run_augmentation_training(
        train_data_rgb,
        "local",
        encoder,
        preprocessing,
        augmentation_pipeline_ops,
    )


# due to the time it takes to run the tests, run only a subset of the tests
# and focus on interaction of Ludwig encoder with image preprocessing and augmentation
@pytest.mark.distributed
@pytest.mark.parametrize("augmentation_pipeline_ops", AUGMENTATION_PIPELINE_OPS)
@pytest.mark.parametrize("preprocessing", IMAGE_PREPROCESSING)
def test_ray_model_training_with_augmentation_pipeline(
    train_data_rgb,
    preprocessing,
    augmentation_pipeline_ops,
    ray_cluster_2cpu,
):
    run_augmentation_training(
        train_data_rgb,
        "ray",
        {"type": "stacked_cnn"},  # Ludwig encoder
        preprocessing,
        augmentation_pipeline_ops,
    )


# this test gray-scale image augmentation pipeline
@pytest.mark.parametrize(
    "augmentation_pipeline_ops",
    [
        False,
        True,
        [
            {"type": "random_horizontal_flip"},
            {"type": "random_vertical_flip"},
            {"type": "random_rotate"},
            {"type": "random_brightness"},
            {"type": "random_blur"},
            {"type": "random_contrast"},
        ],
    ]
)
def test_ludwig_encoder_gray_scale_image_augmentation_pipeline(
    train_data_gray_scale,
    augmentation_pipeline_ops,
):
    run_augmentation_training(
        train_data_gray_scale,
        "local",
        {"type": "stacked_cnn", "num_filters": 1},
        {},
        augmentation_pipeline_ops,
    )
