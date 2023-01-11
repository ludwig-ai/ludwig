import logging
import os

import pytest
import torch
from torchvision.models import resnet18, ResNet18_Weights

from ludwig.api import LudwigModel
from ludwig.data.dataset_synthesizer import cli_synthesize_dataset


# pytest fixture to do one time setup of required data
@pytest.fixture(scope="module")
def setup_data(tmp_path_factory):
    # setup location for training data
    data_dir = tmp_path_factory.mktemp("data", numbered=False)
    train_fp = os.path.join(data_dir, "train.csv")

    # setup local cache to torchvision model to avoid multiple downloads
    tv_cache = tmp_path_factory.mktemp("tv_cache", numbered=False)

    # describe synthetic data to create
    feature_list = [
        {"name": "binary_output_feature", "type": "binary"},
        {
            "name": "image",
            "type": "image",
            "destination_folder": os.path.join(data_dir, "images"),
            "preprocessing": {"height": 600, "width": 600, "num_channels": 3},
        },
    ]

    # create synthetic data
    cli_synthesize_dataset(10, feature_list, train_fp)

    return train_fp, str(tv_cache)


@pytest.mark.parametrize("trainable", [True, False])
def test_trainable_torchvision_layers(setup_data, trainable):
    # retrieve data setup from fixture
    train_fp, tv_cache = setup_data

    config = {
        "input_features": [
            {
                "name": "image",
                "type": "image",
                "encoder": {
                    "type": "resnet",
                    "model_variant": 18,
                    "model_cache_dir": tv_cache,
                    "trainable": trainable,
                },
            },
        ],
        "output_features": [
            {
                "name": "binary_output_feature",
                "type": "binary",
            }
        ],
        "trainer": {
            "epochs": 2,
        },
    }

    model = LudwigModel(config, logging_level=logging.INFO)

    _, _, output_dir = model.train(dataset=train_fp, skip_save_processed_input=True)

    # instantiate native torchvision to get original parameter values
    os.environ["TORCH_HOME"] = tv_cache
    tv_model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # replace last layer to match image encoder setup
    tv_model.fc = torch.nn.Identity()

    # compare Ludwig image encoder parameter against original native torchvision weights
    # if trainable is True, parameters should differ, otherwise all parameters should be unchanged
    if trainable:
        for p1, p2 in zip(model.model.input_features["image"].encoder_obj.model.parameters(), tv_model.parameters()):
            assert not torch.all(p1 == p2)
    else:
        for p1, p2 in zip(model.model.input_features["image"].encoder_obj.model.parameters(), tv_model.parameters()):
            assert torch.all(p1 == p2)
