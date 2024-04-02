import re
from contextlib import nullcontext as no_error_raised

import pytest

from ludwig.api import LudwigModel
from ludwig.constants import TRAINER
from ludwig.encoders.image.torchvision import TVEfficientNetEncoder
from ludwig.schema.trainer import BaseTrainerConfig
from ludwig.utils.misc_utils import set_random_seed
from ludwig.utils.trainer_utils import freeze_layers_regex
from tests.integration_tests.utils import category_feature, generate_data, image_feature

RANDOM_SEED = 130


@pytest.mark.parametrize(
    "regex",
    [
        r"(features\.1.*|features\.2.*|features\.3.*|model\.features\.4\.1\.block\.3\.0\.weight)",
        r"(features\.1.*|features\.2\.*|features\.3.*)",
        r"(features\.4\.0\.block|features\.4\.\d+\.block)",
        r"(features\.5\.*|features\.6\.*|features\.7\.*)",
        r"(features\.8\.\d+\.weight|features\.8\.\d+\.bias)",
    ],
)
def test_tv_efficientnet_freezing(regex):
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVEfficientNetEncoder(
        model_variant="b0", use_pretrained=True, saved_weights_in_checkpoint=True, trainable=True
    )

    config = BaseTrainerConfig(layers_to_freeze_regex=regex)
    freeze_layers_regex(config, pretrained_model)
    for name, param in pretrained_model.named_parameters():
        if re.search(re.compile(regex), name):
            assert not param.requires_grad
        else:
            assert param.requires_grad


def test_frozen_tv_training(tmpdir, csv_filename):
    input_features = [image_feature(tmpdir)]
    output_features = [category_feature()]

    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {
            "layers_to_freeze_regex": r"(features\.1.*|features\.2.*|model\.features\.4\.1\.block\.3\.0\.weight)",
            "epochs": 1,
            "train_steps": 1,
        },
        "encoder": {"type": "efficientnet", "use_pretrained": False},
    }

    training_data_csv_path = generate_data(config["input_features"], config["output_features"], csv_filename)
    model = LudwigModel(config)

    with no_error_raised():
        model.experiment(
            dataset=training_data_csv_path,
            skip_save_training_description=True,
            skip_save_training_statistics=True,
            skip_save_model=True,
            skip_save_progress=True,
            skip_save_log=True,
            skip_save_processed_input=True,
        )
