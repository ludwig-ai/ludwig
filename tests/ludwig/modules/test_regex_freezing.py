import logging
import re
from contextlib import nullcontext as no_error_raised

import pytest

from ludwig.api import LudwigModel
from ludwig.constants import (
    BASE_MODEL,
    BATCH_SIZE,
    EPOCHS,
    GENERATION,
    INPUT_FEATURES,
    MODEL_LLM,
    MODEL_TYPE,
    OUTPUT_FEATURES,
    TRAINER,
    TYPE,
)
from ludwig.encoders.image.torchvision import TVEfficientNetEncoder
from ludwig.schema.trainer import ECDTrainerConfig
from ludwig.utils.misc_utils import set_random_seed
from ludwig.utils.trainer_utils import freeze_layers_regex
from tests.integration_tests.utils import category_feature, generate_data, image_feature, text_feature

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
        model_variant="b0", use_pretrained=False, saved_weights_in_checkpoint=True, trainable=True
    )

    config = ECDTrainerConfig(layers_to_freeze_regex=regex)
    freeze_layers_regex(config, pretrained_model)
    for name, param in pretrained_model.named_parameters():
        if re.search(re.compile(regex), name):
            assert not param.requires_grad
        else:
            assert param.requires_grad


def test_llm_freezing(tmpdir, csv_filename):
    input_features = [text_feature(name="input", encoder={"type": "passthrough"})]
    output_features = [text_feature(name="output")]

    train_df = generate_data(input_features, output_features, filename=csv_filename, num_examples=25)

    config = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: "hf-internal-testing/tiny-random-GPTJForCausalLM",
        INPUT_FEATURES: [text_feature(name="input", encoder={"type": "passthrough"})],
        OUTPUT_FEATURES: [text_feature(name="output")],
        TRAINER: {TYPE: "finetune", BATCH_SIZE: 8, EPOCHS: 1, "layers_to_freeze_regex": r"(h\.0\.attn\.*)"},
        GENERATION: {"pad_token_id": 0},
    }

    model = LudwigModel(config, logging_level=logging.INFO)

    output_directory: str = str(tmpdir)
    model.train(dataset=train_df, output_directory=output_directory, skip_save_processed_input=False)

    for name, p in model.model.named_parameters():
        if "h.0.attn" in name:
            assert not p.requires_grad
        else:
            assert p.requires_grad


def test_frozen_tv_training(tmpdir, csv_filename):
    input_features = [
        image_feature(tmpdir, encoder={"type": "efficientnet", "use_pretrained": False, "model_variant": "b0"})
    ]
    output_features = [category_feature()]

    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {
            "layers_to_freeze_regex": r"(features\.1.*|features\.2\.*|features\.3.*)",
            "epochs": 1,
            "train_steps": 1,
        },
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
