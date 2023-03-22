from contextlib import nullcontext as no_error_raised

from ludwig.api import LudwigModel
from ludwig.constants import BINARY, TRAINER
from tests.integration_tests.utils import binary_feature, category_feature, generate_data


def generate_data_and_train(config, csv_filename):
    # Generate training data
    training_data_csv_path = generate_data(config["input_features"], config["output_features"], csv_filename)

    # Train Ludwig (Pythonic) model:
    ludwig_model = LudwigModel(config)

    with no_error_raised():
        ludwig_model.experiment(
            dataset=training_data_csv_path,
            skip_save_training_description=True,
            skip_save_training_statistics=True,
            skip_save_model=True,
            skip_save_progress=True,
            skip_save_log=True,
            skip_save_processed_input=True,
        )


def test_category_passthrough_encoder(csv_filename):
    input_features = [category_feature(), category_feature()]
    output_features = [category_feature(output_feature=True)]
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"train_steps": 1},
        "defaults": {"category": {"encoder": {"type": "passthrough"}}},
    }
    generate_data_and_train(config, csv_filename)


def test_binary_encoders(csv_filename):
    config = {
        "input_features": [
            {"name": "binary1", "type": BINARY, "encoder": {"type": "passthrough"}},
            {"name": "binary2", "type": BINARY, "encoder": {"type": "dense"}},
        ],
        "output_features": [binary_feature(output_feature=True)],
        TRAINER: {"train_steps": 1},
    }
    generate_data_and_train(config, csv_filename)
