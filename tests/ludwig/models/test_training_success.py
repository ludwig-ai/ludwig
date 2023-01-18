from contextlib import nullcontext as no_error_raised

from ludwig.api import LudwigModel
from ludwig.constants import BATCH_SIZE, TRAINER
from tests.integration_tests.utils import category_feature, generate_data


def test_category_passthrough_encoder(csv_filename):
    input_features = [category_feature(), category_feature()]
    output_features = [category_feature(output_feature=True)]
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"train_steps": 1, BATCH_SIZE: 128},
        "defaults": {"category": {"encoder": {"type": "passthrough"}}},
    }

    # Generate training data
    training_data_csv_path = generate_data(input_features, output_features, csv_filename)

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
