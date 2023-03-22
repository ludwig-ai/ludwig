import logging
import tempfile
import time

from ludwig.api import LudwigModel
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import IMAGE, INPUT_FEATURES, OUTPUT_FEATURES, TEXT, TYPE
from ludwig.error import ConfigValidationError
from ludwig.schema.model_types.base import ModelConfig
from ludwig.types import ModelConfigDict
from tests.integration_tests.utils import generate_data_as_dataframe

logger = logging.getLogger(__name__)


def get_abrupt_config(config: ModelConfigDict):
    """Returns a config with a shortened trainer and a sample ratio of 1.0.

    Uses an embed encoder as a proxy for text encoders. Uses a stacked_cnn as a proxy for image encoders.
    """
    abrupt_model_config = ModelConfig.from_dict(config)

    abrupt_model_config.trainer.train_steps = 2
    abrupt_model_config.preprocessing.sample_ratio = 1.0

    abrupt_model_config = abrupt_model_config.to_dict()
    del abrupt_model_config["hyperopt"]
    del abrupt_model_config["backend"]

    # Replace text encoder with embed encoder.
    for input_feature in abrupt_model_config[INPUT_FEATURES]:
        if input_feature[TYPE] == TEXT:
            reduce_output_value = input_feature["encoder"]["reduce_output"]
            input_feature["encoder"] = {"type": "embed", "reduce_output": reduce_output_value}
            del input_feature["preprocessing"]
        if input_feature[TYPE] == IMAGE:
            input_feature["encoder"] = {"type": "stacked_cnn"}
            del input_feature["preprocessing"]

    return abrupt_model_config


@DeveloperAPI
def validate_config_with_synthetic_data(config: ModelConfigDict) -> None:
    """Validates a config by training a proxy model for 2 training steps using synthetic data and fast encoders.

    Synthetic datasets will have different training set metadata from the real dataset. For example, differences may
    arise in vocabulary, sequence length, image dimensions, etc, which may result in slightly different tensor shapes in
    the network overall.

    This function may not catch all the shape mismatch errors that could arise (though this will certainly be able to
    catch several).

    This function may also flag false positives, again, as synthetic data may not be representative of real data. For
    example, we've disabled validation for GBM models as synthetic data doesn't play well with GBM trainers.

    TODO: Expand to include other performance-related aspects of the model, i.e. estimated memory usage, # trainable
    parameters, etc.
    """
    start_time = time.time()
    if config.get("model_type", "ecd") != "ecd":
        # GBM validation using this method is too noisy.
        return

    # Shortened trainer.
    abrupt_model_config = get_abrupt_config(config)

    # Generate test data, and try training for 2 steps.
    with tempfile.TemporaryDirectory() as tmpdir:
        synthetic_df = generate_data_as_dataframe(
            abrupt_model_config[INPUT_FEATURES], abrupt_model_config[OUTPUT_FEATURES], num_examples=100
        )

        model = LudwigModel(abrupt_model_config)

        try:
            model.train(
                dataset=synthetic_df,
                skip_save_processed_input=True,
                skip_save_progress=True,
                skip_save_unprocessed_output=True,
                output_dir=tmpdir,
            )
        except Exception as e:
            end_time = time.time()
            logger.info(f"Synthetic config validation took: {end_time - start_time:4f}s.")
            raise ConfigValidationError(f"During synthetic config validation, got Exception: {e}")

    end_time = time.time()
    logger.info(f"Synthetic config validation took: {end_time - start_time:4f}s.")
