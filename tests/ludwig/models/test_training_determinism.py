import pytest
import logging
import os
import json

from ludwig.utils.data_utils import NumpyEncoder
from ludwig.api import LudwigModel
from ludwig.constants import TRAINER
from tests.integration_tests.utils import (
    binary_feature,
    category_feature,
    generate_data,
    number_feature,
    sequence_feature,
    set_feature,
    text_feature,
    vector_feature,
    timeseries_feature,
    h3_feature,
    bag_feature,
    image_feature,
    audio_feature,
    date_feature
)


@pytest.mark.parametrize("backend", ["ray", "local"])
def test_training_determinism(csv_filename, backend, tmpdir):
    image_dest_folder = os.path.join(tmpdir, "generated_images")
    audio_dest_folder = os.path.join(tmpdir, "generated_audio")

    # Configure features to be tested:
    input_features = [
        binary_feature(),
        number_feature(),
        category_feature(encoder={"vocab_size": 10}),
        # TODO: future support
        sequence_feature(encoder={"vocab_size": 3}),
        text_feature(encoder={"vocab_size": 3}),
        vector_feature(),
        timeseries_feature(),
        date_feature(),
        h3_feature(),
        set_feature(encoder={"vocab_size": 3}),
        bag_feature(encoder={"vocab_size": 3}),
        image_feature(image_dest_folder),
        audio_feature(audio_dest_folder),
    ]
    output_features = [
        binary_feature(),
        number_feature(),
        category_feature(decoder={"vocab_size": 10}),
    ]
    config = {"input_features": input_features, "output_features": output_features, TRAINER: {"epochs": 2}}

    # Generate training data
    training_data_csv_path = generate_data(input_features, output_features, csv_filename)

    ludwig_model_1 = LudwigModel(config, logging_level=logging.ERROR, backend=backend)
    ludwig_model_2 = LudwigModel(config, logging_level=logging.ERROR, backend=backend)
    eval_stats_1, train_stats_1, preprocessed_data_1, _ = ludwig_model_1.experiment(
        dataset=training_data_csv_path,
        skip_save_training_description=True,
        skip_save_training_statistics=True,
        skip_save_model=True,
        skip_save_progress=True,
        skip_save_log=True,
        skip_save_processed_input=True,
    )
    eval_stats_2, train_stats_2, preprocessed_data_2, _ = ludwig_model_2.experiment(
        dataset=training_data_csv_path,
        skip_save_training_description=True,
        skip_save_training_statistics=True,
        skip_save_model=True,
        skip_save_progress=True,
        skip_save_log=True,
        skip_save_processed_input=True,
    )

    assert json.dumps(eval_stats_1, cls=NumpyEncoder, sort_keys=True, indent=4) == json.dumps(eval_stats_2,
                                                                                              cls=NumpyEncoder,
                                                                                              sort_keys=True, indent=4)
    assert json.dumps(train_stats_1, cls=NumpyEncoder, sort_keys=True, indent=4) == json.dumps(train_stats_2,
                                                                                               cls=NumpyEncoder,
                                                                                               sort_keys=True, indent=4)
