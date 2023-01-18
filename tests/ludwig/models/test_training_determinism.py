import logging
import os

import numpy as np
import pytest

from ludwig.api import LudwigModel
from ludwig.constants import BATCH_SIZE, TRAINER
from ludwig.utils.numerical_test_utils import assert_all_finite
from tests.integration_tests.utils import (
    audio_feature,
    bag_feature,
    binary_feature,
    category_feature,
    date_feature,
    generate_data,
    h3_feature,
    image_feature,
    number_feature,
    sequence_feature,
    set_feature,
    text_feature,
    timeseries_feature,
    vector_feature,
)


@pytest.mark.distributed
@pytest.mark.skip(reason="https://github.com/ludwig-ai/ludwig/issues/2686")
def test_training_determinism_ray_backend(csv_filename, tmpdir, ray_cluster_4cpu):
    experiment_output_1, experiment_output_2 = train_twice("ray", csv_filename, tmpdir)

    eval_stats_1, train_stats_1, _, _ = experiment_output_1
    eval_stats_2, train_stats_2, _, _ = experiment_output_2

    assert_all_finite(eval_stats_1)
    assert_all_finite(eval_stats_2)
    assert_all_finite(train_stats_1)
    assert_all_finite(train_stats_2)

    np.testing.assert_equal(eval_stats_1, eval_stats_2)
    np.testing.assert_equal(train_stats_1, train_stats_2)


def test_training_determinism_local_backend(csv_filename, tmpdir):
    experiment_output_1, experiment_output_2 = train_twice("local", csv_filename, tmpdir)

    eval_stats_1, train_stats_1, _, _ = experiment_output_1
    eval_stats_2, train_stats_2, _, _ = experiment_output_2

    assert_all_finite(eval_stats_1)
    assert_all_finite(eval_stats_2)
    assert_all_finite(train_stats_1)
    assert_all_finite(train_stats_2)

    np.testing.assert_equal(eval_stats_1, eval_stats_2)
    np.testing.assert_equal(train_stats_1, train_stats_2)


def train_twice(backend, csv_filename, tmpdir):
    image_dest_folder = os.path.join(tmpdir, "generated_images")
    audio_dest_folder = os.path.join(tmpdir, "generated_audio")

    # Configure features to be tested:
    input_features = [
        binary_feature(),
        number_feature(),
        category_feature(encoder={"vocab_size": 10}),
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
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }

    # Generate training data
    training_data_csv_path = generate_data(input_features, output_features, csv_filename, num_examples=100)

    ludwig_model_1 = LudwigModel(config, logging_level=logging.ERROR, backend=backend)
    ludwig_model_2 = LudwigModel(config, logging_level=logging.ERROR, backend=backend)
    experiment_output_1 = ludwig_model_1.experiment(
        dataset=training_data_csv_path,
        skip_save_training_description=True,
        skip_save_training_statistics=True,
        skip_save_model=True,
        skip_save_progress=True,
        skip_save_log=True,
        skip_save_processed_input=True,
    )
    experiment_output_2 = ludwig_model_2.experiment(
        dataset=training_data_csv_path,
        skip_save_training_description=True,
        skip_save_training_statistics=True,
        skip_save_model=True,
        skip_save_progress=True,
        skip_save_log=True,
        skip_save_processed_input=True,
    )

    return experiment_output_1, experiment_output_2
