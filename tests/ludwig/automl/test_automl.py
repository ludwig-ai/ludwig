import os

import pandas as pd

from ludwig.automl.automl import _model_select
from ludwig.automl.base_config import _create_default_config, get_dataset_info
from ludwig.constants import (
    ACCURACY,
    ENCODER,
    GOAL,
    HYPEROPT,
    INPUT_FEATURES,
    MAXIMIZE,
    METRIC,
    NAME,
    OUTPUT_FEATURES,
    ROC_AUC,
)

from ludwig.utils.defaults import default_random_seed
from tests.integration_tests.utils import (
    binary_feature,
    category_feature,
    generate_data,
    image_feature,
    LocalTestBackend,
    text_feature
)


def generate_model_select_outputs(input_features, output_features, tmpdir):
    backend = LocalTestBackend()
    user_config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features
    }

    # Generate Dataset
    rel_path = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"))
    dataset = pd.read_csv(rel_path)

    # Get Dataset Info, Default Configs, and Metadata
    dataset_info = get_dataset_info(dataset)
    default_configs, features_metadata = _create_default_config(
        dataset_info, output_features[0][NAME], 10, default_random_seed, 0.9, backend
    )

    # Get _model_select() outputs
    model_config, model_category, row_count = _model_select(
        dataset_info,
        default_configs,
        features_metadata,
        user_config,
        True
    )

    return model_config, model_category, row_count


def test_model_select_defaults(tmpdir):
    image_dest_folder = os.path.join(tmpdir, "generated_images")

    # Generate Test Config
    input_features = [
        text_feature(preprocessing={"tokenizer": "space"}),
        image_feature(folder=image_dest_folder)
    ]
    del input_features[0][ENCODER]
    del input_features[1][ENCODER]

    output_features = [category_feature(output_feature=True)]

    model_config, model_category, row_count = generate_model_select_outputs(input_features, output_features, tmpdir)

    assert model_config[HYPEROPT][METRIC] == ACCURACY
    assert model_config[HYPEROPT][GOAL] == MAXIMIZE


def test_model_select_encoders_set(tmpdir):
    image_dest_folder = os.path.join(tmpdir, "generated_images")

    # Generate Test Config
    input_features = [
        text_feature(preprocessing={"tokenizer": "space"}),
        image_feature(folder=image_dest_folder)
    ]
    output_features = [binary_feature(output_feature=True)]

    model_config, model_category, row_count = generate_model_select_outputs(input_features, output_features, tmpdir)

    assert model_config[HYPEROPT][METRIC] == ROC_AUC
    assert model_config[HYPEROPT][GOAL] == MAXIMIZE

