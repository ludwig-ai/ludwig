import os
import pandas as pd
import pytest
from ludwig.constants import NAME, INPUT_FEATURES, OUTPUT_FEATURES
from ludwig.utils.defaults import default_random_seed
from ludwig.automl.automl import _model_select, DatasetInfo
from ludwig.automl.base_config import get_dataset_info, _create_default_config
from tests.integration_tests.utils import category_feature, generate_data, image_feature, text_feature, LocalTestBackend


@pytest.mark.parametrize(
    "use_reference_config",
    [True, False]
)
def test_model_select(use_reference_config, tmpdir):
    # Test Backend
    backend = LocalTestBackend()

    # Temporary Image Directory
    image_dest_folder = os.path.join(tmpdir, "generated_images")

    # Generate Test Config
    input_features = [
        text_feature(preprocessing={"tokenizer": "space"}),
        image_feature(folder=image_dest_folder)
    ]
    output_features = [category_feature(output_feature=True)]
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

    # Test _mode_select() function
    model_config, model_category, row_count = _model_select(
        dataset_info,
        default_configs,
        features_metadata,
        user_config,
        use_reference_config
    )
