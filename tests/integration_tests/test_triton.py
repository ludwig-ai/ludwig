# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
from typing import List

import pandas as pd
import pytest
import torch

from ludwig.api import LudwigModel
from ludwig.constants import BATCH_SIZE, TRAINER
from ludwig.data.dataset_synthesizer import build_synthetic_dataset_df
from ludwig.utils.data_utils import load_yaml
from ludwig.utils.inference_utils import to_inference_module_input_from_dataframe
from ludwig.utils.triton_utils import export_triton, get_inference_modules, POSTPROCESSOR, PREDICTOR, PREPROCESSOR
from tests.integration_tests.utils import (
    binary_feature,
    category_feature,
    generate_data,
    LocalTestBackend,
    number_feature,
    sequence_feature,
    set_feature,
    text_feature,
    vector_feature,
)


def test_triton_torchscript(csv_filename, tmpdir):
    # Configure features to be tested:
    input_features = [
        binary_feature(),
        number_feature(),
        category_feature(encoder={"vocab_size": 3}),
        # TODO: future support
        # sequence_feature(encoder={"vocab_size": 3}),
        # text_feature(encoder={"vocab_size": 3}),
        # vector_feature(),
        # timeseries_feature(),
        # date_feature(),
        # h3_feature(),
        # set_feature(encoder={"vocab_size": 3}),
        # bag_feature(encoder={"vocab_size": 3}),
        # image_feature(image_dest_folder),
        # audio_feature(audio_dest_folder),
    ]
    output_features = [
        binary_feature(),
        number_feature(),
        category_feature(decoder={"vocab_size": 3}),
        sequence_feature(decoder={"vocab_size": 3}),
        text_feature(decoder={"vocab_size": 3}),
        set_feature(decoder={"vocab_size": 3}),
        vector_feature(),
    ]
    backend = LocalTestBackend()
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 1, BATCH_SIZE: 128},
    }

    # Generate training data
    training_data_csv_path = generate_data(input_features, output_features, csv_filename)

    # Train Ludwig (Pythonic) model:
    ludwig_model = LudwigModel(config, backend=backend)
    ludwig_model.train(
        dataset=training_data_csv_path,
        skip_save_training_description=True,
        skip_save_training_statistics=True,
        skip_save_model=True,
        skip_save_progress=True,
        skip_save_log=True,
        skip_save_processed_input=True,
    )

    # Create graph inference model (Torchscript) from trained Ludwig model.
    triton_path = os.path.join(tmpdir, "triton")
    model_name = "test_triton"
    model_version = "1"
    df = pd.read_csv(training_data_csv_path)
    triton_artifacts = export_triton(
        model=ludwig_model, data_example=df, model_name=model_name, output_path=triton_path, model_version=model_version
    )

    # Validate that artifact paths exist.
    assert os.path.isdir(triton_path)
    assert all(os.path.exists(artifact.path) for artifact in triton_artifacts)

    # Load TorchScript models exported for Triton.
    triton_preprocessor = triton_predictor = triton_postprocessor = None
    for artifact in triton_artifacts:
        if artifact.model_name.endswith(PREPROCESSOR) and artifact.content_type == "application/octet-stream":
            triton_preprocessor = torch.jit.load(artifact.path)
        if artifact.model_name.endswith(PREDICTOR) and artifact.content_type == "application/octet-stream":
            triton_predictor = torch.jit.load(artifact.path)
        if artifact.model_name.endswith(POSTPROCESSOR) and artifact.content_type == "application/octet-stream":
            triton_postprocessor = torch.jit.load(artifact.path)

    assert triton_preprocessor is not None
    assert triton_predictor is not None
    assert triton_postprocessor is not None

    # Forward data through models.
    data_to_predict = to_inference_module_input_from_dataframe(df, ludwig_model.config, load_paths=True, device="cpu")
    triton_preprocessor_output = triton_preprocessor(*data_to_predict.values())
    triton_predictor_output = triton_predictor(*triton_preprocessor_output)
    triton_postprocessor_output = triton_postprocessor(*triton_predictor_output)

    # Get TorchScript inference modules and forward data.
    inference_modules = get_inference_modules(ludwig_model, "cpu")
    preprocessor_output = inference_modules[0](data_to_predict)
    predictor_output = inference_modules[1](preprocessor_output)
    postprocessor_output = inference_modules[2](predictor_output)

    assert len(postprocessor_output) == len(
        triton_postprocessor_output
    ), "Number of output mismatch after postprocessor step"

    for i, (_, out_value) in enumerate(postprocessor_output.items()):
        both_list = isinstance(out_value, list) and isinstance(triton_postprocessor_output[i], list)
        both_tensor = isinstance(out_value, torch.Tensor) and isinstance(triton_postprocessor_output[i], torch.Tensor)
        assert both_list or both_tensor, "Type mismatch in PREDICTIONS, PROBABILITIES, LOGITS output"

        if isinstance(out_value, list) and len(out_value) > 0 and isinstance(out_value[0], str):
            assert out_value == triton_postprocessor_output[i], "Category feature outputs failure."
        elif isinstance(out_value, list) and len(out_value) > 0 and isinstance(out_value[0], torch.Tensor):
            assert len(out_value) == len(triton_postprocessor_output[i]), "Set feature outputs failure."
            assert all(
                torch.allclose(inf, trit) for inf, trit in zip(out_value, triton_postprocessor_output[i])
            ), "Set feature outputs failure."
        elif isinstance(out_value, list) and len(out_value) > 0 and isinstance(out_value[0], list):
            assert len(out_value) == len(
                triton_postprocessor_output[i]
            ), "Sequence (including text, etc.) feature outputs failure."
            assert all(
                inf == trit for inf, trit in zip(out_value, triton_postprocessor_output[i])
            ), "Sequence (including text, etc.) feature outputs failure."
        elif isinstance(out_value, torch.Tensor):
            assert torch.allclose(out_value, triton_postprocessor_output[i])
        else:
            raise ValueError("Value should be either List[str] or torch.Tensor.")


def get_test_config_filenames() -> List[str]:
    """Return list of the config filenames used for Triton export."""
    configs_directory = "/".join(__file__.split("/")[:-1] + ["test_triton_configs"])
    return [os.path.join(configs_directory, config_fp) for config_fp in os.listdir(configs_directory)]


@pytest.mark.parametrize("config_path", get_test_config_filenames())
def test_triton_exportability(config_path, tmpdir):
    """Tests whether Triton export succeeds for a config."""
    config = load_yaml(config_path)
    dataset = build_synthetic_dataset_df(100, config)
    ludwig_model = LudwigModel(config)
    ludwig_model.train(
        dataset=dataset,
        skip_save_training_description=True,
        skip_save_training_statistics=True,
        skip_save_model=True,
        skip_save_progress=True,
        skip_save_log=True,
        skip_save_processed_input=True,
    )

    triton_path = os.path.join(tmpdir, "triton")
    model_name = "test_triton"
    model_version = "1"
    export_triton(
        model=ludwig_model,
        data_example=dataset.head(10),
        model_name=model_name,
        output_path=triton_path,
        model_version=model_version,
    )
