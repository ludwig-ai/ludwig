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
import shutil
import sys

import pandas as pd
import torch

from ludwig.api import LudwigModel
from ludwig.constants import TRAINER
from ludwig.utils.inference_utils import to_inference_module_input_from_dataframe
from ludwig.utils.triton_utils import ENSEMBLE, export_triton, get_inference_modules, INFERENCE_STAGES
from tests.integration_tests.utils import (
    bag_feature,
    binary_feature,
    category_feature,
    date_feature,
    generate_data,
    h3_feature,
    LocalTestBackend,
    number_feature,
    sequence_feature,
    set_feature,
    text_feature,
    timeseries_feature,
    vector_feature,
)


def dont_test_triton_torchscript(csv_filename, tmpdir):
    # data_csv_path = os.path.join(tmpdir, csv_filename)
    # Configure features to be tested:
    input_features = [
        binary_feature(),
        number_feature(),
        category_feature(vocab_size=3),
        sequence_feature(vocab_size=3),
        text_feature(vocab_size=3),
        vector_feature(),
        timeseries_feature(),
        date_feature(),
        h3_feature(),
        set_feature(vocab_size=3),
        bag_feature(vocab_size=3),
        # TODO: future support
        # image_feature(image_dest_folder),
        # audio_feature(audio_dest_folder),
    ]
    output_features = [
        binary_feature(),
        number_feature(),
        category_feature(vocab_size=3),
        sequence_feature(vocab_size=3),
        text_feature(vocab_size=3),
        set_feature(vocab_size=3),
        vector_feature(),
    ]
    backend = LocalTestBackend()
    config = {"input_features": input_features, "output_features": output_features, TRAINER: {"epochs": 1}}

    # Generate training data
    training_data_csv_path = generate_data(input_features, output_features, csv_filename)

    df = pd.read_csv(training_data_csv_path)
    # df.to_csv(training_data_csv_path)

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

    os.remove(training_data_csv_path)

    # Create graph inference model (Torchscript) from trained Ludwig model.
    triton_path = os.path.join(tmpdir, "triton")
    model_name = "test_triton"
    model_version = 1
    paths = export_triton(
        model=ludwig_model, data_example=df, model_name=model_name, output_path=triton_path, model_version=model_version
    )

    # Validate number of models and that paths exist
    assert all(inference_stage in paths for inference_stage in INFERENCE_STAGES)
    assert ENSEMBLE in paths
    assert all(len(value) == 3 for value in paths.values())
    assert os.path.isdir(triton_path)
    assert all(os.path.exists(value[0]) for value in paths.values())
    assert all(os.path.exists(value[1]) for value in paths.values())
    assert all(isinstance(value[2], int) for value in paths.values())

    # Load TorchScript models exported for Triton.
    triton_preprocessor = torch.jit.load(paths[INFERENCE_STAGES[0]][1])
    triton_predictor = torch.jit.load(paths[INFERENCE_STAGES[1]][1])
    triton_postprocessor = torch.jit.load(paths[INFERENCE_STAGES[2]][1])

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
