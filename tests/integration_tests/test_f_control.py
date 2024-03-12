import asyncio
import contextlib
import copy
import logging
import os
import platform
import random
import string
from typing import List, Union
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer

import ludwig
from ludwig.api import LudwigModel
from ludwig.backend import initialize_backend
from ludwig.callbacks import Callback
from ludwig.constants import (
    BASE_MODEL,
    BATCH_SIZE,
    COLUMN,
    DECODER,
    EPOCHS,
    FULL,
    INPUT_FEATURES,
    MODEL_ECD,
    MODEL_LLM,
    MODEL_TYPE,
    NAME,
    OUTPUT_FEATURES,
    PREDICTIONS,
    PREPROCESSING,
    PROC_COLUMN,
    PROMPT,
    SPLIT,
    TRAINER,
    TYPE,
)
from ludwig.data.concatenate_datasets import concatenate_df
from ludwig.data.preprocessing import handle_features_with_prompt_config, preprocess_for_prediction
from ludwig.schema.llms.prompt import PromptConfig
from ludwig.schema.model_types.base import ModelConfig
from ludwig.utils.carton_utils import export_carton
from tests.integration_tests.utils import (
    assert_preprocessed_dataset_shape_and_dtype_for_feature,
    audio_feature,
    binary_feature,
    category_feature,
    generate_data,
    generate_data_as_dataframe,
    image_feature,
    LocalTestBackend,
    number_feature,
    sequence_feature,
    text_feature,
)

NUM_EXAMPLES = 20

# TODO: <Alex>ALEX</Alex>
# pytestmark = pytest.mark.integration_tests_x
# TODO: <Alex>ALEX</Alex>


# TODO: <Alex>ALEX</Alex>
@pytest.mark.integration_tests_x
# TODO: <Alex>ALEX</Alex>
@pytest.mark.skipif(platform.system() == "Windows", reason="Carton is not supported on Windows")
def test_carton_torchscript(csv_filename, tmpdir):
    data_csv_path = os.path.join(tmpdir, csv_filename)

    # Configure features to be tested:
    bin_str_feature = binary_feature()
    input_features = [
        bin_str_feature,
        # binary_feature(),
        number_feature(),
        category_feature(encoder={"vocab_size": 3}),
        # TODO: future support
        # sequence_feature(vocab_size=3),
        # text_feature(vocab_size=3),
        # vector_feature(),
        # image_feature(image_dest_folder),
        # audio_feature(audio_dest_folder),
        # timeseries_feature(),
        # date_feature(),
        # h3_feature(),
        # set_feature(vocab_size=3),
        # bag_feature(vocab_size=3),
    ]
    output_features = [
        bin_str_feature,
        # binary_feature(),
        number_feature(),
        category_feature(decoder={"vocab_size": 3}, output_feature=True),
        # TODO: future support
        # sequence_feature(vocab_size=3),
        # text_feature(vocab_size=3),
        # set_feature(vocab_size=3),
        # vector_feature()
    ]
    backend = LocalTestBackend()
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }

    # Generate training data
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)

    # Convert bool values to strings, e.g., {'Yes', 'No'}
    df = pd.read_csv(training_data_csv_path)
    false_value, true_value = "No", "Yes"
    df[bin_str_feature[NAME]] = df[bin_str_feature[NAME]].map(lambda x: true_value if x else false_value)
    df.to_csv(training_data_csv_path)

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

    # Obtain predictions from Python model
    preds_dict, _ = ludwig_model.predict(dataset=training_data_csv_path, return_type=dict)

    # Create graph inference model (Torchscript) from trained Ludwig model.
    carton_path = os.path.join(tmpdir, "carton")
    export_carton(ludwig_model, carton_path)

    import cartonml as carton

    # Load the carton model
    # See https://pyo3.rs/v0.20.0/ecosystem/async-await#a-note-about-asynciorun for why we wrap it
    # in another function
    async def load():
        return await carton.load(carton_path)

    loop = asyncio.get_event_loop()
    carton_model = loop.run_until_complete(load())

    def to_input(s: pd.Series) -> Union[List[str], torch.Tensor]:
        if s.dtype == "object":
            return np.array(s.to_list())
        return s.to_numpy().astype(np.float32)

    df = pd.read_csv(training_data_csv_path)
    inputs = {name: to_input(df[feature.column]) for name, feature in ludwig_model.model.input_features.items()}

    # See https://pyo3.rs/v0.20.0/ecosystem/async-await#a-note-about-asynciorun for why we wrap it
    # in another function
    async def infer(inputs):
        return await carton_model.infer(inputs)

    outputs = loop.run_until_complete(infer(inputs))

    # Compare results from Python trained model against Carton
    assert len(preds_dict) == len(outputs)
    for feature_name, feature_outputs_expected in preds_dict.items():
        assert feature_name in outputs

        output_values_expected = feature_outputs_expected[PREDICTIONS]
        output_values = outputs[feature_name]
        if output_values.dtype.type in {np.string_, np.str_}:
            # Strings should match exactly
            assert np.all(output_values == output_values_expected), f"feature: {feature_name}, output: predictions"
        else:
            assert np.allclose(output_values, output_values_expected), f"feature: {feature_name}, output: predictions"


# TODO: <Alex>ALEX</Alex>
# TODO: <Alex>ALEX</Alex>
@pytest.mark.integration_tests_x
# TODO: <Alex>ALEX</Alex>
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["false", "true"])
def test_vit_encoder_different_dimension_image(tmpdir, csv_filename, use_pretrained: bool):
    input_features = [
        image_feature(
            os.path.join(tmpdir, "generated_output"),
            preprocessing={"in_memory": True, "height": 224, "width": 206, "num_channels": 3},
            encoder={TYPE: "_vit_legacy", "use_pretrained": use_pretrained},
        )
    ]
    output_features = [category_feature(decoder={"vocab_size": 5}, reduce_input="sum")]

    data_csv = generate_data(
        input_features, output_features, os.path.join(tmpdir, csv_filename), num_examples=NUM_EXAMPLES
    )

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        TRAINER: {"train_steps": 1},
    }

    model = LudwigModel(config)

    # Failure happens post preprocessing but before training during the ECD model creation phase
    # so make sure the model can be created properly and training can proceed
    model.train(dataset=data_csv)


# TODO: <Alex>ALEX</Alex>
