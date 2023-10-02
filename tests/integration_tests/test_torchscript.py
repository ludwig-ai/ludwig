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
from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
import pytest
import torch
import torchtext

from ludwig.api import LudwigModel
from ludwig.backend import RAY
from ludwig.constants import BATCH_SIZE, COMBINER, EVAL_BATCH_SIZE, LOGITS, NAME, PREDICTIONS, PROBABILITIES, TRAINER
from ludwig.data.preprocessing import preprocess_for_prediction
from ludwig.features.number_feature import numeric_transformation_registry
from ludwig.globals import TRAIN_SET_METADATA_FILE_NAME
from ludwig.models.inference import to_inference_module_input_from_dataframe
from ludwig.utils import output_feature_utils
from ludwig.utils.tokenizers import TORCHSCRIPT_COMPATIBLE_TOKENIZERS
from tests.integration_tests import utils
from tests.integration_tests.utils import (
    audio_feature,
    bag_feature,
    binary_feature,
    category_feature,
    date_feature,
    generate_data,
    h3_feature,
    image_feature,
    LocalTestBackend,
    number_feature,
    sequence_feature,
    set_feature,
    text_feature,
    timeseries_feature,
    vector_feature,
)


@pytest.mark.parametrize("should_load_model", [True, False])
@pytest.mark.parametrize("model_type", ["ecd", "gbm"])
def test_torchscript(tmpdir, csv_filename, should_load_model, model_type):
    #######
    # Setup
    #######
    dir_path = tmpdir
    data_csv_path = os.path.join(tmpdir, csv_filename)

    # Single sequence input, single category output
    input_features = [
        binary_feature(),
        number_feature(),
        category_feature(encoder={"type": "passthrough", "vocab_size": 3}),
        category_feature(encoder={"type": "onehot", "vocab_size": 3}),
    ]
    if model_type == "ecd":
        image_dest_folder = os.path.join(tmpdir, "generated_images")
        audio_dest_folder = os.path.join(tmpdir, "generated_audio")
        input_features.extend(
            [
                category_feature(encoder={"type": "dense", "vocab_size": 3}),
                sequence_feature(encoder={"vocab_size": 3}),
                text_feature(encoder={"vocab_size": 3}),
                vector_feature(),
                image_feature(image_dest_folder),
                audio_feature(audio_dest_folder),
                timeseries_feature(),
                date_feature(),
                date_feature(),
                h3_feature(),
                set_feature(encoder={"vocab_size": 3}),
                bag_feature(encoder={"vocab_size": 3}),
            ]
        )

    output_features = [
        category_feature(decoder={"vocab_size": 3}),
    ]
    if model_type == "ecd":
        output_features.extend(
            [
                binary_feature(),
                number_feature(),
                set_feature(decoder={"vocab_size": 3}),
                vector_feature(),
                sequence_feature(decoder={"vocab_size": 3}),
                text_feature(decoder={"vocab_size": 3}),
            ]
        )

    predictions_column_name = "{}_predictions".format(output_features[0]["name"])

    # Generate test data
    data_csv_path = generate_data(input_features, output_features, data_csv_path)

    #############
    # Train model
    #############
    backend = LocalTestBackend()
    config = {
        "model_type": model_type,
        "input_features": input_features,
        "output_features": output_features,
    }
    if model_type == "ecd":
        config[TRAINER] = {"epochs": 2}
    else:
        # Disable feature filtering to avoid having no features due to small test dataset,
        # see https://stackoverflow.com/a/66405983/5222402
        config[TRAINER] = {"num_boost_round": 2, "feature_pre_filter": False}
    ludwig_model = LudwigModel(config, backend=backend)
    ludwig_model.train(
        dataset=data_csv_path,
        skip_save_training_description=True,
        skip_save_training_statistics=True,
        skip_save_model=True,
        skip_save_progress=True,
        skip_save_log=True,
        skip_save_processed_input=True,
    )

    ###################
    # save Ludwig model
    ###################
    ludwigmodel_path = os.path.join(dir_path, "ludwigmodel")
    shutil.rmtree(ludwigmodel_path, ignore_errors=True)
    ludwig_model.save(ludwigmodel_path)

    ###################
    # load Ludwig model
    ###################
    if should_load_model:
        ludwig_model = LudwigModel.load(ludwigmodel_path, backend=backend)

    ##############################
    # collect weight tensors names
    ##############################
    original_predictions_df, _ = ludwig_model.predict(dataset=data_csv_path)
    original_weights = deepcopy(list(ludwig_model.model.parameters()))
    original_weights = [t.cpu() for t in original_weights]

    # Move the model to CPU for tracing
    ludwig_model.model.cpu()

    #################
    # save torchscript
    #################
    torchscript_path = os.path.join(dir_path, "torchscript")
    shutil.rmtree(torchscript_path, ignore_errors=True)
    ludwig_model.model.save_torchscript(torchscript_path)

    ###################################################
    # load Ludwig model, obtain predictions and weights
    ###################################################
    ludwig_model = LudwigModel.load(ludwigmodel_path, backend=backend)
    loaded_prediction_df, _ = ludwig_model.predict(dataset=data_csv_path)
    loaded_weights = deepcopy(list(ludwig_model.model.parameters()))
    loaded_weights = [t.cpu() for t in loaded_weights]

    #####################################################
    # restore torchscript, obtain predictions and weights
    #####################################################
    training_set_metadata_json_fp = os.path.join(ludwigmodel_path, TRAIN_SET_METADATA_FILE_NAME)

    dataset, training_set_metadata = preprocess_for_prediction(
        ludwig_model.config_obj.to_dict(),
        dataset=data_csv_path,
        training_set_metadata=training_set_metadata_json_fp,
        include_outputs=False,
        backend=backend,
    )

    restored_model = torch.jit.load(torchscript_path)

    # Check the outputs for one of the features for correctness
    # Here we choose the first output feature (categorical)
    of_name = list(ludwig_model.model.output_features.keys())[0]

    data_to_predict = {
        name: torch.from_numpy(dataset.dataset[feature.proc_column])
        for name, feature in ludwig_model.model.input_features.items()
    }

    # Get predictions from restored torchscript.
    logits = restored_model(data_to_predict)
    restored_predictions = torch.argmax(output_feature_utils.get_output_feature_tensor(logits, of_name, "logits"), -1)

    restored_predictions = [training_set_metadata[of_name]["idx2str"][idx] for idx in restored_predictions]

    restored_weights = deepcopy(list(restored_model.parameters()))
    restored_weights = [t.cpu() for t in restored_weights]

    ###############################################
    # Check if weights and predictions are the same
    ###############################################

    # Check to weight values match the original model.
    assert utils.is_all_close(original_weights, loaded_weights)
    assert utils.is_all_close(original_weights, restored_weights)

    # Check that predictions are identical to the original model.
    assert np.all(original_predictions_df[predictions_column_name] == loaded_prediction_df[predictions_column_name])

    assert np.all(original_predictions_df[predictions_column_name] == restored_predictions)


def test_torchscript_e2e_tabular(csv_filename, tmpdir):
    data_csv_path = os.path.join(tmpdir, csv_filename)
    # Configure features to be tested:
    bin_str_feature_input_feature = binary_feature()
    bin_str_feature_output_feature = binary_feature(output_feature=True)
    transformed_number_features = [
        number_feature(preprocessing={"normalization": numeric_transformer})
        for numeric_transformer in numeric_transformation_registry.keys()
    ]
    input_features = [
        bin_str_feature_input_feature,
        binary_feature(),
        *transformed_number_features,
        number_feature(preprocessing={"outlier_strategy": "fill_with_mean"}),
        category_feature(encoder={"vocab_size": 3}),
        bag_feature(encoder={"vocab_size": 3}),
        set_feature(encoder={"vocab_size": 3}),
        vector_feature(),
        # TODO: future support
        # date_feature(),
        # h3_feature(),
    ]
    output_features = [
        bin_str_feature_output_feature,
        binary_feature(output_feature=True),
        number_feature(),
        category_feature(decoder={"vocab_size": 3}),
        set_feature(decoder={"vocab_size": 3}),
        vector_feature(),
        sequence_feature(decoder={"vocab_size": 3}),
        text_feature(decoder={"vocab_size": 3}),
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
    df[bin_str_feature_input_feature[NAME]] = df[bin_str_feature_input_feature[NAME]].map(
        lambda x: true_value if x else false_value
    )
    df[bin_str_feature_output_feature[NAME]] = df[bin_str_feature_output_feature[NAME]].map(
        lambda x: true_value if x else false_value
    )
    df.to_csv(training_data_csv_path)

    validate_torchscript_outputs(tmpdir, config, backend, training_data_csv_path)


def test_torchscript_e2e_binary_only(csv_filename, tmpdir):
    data_csv_path = os.path.join(tmpdir, csv_filename)

    input_features = [
        binary_feature(),
    ]
    output_features = [
        binary_feature(),
    ]
    backend = LocalTestBackend()
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }

    # Generate training data
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)

    validate_torchscript_outputs(tmpdir, config, backend, training_data_csv_path)


def test_torchscript_e2e_tabnet_combiner(csv_filename, tmpdir):
    data_csv_path = os.path.join(tmpdir, csv_filename)
    # Configure features to be tested:
    input_features = [
        binary_feature(),
        number_feature(),
        category_feature(encoder={"vocab_size": 3}),
        bag_feature(encoder={"vocab_size": 3}),
        set_feature(encoder={"vocab_size": 3}),
    ]
    output_features = [
        binary_feature(),
        number_feature(),
        category_feature(decoder={"vocab_size": 3}),
    ]
    backend = LocalTestBackend()
    config = {
        "input_features": input_features,
        "output_features": output_features,
        COMBINER: {
            "type": "tabnet",
            "num_total_blocks": 2,
            "num_shared_blocks": 2,
        },
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }

    # Generate training data
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)

    validate_torchscript_outputs(tmpdir, config, backend, training_data_csv_path)


def test_torchscript_e2e_audio(csv_filename, tmpdir):
    data_csv_path = os.path.join(tmpdir, csv_filename)
    audio_dest_folder = os.path.join(tmpdir, "generated_audio")

    input_features = [
        audio_feature(audio_dest_folder),
    ]
    output_features = [
        binary_feature(),
    ]
    backend = LocalTestBackend()
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)

    # NOTE: audio preprocessing mismatches by very small margins ~O(1e-6) but causes flakiness in e2e test.
    # Increasing tolerance is a workaround to reduce flakiness for now.
    # TODO: remove this workaround when audio preprocessing is fixed.
    validate_torchscript_outputs(tmpdir, config, backend, training_data_csv_path, tolerance=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"encoder": {"type": "stacked_cnn"}},  # Ludwig custom encoder
        {"encoder": {"type": "alexnet", "use_pretrained": False}},  # TorchVision pretrained model encoder
    ],
)
def test_torchscript_e2e_image(tmpdir, csv_filename, kwargs):
    data_csv_path = os.path.join(tmpdir, csv_filename)
    image_dest_folder = os.path.join(tmpdir, "generated_images")
    input_features = [
        image_feature(image_dest_folder, **kwargs),
    ]
    output_features = [
        binary_feature(),
    ]
    backend = LocalTestBackend()
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)

    validate_torchscript_outputs(tmpdir, config, backend, training_data_csv_path)


def test_torchscript_e2e_text(tmpdir, csv_filename):
    data_csv_path = os.path.join(tmpdir, csv_filename)
    input_features = [
        text_feature(encoder={"vocab_size": 3}, preprocessing={"tokenizer": tokenizer})
        for tokenizer in TORCHSCRIPT_COMPATIBLE_TOKENIZERS
    ]
    output_features = [
        text_feature(decoder={"vocab_size": 3}),
    ]
    backend = LocalTestBackend()
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)

    validate_torchscript_outputs(tmpdir, config, backend, training_data_csv_path)


@pytest.mark.skipif(
    torch.torch_version.TorchVersion(torchtext.__version__) < (0, 14, 0),
    reason="requires torchtext 0.14.0 or higher",
)
def test_torchscript_e2e_text_hf_tokenizer(tmpdir, csv_filename):
    data_csv_path = os.path.join(tmpdir, csv_filename)
    input_features = [text_feature(encoder={"vocab_size": 3, "type": "bert"})]
    output_features = [
        text_feature(decoder={"vocab_size": 3}),
    ]
    backend = LocalTestBackend()
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128, EVAL_BATCH_SIZE: 128},
    }
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)

    validate_torchscript_outputs(tmpdir, config, backend, training_data_csv_path)


@pytest.mark.skipif(
    torch.torch_version.TorchVersion(torchtext.__version__) < (0, 14, 0),
    reason="requires torchtext 0.14.0 or higher",
)
def test_torchscript_e2e_text_hf_tokenizer_truncated_sequence(tmpdir, csv_filename):
    data_csv_path = os.path.join(tmpdir, csv_filename)
    input_features = [text_feature(encoder={"vocab_size": 3, "type": "bert"}, preprocessing={"max_sequence_length": 3})]
    output_features = [
        text_feature(decoder={"vocab_size": 3}),
    ]
    backend = LocalTestBackend()
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)

    validate_torchscript_outputs(tmpdir, config, backend, training_data_csv_path)


def test_torchscript_e2e_sequence(tmpdir, csv_filename):
    data_csv_path = os.path.join(tmpdir, csv_filename)
    input_features = [
        sequence_feature(encoder={"vocab_size": 3}, preprocessing={"tokenizer": "space"}),
    ]
    output_features = [
        sequence_feature(decoder={"vocab_size": 3}),
    ]
    backend = LocalTestBackend()
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)

    validate_torchscript_outputs(tmpdir, config, backend, training_data_csv_path)


def test_torchscript_e2e_timeseries(tmpdir, csv_filename):
    data_csv_path = os.path.join(tmpdir, csv_filename)
    input_features = [
        timeseries_feature(),
    ]
    output_features = [
        binary_feature(),
    ]
    backend = LocalTestBackend()
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)

    validate_torchscript_outputs(tmpdir, config, backend, training_data_csv_path)


def test_torchscript_e2e_h3(tmpdir, csv_filename):
    data_csv_path = os.path.join(tmpdir, csv_filename)
    input_features = [
        h3_feature(),
    ]
    output_features = [
        binary_feature(),
    ]
    backend = LocalTestBackend()
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)

    validate_torchscript_outputs(tmpdir, config, backend, training_data_csv_path)


def test_torchscript_e2e_date(tmpdir, csv_filename):
    data_csv_path = os.path.join(tmpdir, csv_filename)
    input_features = [
        date_feature(),
    ]
    output_features = [
        binary_feature(),
    ]
    backend = LocalTestBackend()
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)

    validate_torchscript_outputs(tmpdir, config, backend, training_data_csv_path)


@pytest.mark.parametrize("vector_type", [torch.Tensor, List[torch.Tensor]])
def test_torchscript_preproc_vector_alternative_type(tmpdir, csv_filename, vector_type):
    data_csv_path = os.path.join(tmpdir, csv_filename)
    feature = vector_feature()
    input_features = [
        feature,
    ]
    output_features = [
        binary_feature(),
    ]
    backend = LocalTestBackend()
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)

    # Initialize Ludwig model
    ludwig_model, script_module = initialize_torchscript_module(tmpdir, config, backend, training_data_csv_path)

    # Obtain preprocessed inputs from Python model
    preproc_inputs_expected, _ = preprocess_for_prediction(
        ludwig_model.config_obj.to_dict(),
        training_data_csv_path,
        ludwig_model.training_set_metadata,
        backend=backend,
        include_outputs=False,
    )

    df = pd.read_csv(training_data_csv_path)
    inputs = to_inference_module_input_from_dataframe(df, config, load_paths=True)

    def transform_vector_list(vector_list, vector_type):
        vectors = []
        for vector_str in vector_list:
            vectors.append(torch.tensor([float(x) for x in vector_str.split()]))

        if vector_type == torch.Tensor:
            vectors = torch.stack(vectors)
        return vectors

    inputs[feature[NAME]] = transform_vector_list(inputs[feature[NAME]], vector_type)

    preproc_inputs = script_module.preprocessor_forward(inputs)

    # Check that preproc_inputs is the same as preproc_inputs_expected.
    for feature_name_expected, feature_values_expected in preproc_inputs_expected.dataset.items():
        feature_name = feature_name_expected[: feature_name_expected.rfind("_")]  # remove proc suffix
        if feature_name not in preproc_inputs.keys():
            continue

        feature_values = preproc_inputs[feature_name]
        assert utils.is_all_close(feature_values, feature_values_expected), f"feature: {feature_name}"


@pytest.mark.parametrize("padding", ["left", "right"])
@pytest.mark.parametrize("fill_value", ["", "1.0"])
def test_torchscript_preproc_timeseries_alternative_type(tmpdir, csv_filename, padding, fill_value):
    data_csv_path = os.path.join(tmpdir, csv_filename)
    feature = timeseries_feature(
        preprocessing={
            "padding": padding,
            "timeseries_length_limit": 4,
            "fill_value": "1.0",
        },
        encoder={"max_len": 7},
    )
    input_features = [
        feature,
    ]
    output_features = [
        binary_feature(),
    ]
    backend = LocalTestBackend()
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path, nan_percent=0.2)

    # Initialize Ludwig model
    ludwig_model, script_module = initialize_torchscript_module(tmpdir, config, backend, training_data_csv_path)

    # Obtain preprocessed inputs from Python model
    preproc_inputs_expected, _ = preprocess_for_prediction(
        ludwig_model.config_obj.to_dict(),
        training_data_csv_path,
        ludwig_model.training_set_metadata,
        backend=backend,
        include_outputs=False,
    )

    df = pd.read_csv(training_data_csv_path)
    inputs = to_inference_module_input_from_dataframe(df, config, load_paths=True)

    def transform_timeseries_from_str_list_to_tensor_list(timeseries_list):
        timeseries = []
        for timeseries_str in timeseries_list:
            timeseries.append(torch.tensor([float(x) for x in timeseries_str.split()]))
        return timeseries

    inputs[feature[NAME]] = transform_timeseries_from_str_list_to_tensor_list(inputs[feature[NAME]])

    preproc_inputs = script_module.preprocessor_forward(inputs)

    # Check that preproc_inputs is the same as preproc_inputs_expected.
    for feature_name_expected, feature_values_expected in preproc_inputs_expected.dataset.items():
        feature_name = feature_name_expected[: feature_name_expected.rfind("_")]  # remove proc suffix
        assert feature_name in preproc_inputs.keys(), f'feature "{feature_name}" not found.'

        feature_values = preproc_inputs[feature_name]
        assert utils.is_all_close(feature_values, feature_values_expected), f'feature "{feature_name}" value mismatch.'


@pytest.mark.parametrize(
    "feature",
    [
        number_feature(),
        binary_feature(),
        category_feature(encoder={"vocab_size": 3}),
        bag_feature(encoder={"vocab_size": 3}),
        set_feature(encoder={"vocab_size": 3}),
        text_feature(encoder={"vocab_size": 3}),
        sequence_feature(encoder={"vocab_size": 3}),
        timeseries_feature(),
        h3_feature(),
        # TODO: future support
        # audio_feature(),  # default BFILL strategy is unintuitive at inference time
        # image_feature(),  # default BFILL strategy is unintuitive at inference time
        # vector_feature(), # does not have a missing_value_strategy
        # date_feature(),   # default fill with datetime.now() strategy is not scriptable
    ],
)
def test_torchscript_preproc_with_nans(tmpdir, csv_filename, feature):
    data_csv_path = os.path.join(tmpdir, csv_filename)
    input_features = [
        feature,
    ]
    output_features = [
        binary_feature(),
    ]
    backend = LocalTestBackend()
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path, nan_percent=0.2)

    # Initialize Ludwig model
    ludwig_model, script_module = initialize_torchscript_module(tmpdir, config, backend, training_data_csv_path)

    # Obtain preprocessed inputs from Python model
    preproc_inputs_expected, _ = preprocess_for_prediction(
        ludwig_model.config_obj.to_dict(),
        training_data_csv_path,
        ludwig_model.training_set_metadata,
        backend=backend,
        include_outputs=False,
    )

    df = pd.read_csv(training_data_csv_path)
    inputs = to_inference_module_input_from_dataframe(df, config, load_paths=True)
    preproc_inputs = script_module.preprocessor_forward(inputs)

    # Check that preproc_inputs is the same as preproc_inputs_expected.
    for feature_name_expected, feature_values_expected in preproc_inputs_expected.dataset.items():
        feature_name = feature_name_expected[: feature_name_expected.rfind("_")]  # remove proc suffix
        if feature_name not in preproc_inputs.keys():
            continue

        feature_values = preproc_inputs[feature_name]
        assert utils.is_all_close(feature_values, feature_values_expected), f"feature: {feature_name}"


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="test requires at least 1 gpu")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires gpu support")
@pytest.mark.distributed
@pytest.mark.parametrize(
    "feature_fn",
    [
        number_feature,
        image_feature,
        audio_feature,
        h3_feature,
        date_feature,
        # TODO: future support
        # binary_feature(),                # Torchscript takes List[str] as input, so currently CPU only
        # category_feature(encoder={"vocab_size": 3}),  # Torchscript takes List[str] as input, so currently CPU only
        # set_feature(encoder={"vocab_size": 3}),       # Torchscript takes List[str] as input, so currently CPU only
        # sequence_feature(encoder={"vocab_size": 3}),  # Torchscript takes List[str] as input, so currently CPU only
        # text_feature(encoder={"vocab_size": 3}),      # Torchscript takes List[str] as input, so currently CPU only
        # vector_feature(),                # Torchscript takes List[str] as input, so currently CPU only
        # bag_feature(encoder={"vocab_size": 3}),       # Torchscript takes List[str] as input, so currently CPU only
        # timeseries_feature(),            # Torchscript takes List[str] as input, so currently CPU only
    ],
)
def test_torchscript_preproc_gpu(tmpdir, csv_filename, feature_fn):
    data_csv_path = os.path.join(tmpdir, csv_filename)

    feature_kwargs = {}
    if feature_fn in {image_feature, audio_feature}:
        dest_folder = os.path.join(tmpdir, "generated_samples")
        feature_kwargs["folder"] = dest_folder

    input_features = [
        feature_fn(**feature_kwargs),
    ]
    output_features = [
        binary_feature(),
    ]

    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }
    backend = RAY
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)
    _, script_module = initialize_torchscript_module(
        tmpdir,
        config,
        backend,
        training_data_csv_path,
        device=torch.device("cuda"),
    )

    df = pd.read_csv(training_data_csv_path)
    inputs = to_inference_module_input_from_dataframe(
        df,
        config,
        load_paths=True,
        device=torch.device("cuda"),
    )
    preproc_inputs = script_module.preprocessor_forward(inputs)

    for name, values in preproc_inputs.items():
        assert values.is_cuda, f'feature "{name}" tensors are not on GPU'


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="test requires at least 1 gpu")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires gpu support")
@pytest.mark.distributed
@pytest.mark.parametrize(
    "feature_fn",
    [
        number_feature,
        category_feature,
        binary_feature,
        set_feature,
        vector_feature,
        sequence_feature,
        text_feature,
    ],
)
def test_torchscript_postproc_gpu(tmpdir, csv_filename, feature_fn):
    data_csv_path = os.path.join(tmpdir, csv_filename)

    feature_kwargs = {}
    if feature_fn in {category_feature, set_feature, sequence_feature, text_feature}:
        feature_kwargs["vocab_size"] = 3

    input_features = [
        number_feature(),
    ]
    output_features = [
        feature_fn(**feature_kwargs),
    ]

    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }
    backend = RAY
    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)
    _, script_module = initialize_torchscript_module(
        tmpdir,
        config,
        backend,
        training_data_csv_path,
        device=torch.device("cuda"),
    )

    df = pd.read_csv(training_data_csv_path)
    inputs = to_inference_module_input_from_dataframe(
        df,
        config,
        load_paths=True,
        device=torch.device("cuda"),
    )
    postproc_outputs = script_module(inputs)

    for feature_name, feature_outputs in postproc_outputs.items():
        for output_name, output_values in feature_outputs.items():
            assert utils.is_all_tensors_cuda(output_values), f"{feature_name}.{output_name} tensors are not on GPU"


def validate_torchscript_outputs(tmpdir, config, backend, training_data_csv_path, tolerance=1e-8):
    # Train Ludwig (Pythonic) model:
    ludwig_model, script_module = initialize_torchscript_module(
        tmpdir,
        config,
        backend,
        training_data_csv_path,
    )

    # Obtain predictions from Python model
    preds_dict, _ = ludwig_model.predict(dataset=training_data_csv_path, return_type=dict)

    df = pd.read_csv(training_data_csv_path)
    inputs = to_inference_module_input_from_dataframe(df, config, load_paths=True)
    outputs = script_module(inputs)

    # TODO: these are the only outputs we provide from Torchscript for now
    ts_outputs = {PREDICTIONS, PROBABILITIES, LOGITS}

    # Compare results from Python trained model against Torchscript
    for feature_name, feature_outputs_expected in preds_dict.items():
        assert feature_name in outputs

        feature_outputs = outputs[feature_name]
        for output_name, output_values_expected in feature_outputs_expected.items():
            if output_name not in ts_outputs:
                continue

            assert output_name in feature_outputs
            output_values = feature_outputs[output_name]
            assert utils.has_no_grad(output_values), f'"{feature_name}.{output_name}" tensors have gradients'
            assert utils.is_all_close(
                output_values, output_values_expected
            ), f'"{feature_name}.{output_name}" tensors are not close to ludwig model'


def initialize_torchscript_module(tmpdir, config, backend, training_data_csv_path, device=None):
    # Initialize Ludwig model
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

    # Put torchscript model on GPU if available (LudwigModel will run train/predict on GPU if available)
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Create graph inference model (Torchscript) from trained Ludwig model.
    script_module = ludwig_model.to_torchscript(device=device)
    # Ensure torchscript saving/loading does not affect final predictions.
    script_module_path = os.path.join(tmpdir, "inference_module.pt")
    torch.jit.save(script_module, script_module_path)
    script_module = torch.jit.load(script_module_path)
    return ludwig_model, script_module
