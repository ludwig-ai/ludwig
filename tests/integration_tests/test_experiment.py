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
import contextlib
import logging
import os
import shutil
import uuid
from collections import namedtuple

import pandas as pd
import pytest
import torch
import torchvision
import yaml

from ludwig.api import LudwigModel
from ludwig.backend import LOCAL_BACKEND
from ludwig.callbacks import Callback
from ludwig.constants import BATCH_SIZE, COLUMN, ENCODER, H3, NAME, PREPROCESSING, TRAINER, TYPE
from ludwig.data.concatenate_datasets import concatenate_df
from ludwig.data.dataset_synthesizer import build_synthetic_dataset_df
from ludwig.data.preprocessing import preprocess_for_training
from ludwig.encoders.registry import get_encoder_classes
from ludwig.error import ConfigValidationError
from ludwig.experiment import experiment_cli
from ludwig.predict import predict_cli
from ludwig.utils.data_utils import read_csv
from ludwig.utils.defaults import default_random_seed
from tests.integration_tests.utils import (
    audio_feature,
    bag_feature,
    binary_feature,
    category_distribution_feature,
    category_feature,
    create_data_set_to_use,
    date_feature,
    ENCODERS,
    generate_data,
    generate_output_features_with_dependencies,
    generate_output_features_with_dependencies_complex,
    h3_feature,
    image_feature,
    LocalTestBackend,
    number_feature,
    run_experiment,
    sequence_feature,
    set_feature,
    TEXT_ENCODERS,
    text_feature,
    timeseries_feature,
    vector_feature,
)

pytestmark = pytest.mark.integration_tests_d

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)


@pytest.mark.parametrize("encoder", TEXT_ENCODERS)
def test_experiment_text_feature_non_pretrained(encoder, csv_filename):
    input_features = [
        text_feature(encoder={"vocab_size": 30, "min_len": 1, "type": encoder}, preprocessing={"tokenizer": "space"})
    ]
    output_features = [category_feature(decoder={"vocab_size": 2})]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, dataset=rel_path)


def run_experiment_with_encoder(encoder, csv_filename):
    # Run in a subprocess to clear TF and prevent OOM
    # This also allows us to use GPU resources
    input_features = [text_feature(encoder={"vocab_size": 30, "min_len": 1, "type": encoder})]
    output_features = [category_feature(decoder={"vocab_size": 2})]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, dataset=rel_path)


@pytest.mark.parametrize("encoder", ENCODERS)
def test_experiment_seq_seq_generator(csv_filename, encoder):
    input_features = [text_feature(encoder={"type": encoder, "reduce_output": None})]
    output_features = [text_feature(decoder={"type": "generator"}, output_feature=True)]
    rel_path = generate_data(input_features, output_features, csv_filename)

    run_experiment(input_features, output_features, dataset=rel_path)


@pytest.mark.parametrize("encoder", ["embed", "rnn", "parallel_cnn", "stacked_parallel_cnn", "transformer"])
def test_experiment_seq_seq_tagger(csv_filename, encoder):
    input_features = [text_feature(encoder={"type": encoder, "reduce_output": None})]
    output_features = [text_feature(decoder={"type": "tagger"}, reduce_input=None)]
    rel_path = generate_data(input_features, output_features, csv_filename)

    run_experiment(input_features, output_features, dataset=rel_path)


@pytest.mark.parametrize("encoder", ["cnnrnn", "stacked_cnn"])
def test_experiment_seq_seq_tagger_fails_for_non_length_preserving_encoders(csv_filename, encoder):
    input_features = [text_feature(encoder={"type": encoder, "reduce_output": None})]
    output_features = [text_feature(decoder={"type": "tagger"}, reduce_input=None)]
    rel_path = generate_data(input_features, output_features, csv_filename)

    with pytest.raises(ValueError):
        run_experiment(input_features, output_features, dataset=rel_path)


def test_experiment_seq_seq_model_def_file(csv_filename, yaml_filename):
    # seq-to-seq test to use config file instead of dictionary
    input_features = [text_feature(encoder={"reduce_output": None, "type": "embed"})]
    output_features = [text_feature(decoder={"vocab_size": 3, "type": "tagger"}, reduce_input=None)]

    # Save the config to a yaml file
    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }
    with open(yaml_filename, "w") as yaml_out:
        yaml.safe_dump(config, yaml_out)

    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(None, None, dataset=rel_path, config=yaml_filename)


def test_experiment_seq_seq_train_test_valid(tmpdir):
    # seq-to-seq test to use train, test, validation files
    input_features = [text_feature(encoder={"reduce_output": None, "type": "rnn"})]
    output_features = [text_feature(decoder={"vocab_size": 3, "type": "tagger"}, reduce_input=None)]

    train_csv = generate_data(input_features, output_features, os.path.join(tmpdir, "train.csv"))
    test_csv = generate_data(input_features, output_features, os.path.join(tmpdir, "test.csv"), 20)
    valdation_csv = generate_data(input_features, output_features, os.path.join(tmpdir, "val.csv"), 20)

    run_experiment(
        input_features, output_features, training_set=train_csv, test_set=test_csv, validation_set=valdation_csv
    )

    # Save intermediate output
    run_experiment(
        input_features, output_features, training_set=train_csv, test_set=test_csv, validation_set=valdation_csv
    )


@pytest.mark.parametrize("encoder", ENCODERS)
def test_experiment_multi_input_intent_classification(csv_filename, encoder):
    # Multiple inputs, Single category output
    input_features = [
        text_feature(encoder={"vocab_size": 10, "min_len": 1, "representation": "sparse"}),
        category_feature(encoder={"vocab_size": 10}),
    ]
    output_features = [category_feature(decoder={"reduce_input": "sum", "vocab_size": 2})]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    input_features[0][ENCODER][TYPE] = encoder
    run_experiment(input_features, output_features, dataset=rel_path)


def test_experiment_with_torch_module_dict_feature_name(csv_filename):
    input_features = [category_feature(name="type")]
    output_features = [category_feature(name="to", output_feature=True)]
    rel_path = generate_data(input_features, output_features, csv_filename)

    run_experiment(input_features, output_features, dataset=rel_path)


def test_experiment_multiclass_with_class_weights(csv_filename):
    # Multiple inputs, Single category output
    input_features = [category_feature(encoder={"vocab_size": 10})]
    output_features = [category_feature(decoder={"vocab_size": 3}, loss={"class_weights": [0, 1, 2]})]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, dataset=rel_path)


def test_experiment_multilabel_with_class_weights(csv_filename):
    # Multiple inputs, Single category output
    input_features = [category_feature(encoder={"vocab_size": 10})]
    output_features = [set_feature(decoder={"vocab_size": 3}, loss={"class_weights": [0, 1, 2, 3]})]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, dataset=rel_path)


@pytest.mark.parametrize(
    "output_features",
    [
        # baseline test case
        [
            category_feature(decoder={"reduce_input": "sum", "vocab_size": 2}),
            sequence_feature(decoder={"vocab_size": 10, "max_len": 5}),
            number_feature(),
        ],
        # use generator as decoder
        [
            category_feature(decoder={"vocab_size": 2, "reduce_input": "sum"}),
            sequence_feature(decoder={"vocab_size": 10, "max_len": 5, "type": "generator"}),
            number_feature(),
        ],
        # Generator decoder and reduce_input = None
        [
            category_feature(decoder={"vocab_size": 2, "reduce_input": "sum"}),
            sequence_feature(decoder={"max_len": 5, "type": "generator"}, reduce_input=None),
            number_feature(normalization="minmax"),
        ],
        # output features with dependencies single dependency
        generate_output_features_with_dependencies("number_feature", ["category_feature"]),
        # output features with dependencies multiple dependencies
        generate_output_features_with_dependencies("number_feature", ["category_feature", "sequence_feature"]),
        # output features with dependencies multiple dependencies
        generate_output_features_with_dependencies("sequence_feature", ["category_feature", "number_feature"]),
        # output features with dependencies
        generate_output_features_with_dependencies("category_feature", ["sequence_feature"]),
        generate_output_features_with_dependencies_complex(),
    ],
)
def test_experiment_multiple_seq_seq(csv_filename, output_features):
    input_features = [
        text_feature(encoder={"vocab_size": 100, "min_len": 1, "type": "stacked_cnn"}),
        number_feature(normalization="zscore"),
        category_feature(encoder={"vocab_size": 10, "embedding_size": 5}),
        set_feature(),
        sequence_feature(encoder={"vocab_size": 10, "max_len": 10, "type": "embed"}),
    ]
    output_features = output_features

    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, dataset=rel_path)


@pytest.mark.parametrize("skip_save_processed_input", [True, False])
@pytest.mark.parametrize("in_memory", [True, False])
@pytest.mark.parametrize("image_source", ["file", "tensor"])
@pytest.mark.parametrize("num_channels", [1, 3])
def test_basic_image_feature(num_channels, image_source, in_memory, skip_save_processed_input, tmpdir):
    # Image Inputs
    image_dest_folder = os.path.join(tmpdir, "generated_images")

    input_features = [
        image_feature(
            folder=image_dest_folder,
            preprocessing={
                "in_memory": in_memory,
                "height": 12,
                "width": 12,
                "num_channels": num_channels,
                "num_processes": 5,
            },
            encoder={
                "type": "stacked_cnn",
                "output_size": 16,
                "num_filters": 8,
            },
        )
    ]
    output_features = [category_feature(decoder={"reduce_input": "sum", "vocab_size": 2})]

    rel_path = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"))

    if image_source == "file":
        # use images from file
        run_experiment(
            input_features, output_features, dataset=rel_path, skip_save_processed_input=skip_save_processed_input
        )
    else:
        # import image from file and store in dataframe as tensors.
        df = pd.read_csv(rel_path)
        image_feature_name = input_features[0]["name"]
        df[image_feature_name] = df[image_feature_name].apply(lambda x: torchvision.io.read_image(x))

        run_experiment(input_features, output_features, dataset=df, skip_save_processed_input=skip_save_processed_input)


def test_experiment_infer_image_metadata(tmpdir):
    # Image Inputs
    image_dest_folder = os.path.join(tmpdir, "generated_images")

    # Resnet encoder
    input_features = [
        image_feature(folder=image_dest_folder, encoder={"type": "stacked_cnn", "output_size": 16, "num_filters": 8}),
        text_feature(encoder={"type": "embed", "min_len": 1}),
        number_feature(normalization="zscore"),
    ]
    output_features = [category_feature(decoder={"reduce_input": "sum", "vocab_size": 2}), number_feature()]

    rel_path = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"))

    # remove image preprocessing section to force inferring image meta data
    input_features[0].pop("preprocessing")

    run_experiment(input_features, output_features, dataset=rel_path)


ImageParams = namedtuple("ImageTestParams", "image_encoder in_memory_flag skip_save_processed_input")


@pytest.mark.parametrize(
    "image_params",
    [
        ImageParams("stacked_cnn", True, True),
        ImageParams("stacked_cnn", False, False),
    ],
)
def test_experiment_image_inputs(image_params: ImageParams, tmpdir):
    # Image Inputs
    image_dest_folder = os.path.join(tmpdir, "generated_images")

    # Resnet encoder
    input_features = [
        image_feature(
            folder=image_dest_folder,
            preprocessing={"in_memory": True, "height": 12, "width": 12, "num_channels": 3, "num_processes": 5},
            encoder={"type": "resnet", "output_size": 16, "num_filters": 8},
        ),
        text_feature(encoder={"type": "embed", "min_len": 1}),
        number_feature(normalization="zscore"),
    ]
    output_features = [category_feature(decoder={"reduce_input": "sum", "vocab_size": 2}), number_feature()]

    input_features[0]["encoder"]["type"] = image_params.image_encoder
    input_features[0]["preprocessing"]["in_memory"] = image_params.in_memory_flag
    rel_path = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"))

    run_experiment(
        input_features,
        output_features,
        dataset=rel_path,
        skip_save_processed_input=image_params.skip_save_processed_input,
    )


# Primary focus of this test is to determine if exceptions are raised for different data set formats and in_memory
# setting.


@pytest.mark.parametrize("test_in_memory", [True, False])
@pytest.mark.parametrize("test_format", ["csv", "df", "hdf5"])
@pytest.mark.parametrize("train_in_memory", [True, False])
@pytest.mark.parametrize("train_format", ["csv", "df", "hdf5"])
def test_experiment_image_dataset(train_format, train_in_memory, test_format, test_in_memory, tmpdir):
    # Image Inputs
    image_dest_folder = os.path.join(tmpdir, "generated_images")

    input_features = [
        image_feature(
            folder=image_dest_folder,
            preprocessing={"in_memory": True, "height": 12, "width": 12, "num_channels": 3, "num_processes": 5},
            encoder={"type": "stacked_cnn", "output_size": 16, "num_filters": 8},
        ),
    ]
    output_features = [
        category_feature(decoder={"reduce_input": "sum", "vocab_size": 2}),
    ]

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        "preprocessing": {},
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }

    # create temporary name for train and test data sets
    train_csv_filename = os.path.join(tmpdir, "train_" + uuid.uuid4().hex[:10].upper() + ".csv")
    test_csv_filename = os.path.join(tmpdir, "test_" + uuid.uuid4().hex[:10].upper() + ".csv")

    # setup training data format to test
    train_data = generate_data(input_features, output_features, train_csv_filename)
    config["input_features"][0]["preprocessing"]["in_memory"] = train_in_memory
    training_set_metadata = None

    # define Ludwig model
    backend = LocalTestBackend()
    model = LudwigModel(
        config=config,
        backend=backend,
    )

    if train_format == "hdf5":
        # hdf5 format
        train_set, _, _, training_set_metadata = preprocess_for_training(
            model.config,
            dataset=train_data,
            backend=backend,
        )
        train_dataset_to_use = train_set.data_hdf5_fp
    else:
        train_dataset_to_use = create_data_set_to_use(train_format, train_data)

    model.train(dataset=train_dataset_to_use, training_set_metadata=training_set_metadata)

    model.config_obj.input_features.to_list()[0]["preprocessing"]["in_memory"] = test_in_memory

    # setup test data format to test
    test_data = generate_data(input_features, output_features, test_csv_filename)

    if test_format == "hdf5":
        # hdf5 format
        # create hdf5 data set
        _, test_set, _, training_set_metadata_for_test = preprocess_for_training(
            model.config,
            dataset=test_data,
            backend=backend,
        )
        test_dataset_to_use = test_set.data_hdf5_fp
    else:
        test_dataset_to_use = create_data_set_to_use(test_format, test_data)

    # run functions with the specified data format
    model.evaluate(dataset=test_dataset_to_use)
    model.predict(dataset=test_dataset_to_use)


DATA_FORMATS_TO_TEST = [
    "csv",
    "df",
    "dict",
    "excel",
    "feather",
    "fwf",
    "hdf5",
    "html",
    "json",
    "jsonl",
    "parquet",
    "pickle",
    "stata",
    "tsv",
]


@pytest.mark.parametrize("data_format", DATA_FORMATS_TO_TEST)
def test_experiment_dataset_formats(data_format, csv_filename):
    # primary focus of this test is to determine if exceptions are
    # raised for different data set formats and in_memory setting

    input_features = [number_feature(), category_feature()]
    output_features = [category_feature(output_feature=True), number_feature()]

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        "preprocessing": {},
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }

    # setup training data format to test
    raw_data = generate_data(input_features, output_features, csv_filename)

    training_set_metadata = None

    # define Ludwig model
    model = LudwigModel(config=config)

    if data_format == "hdf5":
        # hdf5 format
        training_set, _, _, training_set_metadata = preprocess_for_training(model.config, dataset=raw_data)
        dataset_to_use = training_set.data_hdf5_fp
    else:
        dataset_to_use = create_data_set_to_use(data_format, raw_data)

    model.train(dataset=dataset_to_use, training_set_metadata=training_set_metadata, random_seed=default_random_seed)

    # # run functions with the specified data format
    model.evaluate(dataset=dataset_to_use)
    model.predict(dataset=dataset_to_use)


def test_experiment_audio_inputs(tmpdir):
    # Audio Inputs
    audio_dest_folder = os.path.join(tmpdir, "generated_audio")

    input_features = [audio_feature(folder=audio_dest_folder)]
    output_features = [binary_feature()]

    rel_path = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"))

    run_experiment(input_features, output_features, dataset=rel_path)


def test_experiment_tied_weights(csv_filename):
    # Single sequence input, single category output
    input_features = [
        text_feature(name="text_feature1", encoder={"min_len": 1, "type": "cnnrnn", "reduce_output": "sum"}),
        text_feature(
            name="text_feature2", encoder={"min_len": 1, "type": "cnnrnn", "reduce_output": "sum"}, tied="text_feature1"
        ),
    ]
    output_features = [category_feature(decoder={"reduce_input": "sum", "vocab_size": 2})]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    for encoder in ENCODERS:
        input_features[0][ENCODER][TYPE] = encoder
        input_features[1][ENCODER][TYPE] = encoder
        run_experiment(input_features, output_features, dataset=rel_path)


def test_experiment_tied_weights_sequence_combiner(csv_filename):
    """Tests that tied weights work with sequence combiners if `sequence_length` is provided.

    Addresses https://github.com/ludwig-ai/ludwig/issues/3220
    """
    input_features = [
        text_feature(
            name="feature1",
            encoder={
                "max_len": 5,
                "reduce_output": None,
            },
            preprocessing={"sequence_length": 10},
        ),
        text_feature(
            name="feature2",
            encoder={
                "max_len": 3,
                "reduce_output": None,
            },
            preprocessing={"sequence_length": 10},
            tied="feature1",
        ),
    ]
    output_features = [category_feature(decoder={"reduce_input": "sum", "vocab_size": 2})]
    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "sequence"},
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(config=config, dataset=rel_path)


@pytest.mark.parametrize("enc_cell_type", ["lstm", "rnn", "gru"])
@pytest.mark.parametrize("attention", [False, True])
def test_sequence_tagger(enc_cell_type, attention, csv_filename):
    # Define input and output features
    input_features = [
        sequence_feature(encoder={"max_len": 10, "type": "rnn", "cell_type": enc_cell_type, "reduce_output": None})
    ]
    output_features = [
        sequence_feature(decoder={"max_len": 10, "type": "tagger", "attention": attention}, reduce_input=None)
    ]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    # run the experiment
    run_experiment(input_features, output_features, dataset=rel_path)


def test_sequence_tagger_text(csv_filename):
    # Define input and output features
    input_features = [text_feature(encoder={"max_len": 10, "type": "rnn", "reduce_output": None})]
    output_features = [
        sequence_feature(
            decoder={"max_len": 10, "type": "tagger"},
            reduce_input=None,
        )
    ]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    # run the experiment
    run_experiment(input_features, output_features, dataset=rel_path)


"""
@pytest.mark.distributed
def test_sequence_tagger_text_ray(csv_filename, ray_cluster_2cpu):
    # Define input and output features
    input_features = [text_feature(encoder={"max_len": 10, "type": "rnn", "reduce_output": None})]
    output_features = [
        sequence_feature(
            decoder={"max_len": 10, "type": "tagger"},
            reduce_input=None,
        )
    ]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    # run the experiment
    run_experiment(input_features, output_features, dataset=rel_path, backend="ray")
"""


def test_experiment_sequence_combiner_with_reduction_fails(csv_filename):
    config = {
        "input_features": [
            sequence_feature(
                name="seq1",
                encoder={
                    "min_len": 5,
                    "max_len": 5,
                    "type": "embed",
                    "cell_type": "lstm",
                    "reduce_output": "sum",
                },
            ),
            sequence_feature(
                name="seq2",
                encoder={
                    "min_len": 5,
                    "max_len": 5,
                    "type": "embed",
                    "cell_type": "lstm",
                    "reduce_output": "sum",
                },
            ),
            category_feature(encoder={"vocab_size": 5}),
        ],
        "output_features": [category_feature(decoder={"reduce_input": "sum", "vocab_size": 5})],
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
        "combiner": {
            "type": "sequence",
            "encoder": {"type": "rnn"},
            "main_sequence_feature": "seq1",
            "reduce_output": None,
        },
    }

    # Generate test data
    rel_path = generate_data(config["input_features"], config["output_features"], csv_filename)

    # Encoding sequence features with 'embed' should fail with SequenceConcatCombiner, since at least one sequence
    # feature should be rank 3.
    with pytest.raises(TypeError):
        run_experiment(config=config, dataset=rel_path)


@pytest.mark.parametrize("sequence_encoder", ENCODERS[1:])
def test_experiment_sequence_combiner(sequence_encoder, csv_filename):
    config = {
        "input_features": [
            sequence_feature(
                name="seq1",
                encoder={
                    "min_len": 5,
                    "max_len": 5,
                    "type": sequence_encoder,
                    "cell_type": "lstm",
                    "reduce_output": None,
                },
            ),
            sequence_feature(
                name="seq2",
                encoder={
                    "min_len": 5,
                    "max_len": 5,
                    "type": sequence_encoder,
                    "cell_type": "lstm",
                    "reduce_output": None,
                },
            ),
            category_feature(vocab_size=5),
        ],
        "output_features": [category_feature(decoder={"reduce_input": "sum", "vocab_size": 5})],
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
        "combiner": {
            "type": "sequence",
            "encoder": {"type": "rnn"},
            "main_sequence_feature": "seq1",
            "reduce_output": None,
        },
    }

    # Generate test data
    rel_path = generate_data(config["input_features"], config["output_features"], csv_filename)

    run_experiment(config=config, dataset=rel_path)


def test_experiment_model_resume(tmpdir):
    # Single sequence input, single category output
    # Tests saving a model file, loading it to rerun training and predict
    input_features = [sequence_feature(encoder={"type": "rnn", "reduce_output": "sum"})]
    output_features = [category_feature(decoder={"reduce_input": "sum", "vocab_size": 2})]
    # Generate test data
    rel_path = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"))

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }

    _, _, _, _, output_dir = experiment_cli(config, dataset=rel_path, output_directory=tmpdir)

    experiment_cli(config, dataset=rel_path, model_resume_path=output_dir)

    predict_cli(os.path.join(output_dir, "model"), dataset=rel_path)
    shutil.rmtree(output_dir, ignore_errors=True)


@pytest.mark.slow
@pytest.mark.parametrize(
    "dist_strategy",
    [
        pytest.param("ddp", id="ddp", marks=pytest.mark.distributed),
        pytest.param("horovod", id="horovod", marks=[pytest.mark.distributed, pytest.mark.horovod]),
    ],
)
def test_experiment_model_resume_distributed(tmpdir, dist_strategy, ray_cluster_4cpu):
    _run_experiment_model_resume_distributed(tmpdir, dist_strategy)


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="test requires at least 1 gpu")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires gpu support")
@pytest.mark.parametrize(
    "dist_strategy",
    [
        pytest.param("deepspeed", id="deepspeed", marks=pytest.mark.distributed),
    ],
)
def test_experiment_model_resume_distributed_gpu(tmpdir, dist_strategy, ray_cluster_4cpu):
    _run_experiment_model_resume_distributed(tmpdir, dist_strategy)


def _run_experiment_model_resume_distributed(tmpdir, dist_strategy):
    # Single sequence input, single category output
    # Tests saving a model file, loading it to rerun training and predict
    input_features = [number_feature()]
    output_features = [category_feature(output_feature=True)]
    # Generate test data
    rel_path = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"))

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 8},
        TRAINER: {"epochs": 1, BATCH_SIZE: 128},
        "backend": {"type": "ray", "trainer": {"strategy": dist_strategy, "num_workers": 2}},
    }

    _, _, _, _, output_dir = experiment_cli(config, dataset=rel_path, output_directory=os.path.join(tmpdir, "results1"))

    experiment_cli(
        config, dataset=rel_path, model_resume_path=output_dir, output_directory=os.path.join(tmpdir, "results2")
    )

    predict_cli(os.path.join(output_dir, "model"), dataset=rel_path, output_directory=os.path.join(tmpdir, "results3"))


@pytest.mark.parametrize(
    "missing_file",
    ["training_progress.json", "training_checkpoints"],
    ids=["training_progress", "training_checkpoints"],
)
def test_experiment_model_resume_missing_file(tmpdir, missing_file):
    # Single sequence input, single category output
    # Tests saving a model file, loading it to rerun training and predict
    input_features = [sequence_feature(encoder={"type": "rnn", "reduce_output": "sum"})]
    output_features = [category_feature(decoder={"reduce_input": "sum", "vocab_size": 2})]

    # Generate test data
    rel_path = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"))

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }

    _, _, _, _, output_dir = experiment_cli(config, dataset=rel_path, output_directory=tmpdir)

    try:
        # Remove file to simulate failure during first epoch of training which prevents
        # training_checkpoints to be empty and training_progress.json to not be created
        missing_file_path = os.path.join(output_dir, "model", missing_file)
        if missing_file == "training_progress.json":
            os.remove(missing_file_path)
        else:
            shutil.rmtree(missing_file_path)
    finally:
        # Training should start a fresh model training run without any errors
        experiment_cli(config, dataset=rel_path, model_resume_path=output_dir)

    predict_cli(os.path.join(output_dir, "model"), dataset=rel_path)
    shutil.rmtree(output_dir, ignore_errors=True)


@pytest.mark.slow
@pytest.mark.distributed
def test_experiment_model_resume_before_1st_epoch_distributed(tmpdir, ray_cluster_4cpu):
    # Single sequence input, single category output
    # Tests saving a model file, loading it to rerun training and predict
    input_features = [number_feature()]
    output_features = [category_feature(output_feature=True)]
    # Generate test data
    training_set = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"))

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 8},
        TRAINER: {"train_steps": 1, BATCH_SIZE: 128},
        "backend": {"type": "ray", "trainer": {"strategy": "ddp", "num_workers": 2}},
    }

    class InducedFailureCallback(Callback):
        """Class that defines the methods necessary to hook into process."""

        def on_resume_training(self, is_coordinator):
            if is_coordinator:
                raise RuntimeError("Induced failure")

    class NoFailureCallback(Callback):
        """Class that defines the methods necessary to hook into process."""

        def on_resume_training(self, is_coordinator):
            pass

    try:
        # Define Ludwig model object that drive model training
        model = LudwigModel(config=config, logging_level=logging.INFO, callbacks=[InducedFailureCallback()])
        model.train(
            dataset=training_set,
            experiment_name="simple_experiment",
            model_name="simple_model_incomplete",
            skip_save_processed_input=True,
            output_directory=os.path.join(tmpdir, "results1"),
        )
    except RuntimeError:
        model = LudwigModel(config=config, logging_level=logging.INFO, callbacks=[NoFailureCallback()])
        model.train(
            dataset=training_set,
            skip_save_processed_input=True,
            model_resume_path=os.path.join(tmpdir, "results1"),
        )


@pytest.mark.slow
@pytest.mark.distributed
def test_tabnet_with_batch_size_1(tmpdir, ray_cluster_4cpu):
    input_features = [number_feature()]
    output_features = [category_feature(output_feature=True)]
    training_set = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"))

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "tabnet"},
        TRAINER: {"train_steps": 1, BATCH_SIZE: 1},
        "backend": {"type": "ray", "trainer": {"strategy": "ddp", "num_workers": 2}},
    }
    model = LudwigModel(config=config, logging_level=logging.INFO)
    model.train(
        dataset=training_set,
        skip_save_training_description=True,
        skip_save_training_statistics=True,
        skip_save_model=True,
        skip_save_progress=True,
        skip_save_log=True,
        skip_save_processed_input=True,
    )


def test_experiment_various_feature_types(csv_filename):
    input_features = [binary_feature(), bag_feature()]
    output_features = [set_feature(decoder={"max_len": 3, "vocab_size": 5})]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, dataset=rel_path)


def test_experiment_timeseries(csv_filename):
    input_features = [timeseries_feature()]
    output_features = [binary_feature()]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0][ENCODER][TYPE] = "transformer"
    run_experiment(input_features, output_features, dataset=rel_path)


def test_visual_question_answering(tmpdir):
    image_dest_folder = os.path.join(tmpdir, "generated_images")
    input_features = [
        image_feature(
            folder=image_dest_folder,
            preprocessing={"in_memory": True, "height": 32, "width": 32, "num_channels": 3, "num_processes": 5},
            encoder={
                "type": "stacked_cnn",
            },
        ),
        text_feature(encoder={"type": "embed", "min_len": 1}),
    ]
    output_features = [sequence_feature(decoder={"type": "generator", "cell_type": "lstm"})]
    rel_path = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"))

    run_experiment(input_features, output_features, dataset=rel_path)


def test_image_resizing_num_channel_handling(tmpdir):
    """This test creates two image datasets with 3 channels and 1 channel. The combination of this data is used to
    train a model. This checks the cases where the user may or may not specify a number of channels in the config.

    :param csv_filename:
    :return:
    """
    # Image Inputs
    image_dest_folder = os.path.join(tmpdir, "generated_images")

    # Resnet encoder
    input_features = [
        image_feature(
            folder=image_dest_folder,
            preprocessing={"in_memory": True, "height": 32, "width": 32, "num_channels": 3, "num_processes": 5},
            encoder={
                "type": "stacked_cnn",
            },
        ),
        text_feature(encoder={"type": "embed", "min_len": 1}),
        number_feature(normalization="minmax"),
    ]
    output_features = [binary_feature(), number_feature()]
    rel_path = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset1.csv"), num_examples=50)

    df1 = read_csv(rel_path)

    input_features[0]["preprocessing"]["num_channels"] = 1
    rel_path = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset2.csv"), num_examples=50)
    df2 = read_csv(rel_path)

    df = concatenate_df(df1, df2, None, LOCAL_BACKEND)
    df.to_csv(rel_path, index=False)

    # Here the user specifies number of channels. Exception shouldn't be thrown
    run_experiment(input_features, output_features, dataset=rel_path)

    del input_features[0]["preprocessing"]["num_channels"]

    # User doesn't specify num channels, but num channels is inferred. Exception shouldn't be thrown
    run_experiment(input_features, output_features, dataset=rel_path)


@pytest.mark.parametrize("encoder", ["wave", "embed"])
def test_experiment_date(encoder, csv_filename):
    input_features = [date_feature()]
    output_features = [category_feature(decoder={"vocab_size": 2})]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    input_features[0][ENCODER] = {TYPE: encoder}
    run_experiment(input_features, output_features, dataset=rel_path)


@pytest.mark.parametrize("encoder", get_encoder_classes(H3).keys())
def test_experiment_h3(encoder, csv_filename):
    input_features = [h3_feature()]
    output_features = [binary_feature()]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    input_features[0][ENCODER] = {TYPE: encoder}
    run_experiment(input_features, output_features, dataset=rel_path)


def test_experiment_vector_feature(csv_filename):
    input_features = [vector_feature()]
    output_features = [binary_feature()]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    run_experiment(input_features, output_features, dataset=rel_path)


def test_experiment_vector_feature_infer_size(csv_filename):
    input_features = [vector_feature()]
    output_features = [vector_feature()]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    # Unset vector_size so it needs to be inferred
    del input_features[0][PREPROCESSING]
    del output_features[0][PREPROCESSING]

    run_experiment(input_features, output_features, dataset=rel_path)


@pytest.mark.parametrize("encoder", ["parallel_cnn", "dense", "passthrough"])
def test_forecasting_row_major(csv_filename, encoder):
    input_features = [timeseries_feature(encoder={"type": encoder})]
    output_features = [timeseries_feature(decoder={"type": "projector"})]

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14, "flatten_inputs": True},
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, config=config, dataset=rel_path)


def test_forecasting_column_major(csv_filename):
    input_feature = timeseries_feature(preprocessing={"window_size": 3})
    input_features = [input_feature]

    # Ensure output feature has the same column and the input feature
    output_feature = timeseries_feature(
        name=input_feature[COLUMN], preprocessing={"horizon": 2}, decoder={"type": "projector"}
    )
    output_feature[NAME] = f"{input_feature[NAME]}_out"
    output_features = [output_feature]

    # Generate test data in column-major format. This is just a dataframe of numbers with the same column name
    # as expected by the timeseries input feature
    column_major_feature = number_feature(name=input_feature[COLUMN])
    csv_filename = generate_data([column_major_feature], [], csv_filename)

    input_df = pd.read_csv(csv_filename)

    model, eval_stats, train_stats, preprocessed_data, output_directory = run_experiment(
        input_features, output_features, dataset=csv_filename
    )
    train_set, val_set, test_set, _ = preprocessed_data

    print(input_df)
    # print(train_set.to_df())

    horizon_df = model.forecast(input_df, horizon=5)
    print(horizon_df)


@pytest.mark.parametrize("reduce_output", [("sum"), (None)], ids=["sum", "none"])
def test_experiment_text_output_feature_with_tagger_decoder(csv_filename, reduce_output):
    """Test that the tagger decoder works with text output features when reduce_output is set to None."""
    input_features = [text_feature(encoder={"type": "parallel_cnn", "reduce_output": reduce_output})]
    output_features = [text_feature(output_feature=True, decoder={"type": "tagger"})]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    with pytest.raises(ConfigValidationError) if reduce_output == "sum" else contextlib.nullcontext():
        run_experiment(input_features, output_features, dataset=rel_path)


@pytest.mark.parametrize("reduce_output", [("sum"), (None)], ids=["sum", "none"])
def test_experiment_sequence_output_feature_with_tagger_decoder(csv_filename, reduce_output):
    """Test that the tagger decoder works with sequence output features when reduce_output is set to None."""
    input_features = [text_feature(encoder={"type": "parallel_cnn", "reduce_output": reduce_output})]
    output_features = [sequence_feature(output_feature=True, decoder={"type": "tagger"})]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    with pytest.raises(ConfigValidationError) if reduce_output == "sum" else contextlib.nullcontext():
        run_experiment(input_features, output_features, dataset=rel_path)


def test_experiment_category_input_feature_with_tagger_decoder(csv_filename):
    """Test that the tagger decoder doesn't work with category input features."""
    input_features = [category_feature()]
    output_features = [sequence_feature(output_feature=True, decoder={"type": "tagger"})]

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14, "reduce_output": None},
    }

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    with pytest.raises(ConfigValidationError):
        run_experiment(config=config, dataset=rel_path)


def test_experiment_category_distribution_feature(csv_filename):
    vocab = ["a", "b", "c"]
    input_features = [vector_feature()]
    output_features = [
        category_distribution_feature(
            preprocessing={
                "vocab": vocab,
            }
        )
    ]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    input_df = pd.read_csv(rel_path)

    # set batch_size=auto to ensure we produce the correct shaped synthetic data
    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {"epochs": 2, BATCH_SIZE: "auto"},
    }
    model, _, _, _, _ = run_experiment(input_features, output_features, dataset=rel_path, config=config)
    preds, _ = model.predict(input_df)

    # Check that predictions are category values drawn from the vocab, not distributions
    assert all(v in vocab for v in preds[f"{output_features[0][NAME]}_predictions"].values)


def test_experiment_ordinal_category(csv_filename):
    input_features = [category_feature(num_classes=5), number_feature()]
    output_features = [category_feature(output_feature=True, loss={"type": "corn"})]

    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, dataset=rel_path)


def test_experiment_feature_names_with_non_word_chars(tmpdir):
    config = yaml.safe_load(
        """
input_features:
    - name: Pclass (new)
      type: category
    - name: review.text
      type: category
    - name: other_feature
      type: category
      tied: review.text

output_features:
    - name: Survived (new)
      type: binary
    - name: Thrived
      type: binary
      dependencies:
        - Survived (new)

combiner:
    type: comparator
    entity_1:
        - Pclass (new)
        - other_feature
    entity_2:
        - review.text

"""
    )

    df = build_synthetic_dataset_df(120, config)
    model = LudwigModel(config, logging_level=logging.INFO)

    model.train(dataset=df, output_directory=tmpdir)


def test_text_output_feature_cols(tmpdir, csv_filename):
    """Test ensures that there are 4 output columns when model.predict() is called for text output features."""
    input_features = [text_feature(encoder={"type": "parallel_cnn"})]
    output_features = [text_feature(output_feature=True)]

    # Generate test data
    rel_path = generate_data(input_features, output_features, os.path.join(tmpdir, csv_filename))

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "trainer": {"train_steps": 2, "batch_size": 5},
    }

    model = LudwigModel(config, logging_level=logging.INFO)
    model.train(dataset=rel_path, output_directory=tmpdir)
    predict_output = model.predict(dataset=rel_path)[0]

    assert len(predict_output.columns) == 4

    predict_df_headers = {col_name.split("_")[2] for col_name in list(predict_output.columns)}
    assert predict_df_headers == {"predictions", "probability", "probabilities", "response"}
