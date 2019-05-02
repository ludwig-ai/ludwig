# -*- coding: utf-8 -*-
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
import logging
import os
import shutil

import pytest
import yaml

from ludwig.data.concatenate_datasets import concatenate_df
from ludwig.experiment import experiment
from ludwig.predict import full_predict
from ludwig.utils.data_utils import read_csv
from tests.integration_tests.utils import ENCODERS
from tests.integration_tests.utils import bag_feature
from tests.integration_tests.utils import binary_feature
from tests.integration_tests.utils import categorical_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import image_feature
from tests.integration_tests.utils import numerical_feature
from tests.integration_tests.utils import sequence_feature
from tests.integration_tests.utils import set_feature
from tests.integration_tests.utils import text_feature
from tests.integration_tests.utils import timeseries_feature

# The following imports are pytest fixtures, required for running the tests
from tests.fixtures.filenames import csv_filename
from tests.fixtures.filenames import yaml_filename


def run_experiment(input_features, output_features, **kwargs):
    """
    Helper method to avoid code repetition in running an experiment. Deletes
    the data saved to disk after running the experiment
    :param input_features: list of input feature dictionaries
    :param output_features: list of output feature dictionaries
    **kwargs you may also pass extra parameters to the experiment as keyword
    arguments
    :return: None
    """
    model_definition = None
    if input_features is not None and output_features is not None:
        # This if is necessary so that the caller can call with
        # model_definition_file (and not model_definition)
        model_definition = {
            'input_features': input_features,
            'output_features': output_features,
            'combiner': {
                'type': 'concat',
                'fc_size': 14
            },
            'training': {'epochs': 2}
        }

    args = {
        'model_definition': model_definition,
        'skip_save_processed_input': True,
        'skip_save_progress': True,
        'skip_save_unprocessed_output': True,
    }
    args.update(kwargs)

    exp_dir_name = experiment(**args)
    shutil.rmtree(exp_dir_name, ignore_errors=True)


def test_experiment_seq_seq(csv_filename):
    # Single Sequence input, single sequence output
    # Only the following encoders are working
    input_features = [text_feature(reduce_output=None, encoder='rnn')]
    output_features = [text_feature(reduce_input=None, decoder='tagger')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    encoders2 = ['cnnrnn', 'stacked_cnn']
    for encoder in encoders2:
        logging.info('seq to seq test, Encoder: {0}'.format(encoder))
        input_features[0]['encoder'] = encoder
        run_experiment(input_features, output_features, data_csv=rel_path)


def test_experiment_seq_seq_model_def_file(csv_filename, yaml_filename):
    # seq-to-seq test to use model definition file instead of dictionary
    input_features = [text_feature(reduce_output=None, encoder='embed')]
    output_features = [
        text_feature(reduce_input=None, vocab_size=3, decoder='tagger')
    ]

    # Save the model definition to a yaml file
    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat', 'fc_size': 14},
        'training': {'epochs': 2}
    }
    with open(yaml_filename, 'w') as yaml_out:
        yaml.safe_dump(model_definition, yaml_out)

    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(
        None, None, data_csv=rel_path, model_definition_file=yaml_filename
    )


def test_experiment_seq_seq_train_test_valid(csv_filename):
    # seq-to-seq test to use train, test, validation files
    input_features = [text_feature(reduce_output=None, encoder='rnn')]
    output_features = [
        text_feature(reduce_input=None, vocab_size=3, decoder='tagger')
    ]

    train_csv = generate_data(
        input_features, output_features, 'tr_' + csv_filename
    )
    test_csv = generate_data(
        input_features, output_features, 'test_' + csv_filename, 20
    )
    valdation_csv = generate_data(
        input_features, output_features, 'val_' + csv_filename, 20
    )

    run_experiment(
        input_features,
        output_features,
        data_train_csv=train_csv,
        data_test_csv=test_csv,
        data_validation_csv=valdation_csv
    )

    input_features[0]['encoder'] = 'parallel_cnn'
    # Save intermediate output
    run_experiment(
        input_features,
        output_features,
        data_train_csv=train_csv,
        data_test_csv=test_csv,
        data_validation_csv=valdation_csv
    )

    # Delete the temporary data created
    # This test is saving the processed data to hdf5
    for prefix in ['tr_', 'test_', 'val_']:
        if os.path.isfile(prefix + csv_filename):
            os.remove(prefix + csv_filename)


def test_experiment_multi_input_intent_classification(csv_filename):
    # Multiple inputs, Single category output
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        categorical_feature(
            vocab_size=10,
            loss='sampled_softmax_cross_entropy'
        )
    ]
    output_features = [categorical_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    for encoder in ENCODERS:
        input_features[0]['encoder'] = encoder
        run_experiment(input_features, output_features, data_csv=rel_path)


def test_experiment_multiple_seq_seq(csv_filename):
    # Multiple inputs, Multiple outputs
    input_features = [
        text_feature(vocab_size=100, min_len=1, encoder='stacked_cnn'),
        numerical_feature(),
        categorical_feature(vocab_size=10, embedding_size=5),
        set_feature(),
        sequence_feature(vocab_size=10, max_len=10, encoder='embed')
    ]
    output_features = [
        categorical_feature(vocab_size=2, reduce_input='sum'),
        sequence_feature(vocab_size=10, max_len=5),
        numerical_feature()
    ]

    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, data_csv=rel_path)

    # Use generator as decoder
    output_features = [
        categorical_feature(vocab_size=2, reduce_input='sum'),
        sequence_feature(vocab_size=10, max_len=5, decoder='generator'),
        numerical_feature()
    ]

    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, data_csv=rel_path)

    # Generator decoder and reduce_input = None
    output_features = [
        categorical_feature(vocab_size=2, reduce_input='sum'),
        sequence_feature(max_len=5, decoder='generator', reduce_input=None),
        numerical_feature()
    ]
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, data_csv=rel_path)


def test_experiment_image_inputs(csv_filename):
    # Image Inputs
    image_dest_folder = os.path.join(os.getcwd(), 'generated_images')

    # Resnet encoder
    input_features = [
        image_feature(
            folder=image_dest_folder,
            encoder='resnet',
            preprocessing={
                'in_memory': True,
                'height': 8,
                'width': 8,
                'num_channels': 3
            },
            fc_size=16,
            num_filters=8
        ),
        text_feature(encoder='embed', min_len=1),
        numerical_feature()
    ]
    output_features = [
        categorical_feature(vocab_size=2, reduce_input='sum'),
        numerical_feature()
    ]

    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, data_csv=rel_path)

    # Stacked CNN encoder
    input_features[0]['encoder'] = 'stacked_cnn'
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, data_csv=rel_path)

    # Stacked CNN encoder, in_memory = False
    input_features[0]['preprocessing']['in_memory'] = False
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, data_csv=rel_path)

    # Delete the temporary data created
    shutil.rmtree(image_dest_folder)


def test_experiment_tied_weights(csv_filename):
    # Single sequence input, single category output
    input_features = [
        text_feature(
            name='text_feature1',
            min_len=1,
            encoder='cnnrnn',
            reduce_output='sum'
        ),
        text_feature(
            name='text_feature2',
            min_len=1,
            encoder='cnnrnn',
            reduce_output='sum',
            tied_weights='text_feature1'
        )
    ]
    output_features = [categorical_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    for encoder in ENCODERS:
        input_features[0]['encoder'] = encoder
        input_features[1]['encoder'] = encoder
        run_experiment(input_features, output_features, data_csv=rel_path)


def test_experiment_attention(csv_filename):
    # Machine translation with attention
    input_features = [
            sequence_feature(encoder='rnn', cell_type='lstm', max_len=10)
        ]
    output_features = [
        sequence_feature(
            max_len=10,
            cell_type='lstm',
            decoder='generator',
            attention='bahdanau'
        )
    ]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    for attention in ['bahdanau', 'luong']:
        output_features[0]['attention'] = attention
        run_experiment(input_features, output_features, data_csv=rel_path)


def test_experiment_sequence_combiner(csv_filename):
    # Sequence combiner
    input_features = [
        sequence_feature(
            name='english',
            min_len=5,
            max_len=5,
            encoder='rnn',
            cell_type='lstm',
            reduce_output=None
        ),
        sequence_feature(
            name='spanish',
            min_len=5,
            max_len=5,
            encoder='rnn',
            cell_type='lstm',
            reduce_output=None
        ),
        categorical_feature(vocab_size=5)
    ]
    output_features = [
        categorical_feature(reduce_input='sum', vocab_size=5)
    ]

    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'training': {
            'epochs': 2
        },
        'combiner': {
            'type': 'sequence_concat',
            'encoder': 'rnn',
            'main_sequence_feature': 'random_sequence'
        }
    }

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    for encoder in ENCODERS[:-2]:
        logging.error('sequence combiner. encoders: {0}, {1}'.format(
            encoder,
            encoder
        ))
        input_features[0]['encoder'] = encoder
        input_features[1]['encoder'] = encoder

        model_definition['input_features'] = input_features

        exp_dir_name = experiment(
            model_definition,
            skip_save_processed_input=False,
            skip_save_progress=True,
            skip_save_unprocessed_output=True,
            data_csv=rel_path
        )
        shutil.rmtree(exp_dir_name, ignore_errors=True)


def test_experiment_model_resume(csv_filename):
    # Single sequence input, single category output
    # Tests saving a model file, loading it to rerun training and predict
    input_features = [sequence_feature(encoder='rnn', reduce_output='sum')]
    output_features = [categorical_feature(vocab_size=2, reduce_input='sum')]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat', 'fc_size': 14},
        'training': {'epochs': 2}
    }

    exp_dir_name = experiment(model_definition, data_csv=rel_path)
    logging.info('Experiment Directory: {0}'.format(exp_dir_name))

    experiment(
        model_definition,
        data_csv=rel_path,
        model_resume_path=exp_dir_name
    )

    full_predict(os.path.join(exp_dir_name, 'model'), data_csv=rel_path)
    shutil.rmtree(exp_dir_name, ignore_errors=True)


def test_experiment_various_feature_types(csv_filename):
    input_features = [binary_feature(), bag_feature()]
    output_features = [set_feature(max_len=3, vocab_size=5)]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, data_csv=rel_path)


def test_experiment_timeseries(csv_filename):
    input_features = [timeseries_feature()]
    output_features = [binary_feature()]

    encoders2 = [
        'rnn', 'cnnrnn', 'stacked_cnn', 'parallel_cnn', 'stacked_parallel_cnn'
    ]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    for encoder in encoders2:
        input_features[0]['encoder'] = encoder
        run_experiment(input_features, output_features, data_csv=rel_path)


def test_visual_question_answering(csv_filename):
    image_dest_folder = os.path.join(os.getcwd(), 'generated_images')
    input_features = [
        image_feature(
            folder=image_dest_folder,
            encoder='resnet',
            preprocessing={
                'in_memory': True,
                'height': 8,
                'width': 8,
                'num_channels': 3
            },
            fc_size=8,
            num_filters=8
        ),
        text_feature(encoder='embed', min_len=1, level='word'),
    ]
    output_features = [sequence_feature(decoder='generator', cell_type='lstm')]
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, data_csv=rel_path)

    # Delete the temporary data created
    shutil.rmtree(image_dest_folder)


def test_image_resizing_num_channel_handling(csv_filename):
    """
    This test creates two image datasets with 3 channels and 1 channel. The
    combination of this data is used to train a model. This checks the cases
    where the user may or may not specify a number of channels in the
    model definition
    :param csv_filename:
    :return:
    """
    # Image Inputs
    image_dest_folder = os.path.join(os.getcwd(), 'generated_images')

    # Resnet encoder
    input_features = [
        image_feature(
            folder=image_dest_folder,
            encoder='resnet',
            preprocessing={
                'in_memory': True,
                'height': 8,
                'width': 8,
                'num_channels': 3
            },
            fc_size=8,
            num_filters=8
        ),
        text_feature(encoder='embed', min_len=1),
        numerical_feature()
    ]
    output_features = [binary_feature(), numerical_feature()]
    rel_path = generate_data(
        input_features, output_features, csv_filename, num_examples=50
    )

    df1 = read_csv(rel_path)

    input_features[0]['preprocessing']['num_channels'] = 1
    rel_path = generate_data(
        input_features, output_features, csv_filename, num_examples=50
    )
    df2 = read_csv(rel_path)

    df = concatenate_df(df1, df2, None)
    df.to_csv(rel_path, index=False)

    # Here the user sepcifiies number of channels. Exception shouldn't be thrown
    run_experiment(input_features, output_features, data_csv=rel_path)

    del input_features[0]['preprocessing']['num_channels']

    # User now doesn't specify num channels. Should throw exception
    with pytest.raises(ValueError):
        run_experiment(input_features, output_features, data_csv=rel_path)

    # Delete the temporary data created
    shutil.rmtree(image_dest_folder)


if __name__ == '__main__':
    """
    To run tests individually, run:
    ```python -m pytest tests/integration_tests/test_experiment.py::test_name```
    """
    pass
