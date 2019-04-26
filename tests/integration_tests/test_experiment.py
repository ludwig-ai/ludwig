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
import glob
import logging
import os
from string import Template

import pandas as pd
import pytest
import yaml

from ludwig.data.concatenate_datasets import concatenate_df
from ludwig.experiment import experiment
from ludwig.predict import full_predict
from tests.integration_tests.utils import ENCODERS
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import model_definition_template


def run_experiment(input_features, output_features, data_csv):
    """
    Helper method to avoid code repetition in running an experiment
    :param input_features: input schema
    :param output_features: output schema
    :param data_csv: path to data
    :return: None
    """
    model_definition = model_definition_template.substitute(
        input_name=input_features,
        output_name=output_features
    )

    experiment(
        yaml.safe_load(model_definition),
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True,
        data_csv=data_csv
    )


def test_experiment_seq_seq1(csv_filename):
    # Single Sequence input, single sequence output
    # Only the following encoders are working
    input_features_template = Template(
        '[{name: utterance, type: text, reduce_output: null,'
        ' vocab_size: 10, min_len: 10, max_len: 10, encoder: ${encoder}}]')

    output_features = '[{name: iob, type: text, reduce_input: null,' \
                      ' vocab_size: 3, min_len: 10, max_len: 10,' \
                      ' decoder: tagger}]'
    # Generate test data
    rel_path = generate_data(
        input_features_template.substitute(encoder='rnn'),
        output_features,
        csv_filename
    )

    encoders2 = ['embed', 'rnn', 'cnnrnn']
    for encoder in encoders2:
        logging.info('Test 2, Encoder: {0}'.format(encoder))

        input_features = input_features_template.substitute(encoder=encoder)
        run_experiment(input_features, output_features, data_csv=rel_path)


def test_experiment_multi_input_intent_classification(csv_filename):
    # Multiple inputs, Single category output
    input_features_string = Template(
        "[{type: text, name: random_text, vocab_size: 100, max_len: 10,"
        " encoder: ${encoder1}}, {type: category, name: random_category,"
        " vocab_size: 10, encoder: ${encoder2}}]")
    output_features_string = ("[{type: category, name: intent, reduce_input:"
                              " sum, vocab_size: 2}]")

    # Generate test data
    rel_path = generate_data(
        input_features_string.substitute(encoder1='rnn', encoder2='rnn'),
        output_features_string,
        csv_filename
    )

    for encoder1, encoder2 in zip(ENCODERS, ENCODERS):
        input_features = input_features_string.substitute(encoder1=encoder1,
                                                          encoder2=encoder2)

        run_experiment(input_features, output_features_string, rel_path)


def test_experiment_multiple_seq_seq(csv_filename):
    # Multiple inputs, Multiple outputs
    input_features = "[{type: text, name: random_text, vocab_size: 100," \
                     " max_len: 10, encoder: stacked_cnn}, {type: numerical," \
                     " name: random_number}, " \
                     "{type: category, name: random_category, vocab_size: 10," \
                     " encoder: stacked_parallel_cnn}, " \
                     "{type: set, name: random_set, vocab_size: 10," \
                     " max_len: 10}," \
                     "{type: sequence, name: random_sequence, vocab_size: 10," \
                     " max_len: 10, encoder: embed}]"
    output_features = "[{type: category, name: intent, reduce_input: sum," \
                      " vocab_size: 2}," \
                      "{type: sequence, name: random_seq_output, vocab_size: " \
                      "10, max_len: 5}," \
                      "{type: numerical, name: random_num_output}]"

    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, rel_path)

    # Use generator as decoder
    output_features = "[{type: category, name: intent, reduce_input: sum," \
                      " vocab_size: 2, decoder: generator, " \
                      "reduce_input: sum}," \
                      "{type: sequence, name: random_seq_output, " \
                      "vocab_size: 10, max_len: 5}," \
                      "{type: numerical, name: random_num_output}]"

    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, rel_path)

    # Generator decoder and reduce_input = None
    output_features = "[{type: category, name: intent, reduce_input: sum," \
                      " vocab_size: 2}," \
                      "{type: sequence, name: random_seq_op, vocab_size: 10," \
                      " max_len: 5, decoder: generator, reduce_input: None}," \
                      "{type: numerical, name: random_num_op}]"

    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, rel_path)


def test_experiment_image_inputs(csv_filename):
    # Image Inputs
    image_dest_folder = os.path.join(os.getcwd(), 'generated_images')
    input_features_template = Template(
        "[{type: text, name: random_text, vocab_size: 10,"
        " max_len: 10, encoder: stacked_cnn}, {type: numerical,"
        " name: random_number}, "
        "{type: image, name: random_image, encoder: ${encoder},"
        "preprocessing: {in_memory: ${in_memory}, height: 8, width: 8, "
        "num_channels: 3}, resnet_size: 8, destination_folder: ${folder}, "
        "fc_size: 32, num_filters: 8}]")

    # Resnet encoder
    input_features = input_features_template.substitute(
        encoder='resnet',
        folder=image_dest_folder,
        in_memory='true',
    )
    output_features = "[{type: category, name: intent, reduce_input: sum," \
                      " vocab_size: 2}," \
                      "{type: numerical, name: random_num_output}]"

    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, rel_path)

    # Stacked CNN encoder
    input_features = input_features_template.substitute(
        encoder='stacked_cnn',
        folder=image_dest_folder,
        in_memory='true',
    )

    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, rel_path)

    # Stacked CNN encoder
    input_features = input_features_template.substitute(
        encoder='stacked_cnn',
        folder=image_dest_folder,
        in_memory='false',
    )

    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, rel_path)

    # Delete the temporary data created
    all_images = glob.glob(os.path.join(image_dest_folder, '*.jpg'))
    for im in all_images:
        os.remove(im)

    os.rmdir(image_dest_folder)


def test_experiment_tied_weights(csv_filename):
    # Single sequence input, single category output
    input_features = Template('[{name: utterance1, type: text,'
                              'vocab_size: 10, max_len: 10, '
                              'encoder: ${encoder}, reduce_output: sum},'
                              '{name: utterance2, type: text, vocab_size: 10,'
                              'max_len: 10, encoder: ${encoder}, '
                              'reduce_output: sum, tied_weights: utterance1}]')
    output_features = "[{name: intent, type: category, vocab_size: 2," \
                      " reduce_input: sum}] "

    # Generate test data
    rel_path = generate_data(
        input_features.substitute(encoder='rnn'),
        output_features,
        csv_filename
    )
    for encoder in ENCODERS:
        run_experiment(input_features.substitute(encoder=encoder),
                       output_features,
                       data_csv=rel_path)


def test_experiment_attention(csv_filename):
    # Machine translation with attention
    input_features = '[{name: english, type: sequence, vocab_size: 10,' \
                     ' max_len: 10, encoder: rnn, cell_type: lstm} ]'
    output_features = Template("[{name: spanish, type: sequence,"
                               " vocab_size: 10, max_len: 10,"
                               " decoder: generator, cell_type: lstm,"
                               " attention: ${attention}}] ")

    # Generate test data
    rel_path = generate_data(
        input_features,
        output_features.substitute(attention='bahdanau'),
        csv_filename
    )

    for attention in ['bahdanau', 'luong']:
        run_experiment(input_features, output_features.substitute(
            attention=attention), data_csv=rel_path)


def test_experiment_sequence_combiner(csv_filename):
    # Sequence combiner
    input_features_template = Template(
        '[{name: english, type: sequence, vocab_size: 10,'
        ' max_len: 5, min_len: 5, encoder: rnn, cell_type: lstm,'
        ' reduce_output: null}, {name: spanish, type: sequence, vocab_size: 10,'
        ' max_len: 5, min_len: 5, encoder: rnn, cell_type: lstm,'
        ' reduce_output: null}, {name: category,'
        ' type: category, vocab_size: 5} ]')

    output_features_string = "[{type: category, name: intent, reduce_input:" \
                             " sum, vocab_size: 5}]"

    model_definition_template2 = Template(
        '{input_features: ${input_name}, output_features: ${output_name}, '
        'training: {epochs: 2}, combiner: {type: sequence_concat, encoder: rnn,'
        'main_sequence_feature: random_sequence}}')

    # Generate test data
    rel_path = generate_data(
        input_features_template.substitute(encoder1='rnn', encoder2='rnn'),
        output_features_string,
        csv_filename
    )

    for encoder1, encoder2 in zip(ENCODERS, ENCODERS):
        input_features = input_features_template.substitute(
            encoder1=encoder1,
            encoder2=encoder2)

        model_definition = model_definition_template2.substitute(
            input_name=input_features,
            output_name=output_features_string
        )

        experiment(yaml.safe_load(model_definition),
                   skip_save_processed_input=True,
                   skip_save_progress=True,
                   skip_save_unprocessed_output=True, data_csv=rel_path
                   )


def test_experiment_model_resume(csv_filename):
    # Single sequence input, single category output
    # Tests saving a model file, loading it to rerun training and predict
    input_features = '[{name: utterance, type: sequence, vocab_size: 10,' \
                     ' max_len: 10, encoder: rnn, reduce_output: sum}]'
    output_features = "[{name: intent, type: category, vocab_size: 2," \
                      " reduce_input: sum}] "

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    model_definition = model_definition_template.substitute(
        input_name=input_features, output_name=output_features
    )

    exp_dir_name = experiment(yaml.safe_load(model_definition),
                              data_csv=rel_path)
    logging.info('Experiment Directory: {0}'.format(exp_dir_name))

    experiment(yaml.safe_load(model_definition), data_csv=rel_path,
               model_resume_path=exp_dir_name)

    full_predict(os.path.join(exp_dir_name, 'model'), data_csv=rel_path)


def test_experiment_various_feature_types(csv_filename):
    input_features_template = Template(
        '[{name: binary_input, type: binary}, '
        '{name: bag_input, type: bag, max_len: 5, vocab_size: 10,'
        ' encoder: ${encoder}}]')
    # {name: intent_binary, type: binary},
    output_features = "[ {name: set_output, type: set, max_len: 3," \
                      " vocab_size: 5}] "

    # Generate test data
    rel_path = generate_data(
        input_features_template.substitute(encoder='rnn'),
        output_features,
        csv_filename
    )
    for encoder in ENCODERS:
        run_experiment(input_features_template.substitute(encoder=encoder),
                       output_features, data_csv=rel_path)


def test_experiment_timeseries(csv_filename):
    input_features_template = Template(
        '[{name: time_series, type: timeseries, max_len: 10}]')
    output_features = "[ {name: binary_output, type: binary}, ]"

    # Generate test data
    rel_path = generate_data(
        input_features_template.substitute(encoder='rnn'),
        output_features,
        csv_filename
    )
    for encoder in ENCODERS:
        run_experiment(input_features_template.substitute(encoder=encoder),
                       output_features, data_csv=rel_path)


def test_visual_question_answering(csv_filename):
    image_dest_folder = os.path.join(os.getcwd(), 'generated_images')
    input_features = Template(
        '[{name: image_path, type: image, encoder: stacked_cnn, preprocessing:'
        '{width: 10, height: 10, num_channels: 3}, resnet_size: 8, fc_size: 32,'
        'destination_folder: ${folder}}, {name: question, type: text, '
        'vocab_size: 20, max_len: 10,encoder: parallel_cnn, '
        'level: word}]').substitute(
        folder=image_dest_folder)

    output_features = "[ {name: answer, type: sequence, max_len: 5, " \
                      "vocab_size: 10, decoder: generator, cell_type: lstm}]"

    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, rel_path)

    # Delete the temporary data created
    all_images = glob.glob(os.path.join(image_dest_folder, '*.jpg'))
    for im in all_images:
        os.remove(im)

    os.rmdir(image_dest_folder)


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
    input_features_template = Template(
        "[{type: text, name: random_text, vocab_size: 100,"
        " max_len: 10, encoder: stacked_cnn}, {type: numerical,"
        " name: random_number}, "
        "{type: image, name: random_image, encoder: ${encoder}, preprocessing:"
        " {width: 8, in_memory: ${in_memory},"
        " height: 8, num_channels: 3},"
        " resnet_size: 8, fc_size: 32, destination_folder: ${folder}}]")

    # Resnet encoder
    input_features = input_features_template.substitute(
        encoder='resnet',
        folder=image_dest_folder,
        in_memory='true',
    )
    output_features = "[{type: binary, name: intent, reduce_input: sum}, " \
                      "{type: numerical, name: random_num_output}]"

    rel_path = generate_data(
        input_features, output_features, csv_filename, num_examples=100
    )

    df1 = pd.read_csv(rel_path)

    input_features_template = Template(
        "[{type: text, name: random_text, vocab_size: 100,"
        " max_len: 10, encoder: stacked_cnn}, {type: numerical,"
        " name: random_number}, "
        "{type: image, name: random_image, preprocessing: {width: 8,"
        " in_memory: ${in_memory}, height: 8, num_channels: 1},"
        " encoder: ${encoder}, fc_size: 32, "
        " resnet_size: 8, destination_folder: ${folder}}]")

    input_features = input_features_template.substitute(
        encoder='resnet',
        folder=image_dest_folder,
        in_memory='true',
    )
    rel_path = generate_data(
        input_features, output_features, csv_filename, num_examples=100
    )
    df2 = pd.read_csv(rel_path)

    df = concatenate_df(df1, df2, None)
    df.to_csv(rel_path, index=False)

    # Here the user sepcifiies number of channels. Exception shouldn't be thrown
    run_experiment(input_features, output_features, rel_path)

    input_features_template = Template(
        "[{type: text, name: random_text, vocab_size: 100,"
        " max_len: 10, encoder: stacked_cnn}, {type: numerical,"
        " name: random_number}, "
        "{type: image, name: random_image, preprocessing: {width: 8, "
        "in_memory: ${in_memory}, height: 8} , encoder: ${encoder},"
        " resnet_size: 8, fc_size: 32, destination_folder: ${folder}}]")

    input_features = input_features_template.substitute(
        encoder='resnet',
        folder=image_dest_folder,
        in_memory='true',
    )
    # User now doesn't specify num channels. Should throw exception
    with pytest.raises(ValueError):
        run_experiment(input_features, output_features, rel_path)

    # Delete the temporary data created
    all_images = glob.glob(os.path.join(image_dest_folder, '*.jpg'))
    for im in all_images:
        os.remove(im)

    os.rmdir(image_dest_folder)


if __name__ == '__main__':
    """
    To run tests individually, run:
    ```python -m pytest tests/integration_tests/test_experiment.py::test_name```
    """
    pass
