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
import glob
import sys
from string import Template

import pandas as pd
import yaml

from ludwig.data.dataset_synthesyzer import build_synthetic_dataset
from ludwig.experiment import experiment

encoders = ['embed', 'rnn', 'parallel_cnn', 'cnnrnn', 'stacked_parallel_cnn',
            'stacked_cnn']

model_definition_template = Template(
    '{input_features: ${input_name}, output_features: ${output_name}, '
    'training: {epochs: 2}}')


def generate_data(input_features, output_features, filename='test_csv.csv'):
    features = yaml.load(input_features) + yaml.load(output_features)
    df = build_synthetic_dataset(1000, features)
    data = [next(df) for _ in range(1000)]

    dataframe = pd.DataFrame(data[1:], columns=data[0])

    dataframe.to_csv(filename, index=False)

    return filename


def run_experiment(input_features, output_features, data_csv):
    model_definition = model_definition_template.substitute(
        input_name=input_features,
        output_name=output_features
    )

    experiment(yaml.load(model_definition), skip_save_processed_input=True,
               skip_save_progress_weights=True,
               skip_save_unprocessed_output=True, data_csv=data_csv
               )


def test_experiment_1(delete_temp_data=True):
    # Single sequence input, single category output
    input_features = Template('[{name: utterance, type: sequence,'
                              'vocab_size: 10, max_len: 10, '
                              'encoder: ${encoder}, reduce_output: sum}]')
    output_features = "[{name: intent, type: category, vocab_size: 2," \
                      " reduce_input: sum}] "

    # Generate test data
    rel_path = generate_data(input_features.substitute(encoder='rnn'),
                             output_features, 'test_csv.csv')
    for encoder in encoders:
        run_experiment(input_features.substitute(encoder=encoder),
                       output_features,
                       data_csv=rel_path)

    # Delete the generated data
    if delete_temp_data is True:
        os.remove(rel_path)


def test_experiment_2(delete_temp_data=True):
    # Single Sequence input, single sequence output
    # Only the following encoders are working
    input_features_template = Template(
        '[{name: utterance, type: sequence, reduce_output: null,'
        ' vocab_size: 10, min_len: 10, max_len: 10, encoder: ${encoder}}]')

    output_features = '[{name: iob, type: sequence, reduce_input: null,' \
                      ' vocab_size: 3, min_len: 10, max_len: 10,' \
                      ' decoder: tagger}]'
    # Generate test data
    rel_path = generate_data(
        input_features_template.substitute(encoder='rnn'),
        output_features, 'test_csv.csv')

    encoders2 = ['embed', 'rnn', 'cnnrnn']
    for encoder in encoders2:
        logging.info('Test 2, Encoder: {0}'.format(encoder))

        input_features = input_features_template.substitute(encoder=encoder)
        run_experiment(input_features, output_features, data_csv=rel_path)

    # Delete the generated data
    if delete_temp_data is True:
        os.remove(rel_path)


def test_experiment_3(delete_temp_data=True):
    # Multiple inputs, Single category output
    input_features_string = Template(
        "[{type: text, name: random_text, vocab_size: 100, max_len: 20,"
        " encoder: ${encoder1}}, {type: numerical, name: random_number}, "
        "{type: category, name: random_category, vocab_size: 10,"
        " encoder: ${encoder2}}, {type: set, name: random_set, vocab_size: 10,"
        " max_len: 10}, {type: sequence, name: random_sequence, vocab_size: 10,"
        " max_len: 10}]")
    output_features_string = "[{type: category, name: intent, reduce_input:" \
                             " sum, vocab_size: 2}]"

    # Generate test data
    rel_path = generate_data(
        input_features_string.substitute(encoder1='rnn', encoder2='rnn'),
        output_features_string, 'test_csv.csv')

    for encoder1, encoder2 in zip(encoders, encoders):
        input_features = input_features_string.substitute(encoder1=encoder1,
                                                          encoder2=encoder2)

        run_experiment(input_features, output_features_string, rel_path)

    # Delete the generated data
    if delete_temp_data is True:
        os.remove(rel_path)


def test_experiment_4(delete_temp_data=True):
    # Multiple inputs, Multiple outputs
    input_features = "[{type: text, name: random_text, vocab_size: 100," \
                     " max_len: 20, encoder: stacked_cnn}, {type: numerical," \
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

    rel_path = generate_data(input_features, output_features, 'test_csv.csv')
    run_experiment(input_features, output_features, rel_path)

    input_features = "[{type: text, name: random_text, vocab_size: 100," \
                     " max_len: 20, encoder: stacked_cnn}, " \
                     "{type: numerical, name: random_number}, " \
                     "{type: category, name: random_category, vocab_size: 10," \
                     " encoder: stacked_parallel_cnn}, " \
                     "{type: set, name: random_set, vocab_size: 10," \
                     " max_len: 10}," \
                     "{type: sequence, name: random_sequence, vocab_size: 10," \
                     " max_len: 10, encoder: embed}]"
    output_features = "[{type: category, name: intent, reduce_input: sum," \
                      " vocab_size: 2, decoder: generator, " \
                      "reduce_input: sum}," \
                      "{type: sequence, name: random_seq_output, " \
                      "vocab_size: 10, max_len: 5}," \
                      "{type: numerical, name: random_num_output}]"

    rel_path = generate_data(input_features, output_features, 'test_csv.csv')
    run_experiment(input_features, output_features, rel_path)

    input_features = "[{type: text, name: random_text, vocab_size: 100," \
                     " max_len: 20, encoder: stacked_cnn}, " \
                     "{type: numerical, name: random_number}, " \
                     "{type: category, name: random_category, vocab_size: 10," \
                     " encoder: stacked_parallel_cnn}, " \
                     "{type: set, name: random_set, vocab_size: 10," \
                     " max_len: 10}," \
                     "{type: sequence, name: random_sequence, vocab_size: 10," \
                     " max_len: 10, encoder: embed}]"
    output_features = "[{type: category, name: intent, reduce_input: sum," \
                      " vocab_size: 2}," \
                      "{type: sequence, name: random_seq_op, vocab_size: 10," \
                      " max_len: 5, decoder: generator, reduce_input: None}," \
                      "{type: numerical, name: random_num_op}]"

    rel_path = generate_data(input_features, output_features, 'test_csv.csv')
    run_experiment(input_features, output_features, rel_path)

    if delete_temp_data is True:
        os.remove(rel_path)


def test_experiment_5(delete_temp_data=True):
    # Image Inputs
    image_dest_folder = os.path.join(os.getcwd(), 'generated_images')
    input_features_template = Template(
        "[{type: text, name: random_text, vocab_size: 100,"
        " max_len: 20, encoder: stacked_cnn}, {type: numerical,"
        " name: random_number}, "
        "{type: image, name: random_image, width: 25,"
        " height: 25, num_channels: 3, encoder: ${encoder},"
        " resnet_size: 8, destination_folder: ${folder}}]")

    # Resnet encoder
    input_features = input_features_template.substitute(encoder='resnet',
                                                        folder=image_dest_folder)
    output_features = "[{type: category, name: intent, reduce_input: sum," \
                      " vocab_size: 2}," \
                      "{type: numerical, name: random_num_output}]"

    rel_path = generate_data(input_features, output_features, 'test_csv.csv')
    run_experiment(input_features, output_features, rel_path)

    # Stacked CNN encoder
    input_features = input_features_template.substitute(encoder='stacked_cnn',
                                                        folder=image_dest_folder)

    rel_path = generate_data(input_features, output_features, 'test_csv.csv')
    run_experiment(input_features, output_features, rel_path)

    # Delete the temporary data created
    if delete_temp_data is True:
        all_images = glob.glob(os.path.join(image_dest_folder, '*.jpg'))
        for im in all_images:
            os.remove(im)

        os.rmdir(image_dest_folder)
        os.remove(rel_path)

        
def test_experiment_tied_weights(delete_temp_data=True):
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
        output_features, 'test_csv.csv')
    for encoder in encoders:
        run_experiment(input_features.substitute(encoder=encoder),
                       output_features,
                       data_csv=rel_path)

    # Delete the generated data
    if delete_temp_data is True:
        os.remove(rel_path)


def test_experiment_attention(delete_temp_data=True):
    # Machine translation with attention
    input_features = '[{name: english, type: sequence, vocab_size: 10,' \
                     ' max_len: 10, encoder: rnn, cell_type: lstm} ]'
    output_features = Template("[{name: spanish, type: sequence,"
                               " vocab_size: 10, max_len: 10,"
                               " decoder: generator, cell_type: lstm,"
                               " attention: ${attention}}] ")

    # Generate test data
    rel_path = generate_data(
        input_features, output_features.substitute(attention='bahdanau'),
        'test_csv.csv')

    for attention in ['bahdanau', 'luong']:
        run_experiment(input_features, output_features.substitute(
            attention=attention), data_csv=rel_path)

    # Delete the generated data
    if delete_temp_data is True:
        os.remove(rel_path)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    test_experiment_1()
    test_experiment_2()
    test_experiment_3()
    test_experiment_4()
    test_experiment_5()
    test_experiment_tied_weights()
    test_experiment_attention()

