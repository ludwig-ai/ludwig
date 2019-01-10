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
import sys
from string import Template

import pandas as pd
import yaml

sys.path.append("../")

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
    rel_path = os.path.join(os.path.dirname(__file__),
                            '../../data/{0}'.format(filename))

    dataframe.to_csv(rel_path, index=False)

    return rel_path


def run_experiment(input_features, output_features, data_csv):
    model_definition = model_definition_template.substitute(
        input_name=input_features,
        output_name=output_features
    )

    experiment(yaml.load(model_definition), skip_save_processed_input=True,
               skip_save_progress_weights=True,
               skip_save_unprocessed_output=True, data_csv=data_csv
               )


def test_experiment_1():
    # Single sequence input, single category output

    model_definition_template = Template(
        "{input_features: [{name: utterance, type: sequence, "
        "encoder: ${encoder}, reduce_output: sum}], output_features:"
        "[{name: intent, type: category, reduce_input: sum}] "
        ", training: {epochs: 2}}"
    )
    input_features = Template('[{name: utterance, type: sequence, '
                              'encoder: ${encoder}, reduce_output: sum}]')
    output_features = "[{name: intent, type: category, reduce_input: sum}] "

    for encoder in encoders:
        run_experiment(input_features.substitute(encoder=encoder),
                       output_features,
                       data_csv='data/atis.train.iob.sample.csv')


def test_experiment_2():
    # Single Sequence input, single sequence output
    # Only the following encoders are working
    input_features_template = Template(
        '[{name: utterance, type: sequence, reduce_output: null, encoder: ${encoder}}]')
    output_features = '[{name: iob, type: sequence, reduce_input: null, decoder: tagger}]'

    encoders2 = ['embed', 'rnn', 'cnnrnn']
    for encoder in encoders2:
        logging.info('Test 2, Encoder: {0}'.format(encoder))

        input_features = input_features_template.substitute(encoder=encoder)
        run_experiment(input_features, output_features,
                       data_csv='data/atis.train.iob.sample.csv')


def test_experiment_3(delete_temp_data=True):
    # Multiple inputs, Single category output
    input_features_string = Template(
        "[{type: text, name: random_text, vocab_size: 100, max_len: 20, encoder: ${encoder1}}, "
        "{type: numerical, name: random_number}, "
        "{type: category, name: random_category, vocab_size: 10, encoder: ${encoder2}}, "
        "{type: set, name: random_set, vocab_size: 10, max_len: 10},"
        "{type: sequence, name: random_sequence, vocab_size: 10, max_len: 10}]")
    output_features_string = "[{type: category, name: intent, reduce_input: sum, vocab_size: 2}]"

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
    input_features = "[{type: text, name: random_text, vocab_size: 100, max_len: 20, encoder: stacked_cnn}, " \
                     "{type: numerical, name: random_number}, " \
                     "{type: category, name: random_category, vocab_size: 10, encoder: stacked_parallel_cnn}, " \
                     "{type: set, name: random_set, vocab_size: 10, max_len: 10}," \
                     "{type: sequence, name: random_sequence, vocab_size: 10, max_len: 10, encoder: embed}]"
    output_features = "[{type: category, name: intent, reduce_input: sum, vocab_size: 2}," \
                      "{type: sequence, name: random_seq_output, vocab_size: 10, max_len: 5}," \
                      "{type: numerical, name: random_num_output}]"

    rel_path = generate_data(input_features, output_features, 'test_csv.csv')
    run_experiment(input_features, output_features, rel_path)

    input_features = "[{type: text, name: random_text, vocab_size: 100, max_len: 20, encoder: stacked_cnn}, " \
                     "{type: numerical, name: random_number}, " \
                     "{type: category, name: random_category, vocab_size: 10, encoder: stacked_parallel_cnn}, " \
                     "{type: set, name: random_set, vocab_size: 10, max_len: 10}," \
                     "{type: sequence, name: random_sequence, vocab_size: 10, max_len: 10, encoder: embed}]"
    output_features = "[{type: category, name: intent, reduce_input: sum, vocab_size: 2, decoder: generator, reduce_input: sum}," \
                      "{type: sequence, name: random_seq_output, vocab_size: 10, max_len: 5}," \
                      "{type: numerical, name: random_num_output}]"

    rel_path = generate_data(input_features, output_features, 'test_csv.csv')
    run_experiment(input_features, output_features, rel_path)

    input_features = "[{type: text, name: random_text, vocab_size: 100, max_len: 20, encoder: stacked_cnn}, " \
                     "{type: numerical, name: random_number}, " \
                     "{type: category, name: random_category, vocab_size: 10, encoder: stacked_parallel_cnn}, " \
                     "{type: set, name: random_set, vocab_size: 10, max_len: 10}," \
                     "{type: sequence, name: random_sequence, vocab_size: 10, max_len: 10, encoder: embed}]"
    output_features = "[{type: category, name: intent, reduce_input: sum, vocab_size: 2}," \
                      "{type: sequence, name: random_seq_output, vocab_size: 10, max_len: 5, decoder: generator, reduce_input: None}," \
                      "{type: numerical, name: random_num_output}]"

    rel_path = generate_data(input_features, output_features, 'test_csv.csv')
    run_experiment(input_features, output_features, rel_path)

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
