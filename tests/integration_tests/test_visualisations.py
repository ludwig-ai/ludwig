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
#
# Integration tests for the visualisation commands.
#
# Author: Ivaylo Stefanov
# email: ivaylo.stefanov82@gmail.com
# github: https://github.com/istefano82
# ==============================================================================

import glob
import shutil
import subprocess
import json
import os

from ludwig.experiment import experiment

from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import text_feature, categorical_feature, \
    numerical_feature, set_feature, sequence_feature, binary_feature, \
    bag_feature


# The following imports are pytest fixtures, required for running the tests
from tests.fixtures.filenames import csv_filename


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
        'skip_save_processed_input': False,
        'skip_save_progress': False,
        'skip_save_unprocessed_output': False,
    }
    args.update(kwargs)

    exp_dir_name = experiment(**args)

    return exp_dir_name


def get_output_field_name(experiment_dir, output_feature=0):
    """Helper function to extract specified output feature name.

    :param experiment_dir: Path to the experiment directory
    :param output_feature: position of the output feature the description.json
    :return field_name: name of the first output feature name
                        from the experiment
    """
    description_file = experiment_dir + '/description.json'
    with open(description_file, 'rb') as f:
        content = json.load(f)
    field_name = \
        content['model_definition']['output_features'][output_feature]['name']
    return field_name


def test_visualisation_learning_curves_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [text_feature(reduce_output=None, encoder='rnn')]
    output_features = [text_feature(reduce_input=None, decoder='tagger')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)

    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    train_stats = exp_dir_name + '/training_statistics.json'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'learning_curves',
                    '--training_statistics',
                    train_stats,
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 5 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)


def test_visualisation_confusion_matrix_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [text_feature(reduce_output=None, encoder='rnn')]
    output_features = [text_feature(reduce_input=None, decoder='tagger')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth_metadata = experiment_source_data_name + '.json'
    test_stats = exp_dir_name + '/test_statistics.json'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'confusion_matrix',
                    '--test_statistics',
                    test_stats,
                    '--ground_truth_metadata',
                    ground_truth_metadata,
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']
    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 2 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_compare_performance_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    Compare performance between two models. To reduce test complexity
    one model is compared to it self.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [text_feature(reduce_output=None, encoder='rnn')]
    output_features = [text_feature(reduce_input=None, decoder='tagger')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    experiment_source_data_name = csv_filename.split('.')[0]
    test_stats = exp_dir_name + '/test_statistics.json'

    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'compare_performance',
                    '--test_statistics',
                    test_stats,
                    test_stats,
                    '-m',
                    'Model1',
                    'Model2',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 2 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_compare_classifiers_from_prob_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
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
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)

    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'compare_classifiers_performance_from_prob',
                    '--ground_truth',
                    ground_truth,
                    '--field',
                    field_name,
                    '--probabilities',
                    probability,
                    probability,
                    '--model_names',
                    'Model1',
                    'Model2',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_compare_classifiers_from_pred_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
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
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    prediction = exp_dir_name + '/{}_predictions.csv'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    ground_truth_metadata = experiment_source_data_name + '.json'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'compare_classifiers_performance_from_pred',
                    '--ground_truth_metadata',
                    ground_truth_metadata,
                    '--ground_truth',
                    ground_truth,
                    '--field',
                    field_name,
                    '--predictions',
                    prediction,
                    prediction,
                    '--model_names',
                    'Model1',
                    'Model2',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_compare_classifiers_subset_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
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
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'compare_classifiers_performance_subset',
                    '--field',
                    field_name,
                    '--probabilities',
                    probability,
                    probability,
                    '--model_names',
                    'Model1',
                    'Model2',
                    '--ground_truth',
                    ground_truth,
                    '--top_n_classes',
                    '6',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_compare_classifiers_changing_k_output_pdf(csv_filename):
    """It should be possible to save figures as pdf in the specified directory.

    """
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
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'compare_classifiers_performance_changing_k',
                    '--field',
                    field_name,
                    '--probabilities',
                    probability,
                    probability,
                    '--model_names',
                    'Model1',
                    'Model2',
                    '--ground_truth',
                    ground_truth,
                    '--top_n_classes',
                    '6',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]
    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_compare_classifiers_multiclass_multimetric_output_saved(
        csv_filename
):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
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
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    test_stats = exp_dir_name + '/test_statistics.json'
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth_metadata = experiment_source_data_name + '.json'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'compare_classifiers_multiclass_multimetric',
                    '--field',
                    field_name,
                    '--test_statistics',
                    test_stats,
                    test_stats,
                    '--ground_truth_metadata',
                    ground_truth_metadata,
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 4 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_compare_classifiers_predictions_output_saved(
        csv_filename
):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
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
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    prediction = exp_dir_name + '/{}_predictions.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    ground_truth_metadata = experiment_source_data_name + '.json'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'compare_classifiers_predictions',
                    '--ground_truth_metadata',
                    ground_truth_metadata,
                    '--ground_truth',
                    ground_truth,
                    '--field',
                    field_name,
                    '--predictions',
                    prediction,
                    prediction,
                    '--model_names',
                    'Model1',
                    'Model2',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_cmp_classifiers_predictions_distribution_output_saved(
        csv_filename
):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
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
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    prediction = exp_dir_name + '/{}_predictions.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    ground_truth_metadata = experiment_source_data_name + '.json'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'compare_classifiers_predictions_distribution',
                    '--ground_truth_metadata',
                    ground_truth_metadata,
                    '--ground_truth',
                    ground_truth,
                    '--field',
                    field_name,
                    '--predictions',
                    prediction,
                    prediction,
                    '--model_names',
                    'Model1',
                    'Model2',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_cconfidence_thresholding_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
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
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'confidence_thresholding',
                    '--ground_truth',
                    ground_truth,
                    '--field',
                    field_name,
                    '--probabilities',
                    probability,
                    probability,
                    '--model_names',
                    'Model1',
                    'Model2',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_confidence_thresholding_data_vs_acc_output_saved(
        csv_filename
):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
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
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'confidence_thresholding_data_vs_acc',
                    '--ground_truth',
                    ground_truth,
                    '--field',
                    field_name,
                    '--probabilities',
                    probability,
                    probability,
                    '--model_names',
                    'Model1',
                    'Model2',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_confidence_thresholding_data_vs_acc_subset_output_saved(
        csv_filename
):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
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
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'confidence_thresholding_data_vs_acc_subset',
                    '--ground_truth',
                    ground_truth,
                    '--field',
                    field_name,
                    '--probabilities',
                    probability,
                    probability,
                    '--model_names',
                    'Model1',
                    'Model2',
                    '--top_n_classes',
                    '3',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_vis_confidence_thresholding_data_vs_acc_subset_per_class_output_saved(
        csv_filename
):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        categorical_feature(
            vocab_size=10,
            loss='sampled_softmax_cross_entropy'
        )
    ]
    output_features = [categorical_feature(vocab_size=5, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    ground_truth_metadata = experiment_source_data_name + '.json'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'confidence_thresholding_data_vs_acc_subset_per_class',
                    '--ground_truth',
                    ground_truth,
                    '--ground_truth_metadata',
                    ground_truth_metadata,
                    '--field',
                    field_name,
                    '--probabilities',
                    probability,
                    probability,
                    '--model_names',
                    'Model1',
                    'Model2',
                    '--top_n_classes',
                    '3',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        # 3 figures should be saved because experiment setting top_n_classes = 3
        # hence one figure per class
        assert 3 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_vis_confidence_thresholding_2thresholds_2d_output_saved(
        csv_filename
):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
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

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    treshhold_field1 = get_output_field_name(exp_dir_name)
    treshhold_field2 = get_output_field_name(exp_dir_name, output_feature=1)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(
        treshhold_field1
    )
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'confidence_thresholding_2thresholds_2d',
                    '--ground_truth',
                    ground_truth,
                    '--probabilities',
                    probability,
                    probability,
                    '--threshold_fields',
                    treshhold_field1,
                    treshhold_field2,
                    '--model_names',
                    'Model1',
                    'Model2',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 3 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_vis_confidence_thresholding_2thresholds_3d_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
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

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    treshhold_field1 = get_output_field_name(exp_dir_name)
    treshhold_field2 = get_output_field_name(exp_dir_name, output_feature=1)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(
        treshhold_field1
    )
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'confidence_thresholding_2thresholds_3d',
                    '--ground_truth',
                    ground_truth,
                    '--probabilities',
                    probability,
                    probability,
                    '--threshold_fields',
                    treshhold_field1,
                    treshhold_field2,
                    '--model_names',
                    'Model1',
                    'Model2',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_binary_threshold_vs_metric_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
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

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    ground_truth_metadata = experiment_source_data_name + '.json'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'binary_threshold_vs_metric',
                    '--positive_label',
                    '2',
                    '--metrics',
                    'accuracy',
                    '--ground_truth',
                    ground_truth,
                    '--ground_truth_metadata',
                    ground_truth_metadata,
                    '--field',
                    field_name,
                    '--probabilities',
                    probability,
                    probability,
                    '--model_names',
                    'Model1',
                    'Model2',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_roc_curves_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
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
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    ground_truth_metadata = experiment_source_data_name + '.json'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'roc_curves',
                    '--positive_label',
                    '2',
                    '--metrics',
                    'accuracy',
                    '--ground_truth',
                    ground_truth,
                    '--ground_truth_metadata',
                    ground_truth_metadata,
                    '--field',
                    field_name,
                    '--probabilities',
                    probability,
                    probability,
                    '--model_names',
                    'Model1',
                    'Model2',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_roc_curves_from_test_statistics_output_saved(
        csv_filename
):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [binary_feature(), bag_feature()]
    output_features = [binary_feature()]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    test_stats = exp_dir_name + '/test_statistics.json'
    experiment_source_data_name = csv_filename.split('.')[0]
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'roc_curves_from_test_statistics',
                    '--field',
                    field_name,
                    '--test_statistics',
                    test_stats,
                    '--model_names',
                    'Model1',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_calibration_1_vs_all_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=50, min_len=1, representation='sparse'),
        categorical_feature(
            vocab_size=10,
            loss='sampled_softmax_cross_entropy'
        )
    ]
    output_features = [categorical_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'calibration_1_vs_all',
                    '--positive_label',
                    '2',
                    '--metrics',
                    'accuracy',
                    '--ground_truth',
                    ground_truth,
                    '--field',
                    field_name,
                    '--probabilities',
                    probability,
                    probability,
                    '--model_names',
                    'Model1',
                    'Model2',
                    '--top_k',
                    '6',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 5 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_calibration_multiclass_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=50, min_len=1, representation='sparse'),
        categorical_feature(
            vocab_size=10,
            loss='sampled_softmax_cross_entropy'
        )
    ]
    output_features = [categorical_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'calibration_multiclass',
                    '--ground_truth',
                    ground_truth,
                    '--field',
                    field_name,
                    '--probabilities',
                    probability,
                    probability,
                    '--model_names',
                    'Model1',
                    'Model2',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 2 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_frequency_vs_f1_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=50, min_len=1, representation='sparse'),
        categorical_feature(
            vocab_size=10,
            loss='sampled_softmax_cross_entropy'
        )
    ]
    output_features = [categorical_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    test_stats = exp_dir_name + '/test_statistics.json'
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth_metadata = experiment_source_data_name + '.json'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'frequency_vs_f1',
                    '--ground_truth_metadata',
                    ground_truth_metadata,
                    '--field',
                    field_name,
                    '--test_statistics',
                    test_stats,
                    test_stats,
                    '--model_names',
                    'Model1',
                    'Model2',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 2 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))
