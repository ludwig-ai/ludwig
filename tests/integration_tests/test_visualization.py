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
# Integration tests for the visualization commands.
#
# Author: Ivaylo Stefanov
# email: ivaylo.stefanov82@gmail.com
# github: https://github.com/istefano82
# ==============================================================================

import glob
import json
import os
import shutil
import subprocess

import numpy as np

from ludwig.experiment import full_experiment
from ludwig.utils.data_utils import load_from_file, load_json
from tests.integration_tests.test_visualization_api import obtain_df_splits
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import text_feature, category_feature, \
    numerical_feature, set_feature, sequence_feature, binary_feature, \
    bag_feature


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

    exp_dir_name = full_experiment(**args)

    return exp_dir_name


def get_output_feature_name(experiment_dir, output_feature=0):
    """Helper function to extract specified output feature name.

    :param experiment_dir: Path to the experiment directory
    :param output_feature: position of the output feature the description.json
    :return output_feature_name: name of the first output feature name
                        from the experiment
    """
    description_file = experiment_dir + '/description.json'
    with open(description_file, 'rb') as f:
        content = json.load(f)
    output_feature_name = \
        content['model_definition']['output_features'][output_feature]['name']
    return output_feature_name


def test_visualization_learning_curves_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [text_feature(encoder='parallel_cnn')]
    output_features = [category_feature()]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )

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


def test_visualization_confusion_matrix_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [text_feature(encoder='parallel_cnn')]
    output_features = [category_feature()]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
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
        result = subprocess.run(command)
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


def test_visualization_compare_performance_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    Compare performance between two models. To reduce test complexity
    one model is compared to it self.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [text_feature(encoder='parallel_cnn')]
    output_features = [category_feature()]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
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

def test_visualization_compare_classifiers_from_prob_csv_output_saved(
        csv_filename
):
    """Ensure pdf and png figures from the experiments can be saved.

    Probabilities are loaded from csv file.
    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        category_feature(vocab_size=10)
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )

    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.csv'.format(output_feature_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'compare_classifiers_performance_from_prob',
                    '--ground_truth',
                    ground_truth,
                    '--output_feature_name',
                    output_feature_name,
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
        result = subprocess.run(command)
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


def test_visualization_compare_classifiers_from_prob_npy_output_saved(
        csv_filename
):
    """Ensure pdf and png figures from the experiments can be saved.

    Probabilities are loaded from npy file.
    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        category_feature(vocab_size=10)
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )

    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(output_feature_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'compare_classifiers_performance_from_prob',
                    '--ground_truth',
                    ground_truth,
                    '--output_feature_name',
                    output_feature_name,
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
        result = subprocess.run(command)
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

def test_visualization_compare_classifiers_from_pred_npy_output_saved(
        csv_filename
):
    """Ensure pdf and png figures from the experiments can be saved.

    Predictions are loaded from npy file.
    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        category_feature(vocab_size=10)
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    output_feature_name = get_output_feature_name(exp_dir_name)
    prediction = exp_dir_name + '/{}_predictions.npy'.format(output_feature_name)
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
                    '--output_feature_name',
                    output_feature_name,
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
        result = subprocess.run(command)
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

def test_visualization_compare_classifiers_from_pred_csv_output_saved(
        csv_filename
):
    """Ensure pdf and png figures from the experiments can be saved.

    Predictions are loaded from csv file.
    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        category_feature(vocab_size=10)
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    output_feature_name = get_output_feature_name(exp_dir_name)
    prediction = exp_dir_name + '/{}_predictions.csv'.format(output_feature_name)
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
                    '--output_feature_name',
                    output_feature_name,
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
        result = subprocess.run(command)
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


def test_visualization_compare_classifiers_subset_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        category_feature(vocab_size=10)
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(output_feature_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'compare_classifiers_performance_subset',
                    '--output_feature_name',
                    output_feature_name,
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
        result = subprocess.run(command)
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


def test_visualization_compare_classifiers_changing_k_output_pdf(csv_filename):
    """It should be possible to save figures as pdf in the specified directory.

    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        category_feature(vocab_size=10)
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(output_feature_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'compare_classifiers_performance_changing_k',
                    '--output_feature_name',
                    output_feature_name,
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
        result = subprocess.run(command)
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


def test_visualization_compare_classifiers_multiclass_multimetric_output_saved(
        csv_filename
):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        category_feature(vocab_size=10)
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    output_feature_name = get_output_feature_name(exp_dir_name)
    test_stats = exp_dir_name + '/test_statistics.json'
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth_metadata = experiment_source_data_name + '.json'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'compare_classifiers_multiclass_multimetric',
                    '--output_feature_name',
                    output_feature_name,
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
        result = subprocess.run(command)
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


def test_visualization_compare_classifiers_predictions_npy_output_saved(
        csv_filename
):
    """Ensure pdf and png figures from the experiments can be saved.

    Predictions are loaded form npy file.
    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        category_feature(vocab_size=10)
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    output_feature_name = get_output_feature_name(exp_dir_name)
    prediction = exp_dir_name + '/{}_predictions.npy'.format(output_feature_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'compare_classifiers_predictions',
                    '--ground_truth',
                    ground_truth,
                    '--output_feature_name',
                    output_feature_name,
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
        result = subprocess.run(command)
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


def test_visualization_compare_classifiers_predictions_csv_output_saved(
        csv_filename
):
    """Ensure pdf and png figures from the experiments can be saved.

    Predictions are loaded form csv file.
    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        category_feature(vocab_size=10)
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    output_feature_name = get_output_feature_name(exp_dir_name)
    prediction = exp_dir_name + '/{}_predictions.csv'.format(output_feature_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'compare_classifiers_predictions',
                    '--ground_truth',
                    ground_truth,
                    '--output_feature_name',
                    output_feature_name,
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
        result = subprocess.run(command)
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

def test_visualization_cmp_classifiers_predictions_distribution_output_saved(
        csv_filename
):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        category_feature(vocab_size=10)
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    output_feature_name = get_output_feature_name(exp_dir_name)
    prediction = exp_dir_name + '/{}_predictions.npy'.format(output_feature_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'compare_classifiers_predictions_distribution',
                    '--ground_truth',
                    ground_truth,
                    '--output_feature_name',
                    output_feature_name,
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
        result = subprocess.run(command)
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


def test_visualization_cconfidence_thresholding_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        category_feature(vocab_size=10)
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(output_feature_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'confidence_thresholding',
                    '--ground_truth',
                    ground_truth,
                    '--output_feature_name',
                    output_feature_name,
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
        result = subprocess.run(command)
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


def test_visualization_confidence_thresholding_data_vs_acc_output_saved(
        csv_filename
):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        category_feature(vocab_size=10)
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(output_feature_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'confidence_thresholding_data_vs_acc',
                    '--ground_truth',
                    ground_truth,
                    '--output_feature_name',
                    output_feature_name,
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
        result = subprocess.run(command)
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


def test_visualization_confidence_thresholding_data_vs_acc_subset_output_saved(
        csv_filename
):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        category_feature(vocab_size=10)
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(output_feature_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'confidence_thresholding_data_vs_acc_subset',
                    '--ground_truth',
                    ground_truth,
                    '--output_feature_name',
                    output_feature_name,
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
        result = subprocess.run(command)
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

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        category_feature(vocab_size=10)
    ]
    output_features = [category_feature(vocab_size=5, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(output_feature_name)
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
                    '--output_feature_name',
                    output_feature_name,
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
        result = subprocess.run(command)
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

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, encoder='stacked_cnn'),
        numerical_feature(),
        category_feature(vocab_size=10, embedding_size=5),
        set_feature(),
        sequence_feature(vocab_size=10, max_len=10, encoder='embed')
    ]
    output_features = [
        category_feature(vocab_size=2, reduce_input='sum'),
        category_feature(vocab_size=2, reduce_input='sum')
    ]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    treshhold_output_feature_name1 = get_output_feature_name(exp_dir_name)
    treshhold_output_feature_name2 = get_output_feature_name(exp_dir_name, output_feature=1)
    probability1 = exp_dir_name + '/{}_probabilities.npy'.format(
        treshhold_output_feature_name1
    )
    probability2 = exp_dir_name + '/{}_probabilities.npy'.format(
        treshhold_output_feature_name2
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
                    probability1,
                    probability2,
                    '--threshold_output_feature_names',
                    treshhold_output_feature_name1,
                    treshhold_output_feature_name2,
                    '--model_names',
                    'Model1',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
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

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, encoder='stacked_cnn'),
        numerical_feature(),
        category_feature(vocab_size=10, embedding_size=5),
        set_feature(),
        sequence_feature(vocab_size=10, max_len=10, encoder='embed')
    ]
    output_features = [
        category_feature(vocab_size=2, reduce_input='sum'),
        category_feature(vocab_size=2, reduce_input='sum')
    ]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    treshhold_output_feature_name1 = get_output_feature_name(exp_dir_name)
    treshhold_output_feature_name2 = get_output_feature_name(exp_dir_name, output_feature=1)
    probability1 = exp_dir_name + '/{}_probabilities.npy'.format(
        treshhold_output_feature_name1
    )
    probability2 = exp_dir_name + '/{}_probabilities.npy'.format(
        treshhold_output_feature_name2
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
                    probability1,
                    probability2,
                    '--threshold_output_feature_names',
                    treshhold_output_feature_name1,
                    treshhold_output_feature_name2,
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
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


def test_visualization_binary_threshold_vs_metric_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, encoder='stacked_cnn'),
        numerical_feature(),
        category_feature(vocab_size=10, embedding_size=5),
        set_feature(),
        sequence_feature(vocab_size=10, max_len=10, encoder='embed')
    ]
    output_features = [
        category_feature(vocab_size=4, reduce_input='sum')
    ]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(output_feature_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
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
                    '--output_feature_name',
                    output_feature_name,
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
        result = subprocess.run(command)
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


def test_visualization_roc_curves_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        category_feature(vocab_size=10)
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(output_feature_name)
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
                    '--output_feature_name',
                    output_feature_name,
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
        result = subprocess.run(command)
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


def test_visualization_roc_curves_from_test_statistics_output_saved(
        csv_filename
):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [binary_feature(), bag_feature()]
    output_features = [binary_feature()]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    output_feature_name = get_output_feature_name(exp_dir_name)
    test_stats = exp_dir_name + '/test_statistics.json'
    experiment_source_data_name = csv_filename.split('.')[0]
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'roc_curves_from_test_statistics',
                    '--output_feature_name',
                    output_feature_name,
                    '--test_statistics',
                    test_stats,
                    '--model_names',
                    'Model1',
                    '-od', exp_dir_name]
    test_cmd_png = test_cmd_pdf.copy() + ['-ff', 'png']

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
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


def test_visualization_calibration_1_vs_all_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        category_feature(vocab_size=10)
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(output_feature_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'calibration_1_vs_all',
                    '--metrics',
                    'accuracy',
                    '--ground_truth',
                    ground_truth,
                    '--output_feature_name',
                    output_feature_name,
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
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 7 == len(figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualization_calibration_multiclass_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        category_feature(vocab_size=10)
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(output_feature_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd_pdf = ['python',
                    '-m',
                    'ludwig.visualize',
                    '--visualization',
                    'calibration_multiclass',
                    '--ground_truth',
                    ground_truth,
                    '--output_feature_name',
                    output_feature_name,
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
        result = subprocess.run(command)
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


def test_visualization_frequency_vs_f1_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        category_feature(vocab_size=10)
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    vis_output_pattern_pdf = exp_dir_name + '/*.pdf'
    vis_output_pattern_png = exp_dir_name + '/*.png'
    output_feature_name = get_output_feature_name(exp_dir_name)
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
                    '--output_feature_name',
                    output_feature_name,
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
        result = subprocess.run(command)
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

def test_load_ground_truth_split_from_file(csv_filename):
    """Ensure correct ground truth split is loaded when ground_truth_split is given.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [
        text_feature(vocab_size=10, min_len=1, representation='sparse'),
        category_feature(vocab_size=10)
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = 'parallel_cnn'
    exp_dir_name = run_experiment(
        input_features,
        output_features,
        data_csv=rel_path
    )
    output_feature_name = get_output_feature_name(exp_dir_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'

    ground_truth_train_split = load_from_file(ground_truth, output_feature_name,
                                              ground_truth_split=0)
    ground_truth_val_split = load_from_file(ground_truth, output_feature_name,
                                              ground_truth_split=1)
    ground_truth_test_split = load_from_file(ground_truth, output_feature_name)

    test_df, train_df, val_df = obtain_df_splits(csv_filename)
    target_predictions_from_train = train_df[output_feature_name]
    target_predictions_from_val = val_df[output_feature_name]
    target_predictions_from_test = test_df[output_feature_name]
    gtm_name = experiment_source_data_name + '.json'
    ground_truth_metadata = load_json(gtm_name)
    ground_truth_loaded_train_split = np.asarray([
        ground_truth_metadata[output_feature_name]['str2idx'][train_row]
        for train_row in target_predictions_from_train
    ])
    ground_truth_loaded_val_split = np.asarray([
        ground_truth_metadata[output_feature_name]['str2idx'][val_row]
        for val_row in target_predictions_from_val
    ])
    ground_truth_loaded_test_split = np.asarray([
        ground_truth_metadata[output_feature_name]['str2idx'][test_row]
        for test_row in target_predictions_from_test
    ])

    assert str(ground_truth_train_split) == str(ground_truth_loaded_train_split)
    assert str(ground_truth_val_split) == str(ground_truth_loaded_val_split)
    assert str(ground_truth_test_split) == str(ground_truth_loaded_test_split)
