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
import shutil
import subprocess
import glob
import json

from ludwig.experiment import experiment
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import text_feature, categorical_feature

# The following imports are pytest fixtures, required for running the tests
from tests.fixtures.filenames import *


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


def test_visualisation_learning_curves_output_pdf(csv_filename):
    """It should be possible to save figures as pdf in the specified directory.

    """
    input_features = [text_feature(reduce_output=None, encoder='rnn')]
    output_features = [text_feature(reduce_input=None, decoder='tagger')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)

    vis_output_pattern = exp_dir_name + '/*.pdf'
    train_stats = exp_dir_name + '/training_statistics.json'
    test_cmd = ['python', '-m', 'ludwig.visualize', '--visualization',
                'learning_curves', '--training_statistics', train_stats,
                '-od', exp_dir_name]

    result = subprocess.run(
        test_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    pdf_figure_cnt = glob.glob(vis_output_pattern)

    assert 0 == result.returncode
    assert 5 == len(pdf_figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)


def test_visualisation_learning_curves_output_png(csv_filename):
    """It should be possible to save figures as png in the specified directory.

    """
    input_features = [text_feature(reduce_output=None, encoder='rnn')]
    output_features = [text_feature(reduce_input=None, decoder='tagger')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)

    vis_output_pattern = exp_dir_name + '/*.png'
    train_stats = exp_dir_name + '/training_statistics.json'
    test_cmd = ['python', '-m', 'ludwig.visualize', '--visualization',
                'learning_curves', '--training_statistics', train_stats,
                '-od', exp_dir_name, '-ff', 'png']
    result = subprocess.run(
        test_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    png_figure_cnt = glob.glob(vis_output_pattern)

    assert 0 == result.returncode
    assert 5 == len(png_figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)


def test_visualisation_confusion_matrix_output_pdf(csv_filename):
    """It should be possible to save figures as pdf in the specified directory.

    """
    input_features = [text_feature(reduce_output=None, encoder='rnn')]
    output_features = [text_feature(reduce_input=None, decoder='tagger')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern = exp_dir_name + '/*.pdf'
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth_metadata = experiment_source_data_name + '.json'
    test_stats = exp_dir_name + '/test_statistics.json'
    test_cmd = ['python', '-m', 'ludwig.visualize', '--visualization',
                'confusion_matrix', '--test_statistics', test_stats,
                '--ground_truth_metadata', ground_truth_metadata,
                '-od', exp_dir_name]

    result = subprocess.run(
        test_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    pdf_figure_cnt = glob.glob(vis_output_pattern)

    assert 0 == result.returncode
    assert 2 == len(pdf_figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_confusion_matrix_output_png(csv_filename):
    """It should be possible to save figures as png in the specified directory.

    """
    input_features = [text_feature(reduce_output=None, encoder='rnn')]
    output_features = [text_feature(reduce_input=None, decoder='tagger')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern = exp_dir_name + '/*.png'
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth_metadata = experiment_source_data_name + '.json'
    test_stats = exp_dir_name + '/test_statistics.json'
    test_cmd = ['python', '-m', 'ludwig.visualize', '--visualization',
                'confusion_matrix', '--test_statistics', test_stats,
                '--ground_truth_metadata', ground_truth_metadata,
                '-od', exp_dir_name, '-ff', 'png']

    result = subprocess.run(
        test_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    png_figure_cnt = glob.glob(vis_output_pattern)

    assert 0 == result.returncode
    assert 2 == len(png_figure_cnt)

    # clean up experiment files.
    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_compare_performance_output_pdf(csv_filename):
    """It should be possible to save figures as pdf in the specified directory.

    Compare performance between two models. To reduce test complexity
    one model is compared to it self.
    """
    input_features = [text_feature(reduce_output=None, encoder='rnn')]
    output_features = [text_feature(reduce_input=None, decoder='tagger')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern = exp_dir_name + '/*.pdf'
    experiment_source_data_name = csv_filename.split('.')[0]
    test_stats = exp_dir_name + '/test_statistics.json'

    test_cmd = ['python',
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

    result = subprocess.run(
        test_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    pdf_figure_cnt = glob.glob(vis_output_pattern)

    assert 0 == result.returncode
    assert 2 == len(pdf_figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_compare_performance_output_png(csv_filename):
    """It should be possible to save figures as png in the specified directory.

    Compare performance between two models. To reduce test complexity
    one model is compared to it self.
    """
    input_features = [text_feature(reduce_output=None, encoder='rnn')]
    output_features = [text_feature(reduce_input=None, decoder='tagger')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    encoder = 'cnnrnn'
    input_features[0]['encoder'] = encoder
    exp_dir_name = run_experiment(input_features, output_features,
                                  data_csv=rel_path)
    vis_output_pattern = exp_dir_name + '/*.png'
    experiment_source_data_name = csv_filename.split('.')[0]
    test_stats = exp_dir_name + '/test_statistics.json'

    test_cmd = ['python',
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
                '-od', exp_dir_name,
                '-ff', 'png']

    result = subprocess.run(
        test_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    png_figure_cnt = glob.glob(vis_output_pattern)

    assert 0 == result.returncode
    assert 2 == len(png_figure_cnt)

    # clean up experiment files.
    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def get_output_field_name(experiment_dir):
    """Helper function to extract output feature name."""
    description_file = experiment_dir + '/description.json'
    with open(description_file, 'rb') as f:
        content = json.load(f)
    field_name = content['model_definition']['output_features'][0]['name']
    return field_name


def test_visualisation_compare_classifiers_from_prob_output_pdf(csv_filename):
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
    vis_output_pattern = exp_dir_name + '/*.pdf'
    field_name = get_output_field_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd = ['python',
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
    result = subprocess.run(
        test_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    pdf_figure_cnt = glob.glob(vis_output_pattern)

    assert 0 == result.returncode
    assert 1 == len(pdf_figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_compare_classifiers_from_prob_output_png(csv_filename):
    """It should be possible to save figures as png in the specified directory.

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
    vis_output_pattern = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd = ['python',
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
                '-od', exp_dir_name,
                '-ff', 'png']
    result = subprocess.run(
        test_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    png_figure_cnt = glob.glob(vis_output_pattern)

    assert 0 == result.returncode
    assert 1 == len(png_figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_compare_classifiers_from_pred_output_pdf(csv_filename):
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
    vis_output_pattern = exp_dir_name + '/*.pdf'
    field_name = get_output_field_name(exp_dir_name)
    prediction = exp_dir_name + '/{}_predictions.csv'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    ground_truth_metadata = experiment_source_data_name + '.json'
    test_cmd = ['python',
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
    result = subprocess.run(
        test_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    pdf_figure_cnt = glob.glob(vis_output_pattern)

    assert 0 == result.returncode
    assert 1 == len(pdf_figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_compare_classifiers_from_pred_output_png(csv_filename):
    """It should be possible to save figures as png in the specified directory.

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
    vis_output_pattern = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    prediction = exp_dir_name + '/{}_predictions.csv'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    ground_truth_metadata = experiment_source_data_name + '.json'
    test_cmd = ['python',
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
                '-od', exp_dir_name,
                '-ff', 'png']

    result = subprocess.run(
        test_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    png_figure_cnt = glob.glob(vis_output_pattern)

    assert 0 == result.returncode
    assert 1 == len(png_figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_compare_classifiers_subset_output_pdf(csv_filename):
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
    vis_output_pattern = exp_dir_name + '/*.pdf'
    field_name = get_output_field_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd = ['python',
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

    result = subprocess.run(
        test_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    pdf_figure_cnt = glob.glob(vis_output_pattern)

    assert 0 == result.returncode
    assert 1 == len(pdf_figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_compare_classifiers_subset_output_png(csv_filename):
    """It should be possible to save figures as png in the specified directory.

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
    vis_output_pattern = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd = ['python',
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
                '-od', exp_dir_name,
                '-ff', 'png']

    result = subprocess.run(
        test_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    png_figure_cnt = glob.glob(vis_output_pattern)

    assert 0 == result.returncode
    assert 1 == len(png_figure_cnt)

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
    vis_output_pattern = exp_dir_name + '/*.pdf'
    field_name = get_output_field_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'

    test_cmd = ['python',
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

    result = subprocess.run(
        test_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    pdf_figure_cnt = glob.glob(vis_output_pattern)

    assert 0 == result.returncode
    assert 1 == len(pdf_figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_compare_classifiers_changing_k_output_png(csv_filename):
    """It should be possible to save figures as png in the specified directory.

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
    vis_output_pattern = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    probability = exp_dir_name + '/{}_probabilities.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    test_cmd = ['python',
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
                '-od', exp_dir_name,
                '-ff', 'png']

    result = subprocess.run(
        test_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    png_figure_cnt = glob.glob(vis_output_pattern)

    assert 0 == result.returncode
    assert 1 == len(png_figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_compare_classifiers_multiclass_multimetric_output_pdf(
        csv_filename
):
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
    vis_output_pattern = exp_dir_name + '/*.pdf'
    test_stats = exp_dir_name + '/test_statistics.json'
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth_metadata = experiment_source_data_name + '.json'
    test_cmd = ['python',
                '-m',
                'ludwig.visualize',
                '--visualization',
                'compare_classifiers_multiclass_multimetric',
                '--test_statistics',
                test_stats,
                test_stats,
                '--ground_truth_metadata',
                ground_truth_metadata,
                '-od', exp_dir_name]

    result = subprocess.run(
        test_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    pdf_figure_cnt = glob.glob(vis_output_pattern)

    assert 0 == result.returncode
    assert 4 == len(pdf_figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_compare_classifiers_multiclass_multimetric_output_png(
        csv_filename
):
    """It should be possible to save figures as png in the specified directory.

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
    vis_output_pattern = exp_dir_name + '/*.png'
    test_stats = exp_dir_name + '/test_statistics.json'
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth_metadata = experiment_source_data_name + '.json'
    test_cmd = ['python',
                '-m',
                'ludwig.visualize',
                '--visualization',
                'compare_classifiers_multiclass_multimetric',
                '--test_statistics',
                test_stats,
                test_stats,
                '--ground_truth_metadata',
                ground_truth_metadata,
                '-od', exp_dir_name,
                '-ff', 'png']

    result = subprocess.run(
        test_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    png_figure_cnt = glob.glob(vis_output_pattern)

    assert 0 == result.returncode
    assert 4 == len(png_figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_compare_classifiers_predictions_output_pdf(csv_filename):
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
    vis_output_pattern = exp_dir_name + '/*.pdf'
    field_name = get_output_field_name(exp_dir_name)
    prediction = exp_dir_name + '/{}_predictions.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    ground_truth_metadata = experiment_source_data_name + '.json'
    test_cmd = ['python',
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
    result = subprocess.run(
        test_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    pdf_figure_cnt = glob.glob(vis_output_pattern)

    assert 0 == result.returncode
    assert 1 == len(pdf_figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))


def test_visualisation_compare_classifiers_predictions_output_png(csv_filename):
    """It should be possible to save figures as png in the specified directory.

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
    vis_output_pattern = exp_dir_name + '/*.png'
    field_name = get_output_field_name(exp_dir_name)
    prediction = exp_dir_name + '/{}_predictions.npy'.format(field_name)
    experiment_source_data_name = csv_filename.split('.')[0]
    ground_truth = experiment_source_data_name + '.hdf5'
    ground_truth_metadata = experiment_source_data_name + '.json'
    test_cmd = ['python',
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
                '-od', exp_dir_name,
                '-ff', 'png']
    result = subprocess.run(
        test_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    png_figure_cnt = glob.glob(vis_output_pattern)

    assert 0 == result.returncode
    assert 1 == len(png_figure_cnt)

    shutil.rmtree(exp_dir_name, ignore_errors=True)
    shutil.rmtree('results', ignore_errors=True)
    for file in glob.glob(experiment_source_data_name + '.*'):
        try:
            os.remove(file)
        except OSError as e:  # if failed, report it back to the user
            print("Error: %s - %s." % (e.filename, e.strerror))

if __name__ == '__main__':
    """
    To run tests individually, run:
    ```python -m pytest tests/integration_tests/test_visualisations.py::test_name```
    """
    pass
