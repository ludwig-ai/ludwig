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
import random
import subprocess

import numpy as np
import pytest

from ludwig.constants import BATCH_SIZE, ENCODER, TRAINER, TYPE
from ludwig.experiment import experiment_cli
from ludwig.globals import DESCRIPTION_FILE_NAME, PREDICTIONS_PARQUET_FILE_NAME, TEST_STATISTICS_FILE_NAME
from ludwig.utils.data_utils import get_split_path
from ludwig.visualize import _extract_ground_truth_values
from tests.integration_tests.test_visualization_api import obtain_df_splits
from tests.integration_tests.utils import (
    bag_feature,
    binary_feature,
    category_feature,
    generate_data,
    number_feature,
    sequence_feature,
    set_feature,
    text_feature,
)


def run_experiment_with_visualization(input_features, output_features, dataset):
    """Helper method to run an experiment with visualization enabled.

    Does not garbage collect.
    """
    output_directory = os.path.dirname(dataset)
    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }

    args = {
        "config": config,
        "skip_save_processed_input": False,
        "skip_save_progress": False,
        "skip_save_unprocessed_output": False,
        "skip_save_eval_stats": False,
        "dataset": dataset,
        "output_directory": output_directory,
    }

    _, _, _, _, experiment_dir = experiment_cli(**args)

    return experiment_dir


def get_output_feature_name(experiment_dir, output_feature=0):
    """Helper function to extract specified output feature name.

    :param experiment_dir: Path to the experiment directory
    :param output_feature: position of the output feature the description.json
    :return output_feature_name: name of the first output feature name
                        from the experiment
    """
    description_file = os.path.join(experiment_dir, DESCRIPTION_FILE_NAME)
    with open(description_file, "rb") as f:
        content = json.load(f)
    output_feature_name = content["config"]["output_features"][output_feature]["name"]
    return output_feature_name


def test_visualization_learning_curves_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [text_feature(encoder={"type": "parallel_cnn"})]
    output_features = [category_feature(output_feature=True)]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0][ENCODER][TYPE] = "parallel_cnn"
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)

    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    train_stats = os.path.join(exp_dir_name, "training_statistics.json")
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "learning_curves",
        "--training_statistics",
        train_stats,
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 4 == len(figure_cnt)


def test_visualization_confusion_matrix_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [text_feature(encoder={"type": "parallel_cnn"})]
    output_features = [category_feature(output_feature=True)]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0][ENCODER][TYPE] = "parallel_cnn"
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth_metadata = experiment_source_data_name + ".meta.json"
    test_stats = os.path.join(exp_dir_name, TEST_STATISTICS_FILE_NAME)
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "confusion_matrix",
        "--test_statistics",
        test_stats,
        "--ground_truth_metadata",
        ground_truth_metadata,
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]
    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 2 == len(figure_cnt)


def test_visualization_compare_performance_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    Compare performance between two models. To reduce test complexity
    one model is compared to it self.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [text_feature(encoder={"type": "parallel_cnn"})]
    output_features = [category_feature(output_feature=True)]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0][ENCODER][TYPE] = "parallel_cnn"
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    test_stats = os.path.join(exp_dir_name, TEST_STATISTICS_FILE_NAME)

    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "compare_performance",
        "--test_statistics",
        test_stats,
        test_stats,
        "-m",
        "Model1",
        "Model2",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)


def test_visualization_compare_classifiers_from_prob_csv_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    Probabilities are loaded from csv file.
    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [category_feature(encoder={"vocab_size": 10})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)

    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = os.path.join(exp_dir_name, PREDICTIONS_PARQUET_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = get_split_path(csv_filename)
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "compare_classifiers_performance_from_prob",
        "--ground_truth",
        ground_truth,
        "--output_feature_name",
        output_feature_name,
        "--split_file",
        split_file,
        "--ground_truth_metadata",
        exp_dir_name + "/model/training_set_metadata.json",
        "--probabilities",
        probability,
        probability,
        "--model_names",
        "Model1",
        "Model2",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)


def test_visualization_compare_classifiers_from_prob_npy_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    Probabilities are loaded from npy file.
    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [category_feature(encoder={"vocab_size": 10})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)

    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = os.path.join(exp_dir_name, PREDICTIONS_PARQUET_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = experiment_source_data_name + ".split.parquet"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "compare_classifiers_performance_from_prob",
        "--ground_truth",
        ground_truth,
        "--output_feature_name",
        output_feature_name,
        "--split_file",
        split_file,
        "--ground_truth_metadata",
        exp_dir_name + "/model/training_set_metadata.json",
        "--probabilities",
        probability,
        probability,
        "--model_names",
        "Model1",
        "Model2",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)


def test_visualization_compare_classifiers_from_pred_npy_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    Predictions are loaded from npy file.
    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [category_feature(encoder={"vocab_size": 10})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    prediction = os.path.join(exp_dir_name, PREDICTIONS_PARQUET_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = experiment_source_data_name + ".split.parquet"
    ground_truth_metadata = experiment_source_data_name + ".meta.json"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "compare_classifiers_performance_from_pred",
        "--ground_truth_metadata",
        ground_truth_metadata,
        "--ground_truth",
        ground_truth,
        "--output_feature_name",
        output_feature_name,
        "--split_file",
        split_file,
        "--predictions",
        prediction,
        prediction,
        "--model_names",
        "Model1",
        "Model2",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)


def test_visualization_compare_classifiers_from_pred_csv_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    Predictions are loaded from csv file.
    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [category_feature(encoder={"vocab_size": 10})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    prediction = os.path.join(exp_dir_name, PREDICTIONS_PARQUET_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = experiment_source_data_name + ".split.parquet"
    ground_truth_metadata = experiment_source_data_name + ".meta.json"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "compare_classifiers_performance_from_pred",
        "--ground_truth_metadata",
        ground_truth_metadata,
        "--ground_truth",
        ground_truth,
        "--output_feature_name",
        output_feature_name,
        "--split_file",
        split_file,
        "--predictions",
        prediction,
        prediction,
        "--model_names",
        "Model1",
        "Model2",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)


def test_visualization_compare_classifiers_subset_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [category_feature(encoder={"vocab_size": 10})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = os.path.join(exp_dir_name, PREDICTIONS_PARQUET_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = experiment_source_data_name + ".split.parquet"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "compare_classifiers_performance_subset",
        "--output_feature_name",
        output_feature_name,
        "--split_file",
        split_file,
        "--ground_truth_metadata",
        exp_dir_name + "/model/training_set_metadata.json",
        "--probabilities",
        probability,
        probability,
        "--model_names",
        "Model1",
        "Model2",
        "--ground_truth",
        ground_truth,
        "--top_n_classes",
        "6",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)


def test_visualization_compare_classifiers_changing_k_output_pdf(csv_filename):
    """It should be possible to save figures as pdf in the specified directory."""
    input_features = [category_feature(encoder={"vocab_size": 10})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = os.path.join(exp_dir_name, PREDICTIONS_PARQUET_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = experiment_source_data_name + ".split.parquet"
    ground_truth_metadata = exp_dir_name + "/model/training_set_metadata.json"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "compare_classifiers_performance_changing_k",
        "--output_feature_name",
        output_feature_name,
        "--split_file",
        split_file,
        "--ground_truth_metadata",
        ground_truth_metadata,
        "--probabilities",
        probability,
        probability,
        "--model_names",
        "Model1",
        "Model2",
        "--ground_truth",
        ground_truth,
        "--top_n_classes",
        "6",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]
    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)


def test_visualization_compare_classifiers_multiclass_multimetric_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [category_feature(encoder={"vocab_size": 10})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    test_stats = os.path.join(exp_dir_name, TEST_STATISTICS_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth_metadata = experiment_source_data_name + ".meta.json"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "compare_classifiers_multiclass_multimetric",
        "--output_feature_name",
        output_feature_name,
        "--test_statistics",
        test_stats,
        test_stats,
        "--ground_truth_metadata",
        ground_truth_metadata,
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 4 == len(figure_cnt)


def test_visualization_compare_classifiers_predictions_npy_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    Predictions are loaded form npy file.
    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [category_feature(encoder={"vocab_size": 10})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    prediction = os.path.join(exp_dir_name, PREDICTIONS_PARQUET_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = experiment_source_data_name + ".split.parquet"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "compare_classifiers_predictions",
        "--ground_truth",
        ground_truth,
        "--output_feature_name",
        output_feature_name,
        "--split_file",
        split_file,
        "--ground_truth_metadata",
        exp_dir_name + "/model/training_set_metadata.json",
        "--predictions",
        prediction,
        prediction,
        "--model_names",
        "Model1",
        "Model2",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)


def test_visualization_compare_classifiers_predictions_csv_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    Predictions are loaded form csv file.
    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [category_feature(encoder={"vocab_size": 10})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    prediction = os.path.join(exp_dir_name, PREDICTIONS_PARQUET_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = experiment_source_data_name + ".split.parquet"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "compare_classifiers_predictions",
        "--ground_truth",
        ground_truth,
        "--output_feature_name",
        output_feature_name,
        "--split_file",
        split_file,
        "--ground_truth_metadata",
        exp_dir_name + "/model/training_set_metadata.json",
        "--predictions",
        prediction,
        prediction,
        "--model_names",
        "Model1",
        "Model2",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)


def test_visualization_cmp_classifiers_predictions_distribution_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [category_feature(encoder={"vocab_size": 10})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    prediction = os.path.join(exp_dir_name, PREDICTIONS_PARQUET_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = experiment_source_data_name + ".split.parquet"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "compare_classifiers_predictions_distribution",
        "--ground_truth",
        ground_truth,
        "--output_feature_name",
        output_feature_name,
        "--split_file",
        split_file,
        "--ground_truth_metadata",
        exp_dir_name + "/model/training_set_metadata.json",
        "--predictions",
        prediction,
        prediction,
        "--model_names",
        "Model1",
        "Model2",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)


def test_visualization_cconfidence_thresholding_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [category_feature(encoder={"vocab_size": 10})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = os.path.join(exp_dir_name, PREDICTIONS_PARQUET_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = experiment_source_data_name + ".split.parquet"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "confidence_thresholding",
        "--ground_truth",
        ground_truth,
        "--output_feature_name",
        output_feature_name,
        "--split_file",
        split_file,
        "--ground_truth_metadata",
        exp_dir_name + "/model/training_set_metadata.json",
        "--probabilities",
        probability,
        probability,
        "--model_names",
        "Model1",
        "Model2",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)


def test_visualization_confidence_thresholding_data_vs_acc_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [category_feature(encoder={"vocab_size": 10})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = os.path.join(exp_dir_name, PREDICTIONS_PARQUET_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = experiment_source_data_name + ".split.parquet"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "confidence_thresholding_data_vs_acc",
        "--ground_truth",
        ground_truth,
        "--output_feature_name",
        output_feature_name,
        "--split_file",
        split_file,
        "--ground_truth_metadata",
        exp_dir_name + "/model/training_set_metadata.json",
        "--probabilities",
        probability,
        probability,
        "--model_names",
        "Model1",
        "Model2",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)


def test_visualization_confidence_thresholding_data_vs_acc_subset_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [category_feature(encoder={"vocab_size": 10})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = os.path.join(exp_dir_name, PREDICTIONS_PARQUET_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = experiment_source_data_name + ".split.parquet"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "confidence_thresholding_data_vs_acc_subset",
        "--ground_truth",
        ground_truth,
        "--output_feature_name",
        output_feature_name,
        "--split_file",
        split_file,
        "--ground_truth_metadata",
        exp_dir_name + "/model/training_set_metadata.json",
        "--probabilities",
        probability,
        probability,
        "--model_names",
        "Model1",
        "Model2",
        "--top_n_classes",
        "3",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)


def test_vis_confidence_thresholding_data_vs_acc_subset_per_class_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [category_feature(encoder={"vocab_size": 10})]
    output_features = [category_feature(decoder={"vocab_size": 5}, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = os.path.join(exp_dir_name, PREDICTIONS_PARQUET_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = experiment_source_data_name + ".split.parquet"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "confidence_thresholding_data_vs_acc_subset_per_class",
        "--ground_truth",
        ground_truth,
        "--output_feature_name",
        output_feature_name,
        "--split_file",
        split_file,
        "--ground_truth_metadata",
        exp_dir_name + "/model/training_set_metadata.json",
        "--probabilities",
        probability,
        probability,
        "--model_names",
        "Model1",
        "Model2",
        "--top_n_classes",
        "3",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        # 3 figures should be saved because experiment setting top_n_classes = 3
        # hence one figure per class
        assert 3 == len(figure_cnt)


def test_vis_confidence_thresholding_2thresholds_2d_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [
        text_feature(encoder={"vocab_size": 10, "min_len": 1, "type": "stacked_cnn"}),
        number_feature(),
        category_feature(encoder={"vocab_size": 10, "embedding_size": 5}),
        set_feature(),
        sequence_feature(encoder={"vocab_size": 10, "max_len": 10, "type": "embed"}),
    ]
    output_features = [
        category_feature(decoder={"vocab_size": 2}, reduce_input="sum"),
        category_feature(decoder={"vocab_size": 2}, reduce_input="sum"),
    ]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0][ENCODER][TYPE] = "parallel_cnn"
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    threshold_output_feature_name1 = get_output_feature_name(exp_dir_name)
    threshold_output_feature_name2 = get_output_feature_name(exp_dir_name, output_feature=1)
    probability = os.path.join(exp_dir_name, PREDICTIONS_PARQUET_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = experiment_source_data_name + ".split.parquet"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "confidence_thresholding_2thresholds_2d",
        "--ground_truth",
        ground_truth,
        "--split_file",
        split_file,
        "--ground_truth_metadata",
        exp_dir_name + "/model/training_set_metadata.json",
        "--probabilities",
        probability,
        "--threshold_output_feature_names",
        threshold_output_feature_name1,
        threshold_output_feature_name2,
        "--model_names",
        "Model1",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 3 == len(figure_cnt)


def test_vis_confidence_thresholding_2thresholds_3d_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [
        text_feature(encoder={"vocab_size": 10, "min_len": 1, "type": "stacked_cnn"}),
        number_feature(),
        category_feature(encoder={"vocab_size": 10, "embedding_size": 5}),
        set_feature(),
        sequence_feature(encoder={"vocab_size": 10, "max_len": 10, "type": "embed"}),
    ]
    output_features = [
        category_feature(decoder={"vocab_size": 2}, reduce_input="sum"),
        category_feature(decoder={"vocab_size": 2}, reduce_input="sum"),
    ]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0][ENCODER][TYPE] = "parallel_cnn"
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    threshold_output_feature_name1 = get_output_feature_name(exp_dir_name)
    threshold_output_feature_name2 = get_output_feature_name(exp_dir_name, output_feature=1)
    probability = os.path.join(exp_dir_name, PREDICTIONS_PARQUET_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = experiment_source_data_name + ".split.parquet"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "confidence_thresholding_2thresholds_3d",
        "--ground_truth",
        ground_truth,
        "--split_file",
        split_file,
        "--ground_truth_metadata",
        exp_dir_name + "/model/training_set_metadata.json",
        "--probabilities",
        probability,
        "--threshold_output_feature_names",
        threshold_output_feature_name1,
        threshold_output_feature_name2,
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(
            command,
        )
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)


@pytest.mark.parametrize("binary_output_type", [True, False])
def test_visualization_binary_threshold_vs_metric_output_saved(csv_filename, binary_output_type):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [
        text_feature(encoder={"vocab_size": 10, "min_len": 1, "type": "stacked_cnn"}),
        number_feature(),
        category_feature(encoder={"vocab_size": 10, "embedding_size": 5}),
        set_feature(),
        sequence_feature(encoder={"vocab_size": 10, "max_len": 10, "type": "embed"}),
    ]
    if binary_output_type:
        output_features = [binary_feature()]
    else:
        output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    random.seed(1919)
    rel_path = generate_data(input_features, output_features, csv_filename)
    input_features[0][ENCODER][TYPE] = "parallel_cnn"
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = os.path.join(exp_dir_name, PREDICTIONS_PARQUET_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = experiment_source_data_name + ".split.parquet"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "binary_threshold_vs_metric",
        "--positive_label",
        "1",
        "--metrics",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "--ground_truth",
        ground_truth,
        "--output_feature_name",
        output_feature_name,
        "--split_file",
        split_file,
        "--ground_truth_metadata",
        exp_dir_name + "/model/training_set_metadata.json",
        "--probabilities",
        probability,
        probability,
        "--model_names",
        "Model1",
        "Model2",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 4 == len(figure_cnt)


@pytest.mark.parametrize("binary_output_type", [True, False])
def test_visualization_precision_recall_curves_output_saved(csv_filename, binary_output_type):
    """Ensure pdf and png figures for precision recall curves from the experiments can be saved."""
    input_features = [category_feature(encoder={"vocab_size": 10})]
    if binary_output_type:
        output_features = [binary_feature()]
    else:
        output_features = [category_feature(decoder={"vocab_size": 3}, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename, num_examples=1000)
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = os.path.join(exp_dir_name, PREDICTIONS_PARQUET_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = experiment_source_data_name + ".split.parquet"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "precision_recall_curves",
        "--positive_label",
        "1",
        "--ground_truth",
        ground_truth,
        "--output_feature_name",
        output_feature_name,
        "--split_file",
        split_file,
        "--ground_truth_metadata",
        exp_dir_name + "/model/training_set_metadata.json",
        "--probabilities",
        probability,
        probability,
        "--model_names",
        "Model1",
        "Model2",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)


def test_visualization_precision_recall_curves_from_test_statistics_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [binary_feature(), bag_feature()]
    output_features = [binary_feature()]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename, num_examples=1000)

    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    test_stats = os.path.join(exp_dir_name, TEST_STATISTICS_FILE_NAME)
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "precision_recall_curves_from_test_statistics",
        "--output_feature_name",
        output_feature_name,
        "--test_statistics",
        test_stats,
        "--model_names",
        "Model1",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)


@pytest.mark.parametrize("binary_output_type", [True, False])
def test_visualization_roc_curves_output_saved(csv_filename, binary_output_type):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [category_feature(encoder={"vocab_size": 10})]
    if binary_output_type:
        output_features = [binary_feature()]
    else:
        output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = os.path.join(exp_dir_name, PREDICTIONS_PARQUET_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = experiment_source_data_name + ".split.parquet"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "roc_curves",
        "--positive_label",
        "1",
        "--metrics",
        "accuracy",
        "--ground_truth",
        ground_truth,
        "--output_feature_name",
        output_feature_name,
        "--split_file",
        split_file,
        "--ground_truth_metadata",
        exp_dir_name + "/model/training_set_metadata.json",
        "--probabilities",
        probability,
        probability,
        "--model_names",
        "Model1",
        "Model2",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)


def test_visualization_roc_curves_from_test_statistics_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [binary_feature(), bag_feature()]
    output_features = [binary_feature()]
    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    test_stats = os.path.join(exp_dir_name, TEST_STATISTICS_FILE_NAME)
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "roc_curves_from_test_statistics",
        "--output_feature_name",
        output_feature_name,
        "--test_statistics",
        test_stats,
        "--model_names",
        "Model1",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 1 == len(figure_cnt)


def test_visualization_calibration_1_vs_all_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [category_feature(encoder={"vocab_size": 10})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = os.path.join(exp_dir_name, PREDICTIONS_PARQUET_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = experiment_source_data_name + ".split.parquet"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "calibration_1_vs_all",
        "--metrics",
        "accuracy",
        "--ground_truth",
        ground_truth,
        "--output_feature_name",
        output_feature_name,
        "--split_file",
        split_file,
        "--ground_truth_metadata",
        exp_dir_name + "/model/training_set_metadata.json",
        "--probabilities",
        probability,
        probability,
        "--model_names",
        "Model1",
        "Model2",
        "--top_k",
        "6",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 5 == len(figure_cnt)


def test_visualization_calibration_multiclass_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [category_feature(encoder={"vocab_size": 10})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    probability = os.path.join(exp_dir_name, PREDICTIONS_PARQUET_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = experiment_source_data_name + ".split.parquet"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "calibration_multiclass",
        "--ground_truth",
        ground_truth,
        "--output_feature_name",
        output_feature_name,
        "--split_file",
        split_file,
        "--ground_truth_metadata",
        exp_dir_name + "/model/training_set_metadata.json",
        "--probabilities",
        probability,
        probability,
        "--model_names",
        "Model1",
        "Model2",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 2 == len(figure_cnt)


def test_visualization_frequency_vs_f1_output_saved(csv_filename):
    """Ensure pdf and png figures from the experiments can be saved.

    :param csv_filename: csv fixture from tests.conftest.csv_filename
    :return: None
    """
    input_features = [category_feature(encoder={"vocab_size": 10})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    vis_output_pattern_pdf = os.path.join(exp_dir_name, "*.pdf")
    vis_output_pattern_png = os.path.join(exp_dir_name, "*.png")
    output_feature_name = get_output_feature_name(exp_dir_name)
    test_stats = os.path.join(exp_dir_name, TEST_STATISTICS_FILE_NAME)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth_metadata = experiment_source_data_name + ".meta.json"
    test_cmd_pdf = [
        "python",
        "-m",
        "ludwig.visualize",
        "--visualization",
        "frequency_vs_f1",
        "--ground_truth_metadata",
        ground_truth_metadata,
        "--output_feature_name",
        output_feature_name,
        "--test_statistics",
        test_stats,
        test_stats,
        "--model_names",
        "Model1",
        "Model2",
        "-od",
        exp_dir_name,
    ]
    test_cmd_png = test_cmd_pdf.copy() + ["-ff", "png"]

    commands = [test_cmd_pdf, test_cmd_png]
    vis_patterns = [vis_output_pattern_pdf, vis_output_pattern_png]

    for command, viz_pattern in zip(commands, vis_patterns):
        result = subprocess.run(command)
        figure_cnt = glob.glob(viz_pattern)

        assert 0 == result.returncode
        assert 2 == len(figure_cnt)


def test_load_ground_truth_split_from_df(csv_filename):
    import pandas as pd

    ground_truth = pd.DataFrame(
        {
            "PassengerId": [1],
            "Survived": [0],
            "Pclass": [3],
            "Name": ["Braund, Mr. Owen Harris"],
            "Sex": ["male"],
            "Age": [22.0],
            "SibSp": [1],
            "Parch": [0],
            "Ticket": ["A/5 21171"],
            "Fare": ["7.25"],
            "Cabin": [None],
            "Embarked": ["S"],
            "split": [0],
        }
    )
    output_feature = "Survived"
    ground_truth_train_split = _extract_ground_truth_values(ground_truth, output_feature, 0)
    ground_truth_val_split = _extract_ground_truth_values(ground_truth, output_feature, 1)
    ground_truth_test_split = _extract_ground_truth_values(ground_truth, output_feature, 2)

    assert ground_truth_train_split.equals(pd.Series([0]))
    assert ground_truth_val_split.empty
    assert ground_truth_test_split.empty


def test_load_ground_truth_split_from_file(csv_filename):
    """Ensure correct ground truth split is loaded when ground_truth_split is given.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [category_feature(encoder={"vocab_size": 10})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    exp_dir_name = run_experiment_with_visualization(input_features, output_features, dataset=rel_path)
    output_feature_name = get_output_feature_name(exp_dir_name)
    experiment_source_data_name = csv_filename.split(".")[0]
    ground_truth = experiment_source_data_name + ".csv"
    split_file = experiment_source_data_name + ".split.parquet"

    # retrieve ground truth from source data set
    ground_truth_train_split = _extract_ground_truth_values(ground_truth, output_feature_name, 0, split_file)
    ground_truth_val_split = _extract_ground_truth_values(ground_truth, output_feature_name, 1, split_file)
    ground_truth_test_split = _extract_ground_truth_values(ground_truth, output_feature_name, 2, split_file)

    test_df, train_df, val_df = obtain_df_splits(csv_filename)
    target_predictions_from_train = train_df[output_feature_name]
    target_predictions_from_val = val_df[output_feature_name]
    target_predictions_from_test = test_df[output_feature_name]

    assert np.all(ground_truth_train_split.eq(target_predictions_from_train))
    assert np.all(ground_truth_val_split.eq(target_predictions_from_val))
    assert np.all(ground_truth_test_split.eq(target_predictions_from_test))
