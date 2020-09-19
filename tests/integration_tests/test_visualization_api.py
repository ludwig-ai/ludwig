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
import os
import shutil

import numpy as np
import pandas as pd

from ludwig import visualize
from ludwig.api import LudwigModel
from ludwig.data.preprocessing import get_split
from ludwig.utils.data_utils import read_csv, split_dataset_ttv
from tests.integration_tests.utils import category_feature, \
    numerical_feature, set_feature, generate_data, sequence_feature, \
    text_feature, binary_feature, bag_feature


def run_api_experiment(input_features, output_features):
    """
    Helper method to avoid code repetition in running an experiment
    :param input_features: input schema
    :param output_features: output schema
    :return: None
    """
    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat', 'fc_size': 14},
        'training': {'epochs': 2}
    }

    model = LudwigModel(model_definition)
    return model


class Experiment:
    """Helper class to create model test data, setup and run experiment.

    Contain the needed model experiment statistics as class attributes.
    """

    def __init__(self, csv_filename):
        self.csv_file = csv_filename
        self.input_features = [
            text_feature(vocab_size=10, min_len=1, representation='sparse'),
            category_feature(vocab_size=10)
        ]
        self.output_features = [
            category_feature(vocab_size=2, reduce_input='sum')]
        encoder = 'parallel_cnn'
        data_csv = generate_data(
            self.input_features,
            self.output_features,
            self.csv_file
        )
        self.input_features[0]['encoder'] = encoder
        self.model = self._create_model()
        test_df, train_df, val_df = obtain_df_splits(data_csv)
        self.train_stats, _, self.output_dir = self.model.train(
            training_set=train_df,
            validation_set=val_df
        )
        self.test_stats_full, predictions, self.output_dir = self.model.evaluate(
            dataset=test_df,
            collect_overall_stats=True,
            collect_predictions=True,
            output_directory=self.output_dir
        )
        self.output_feature_name = self.output_features[0]['name']
        # probabilities need to be list of lists containing each row data
        # from the probability columns
        # ref: https://ludwig-ai.github.io/ludwig-docs/api/#test - Return
        num_probs = self.output_features[0]['vocab_size']
        self.probability = predictions.iloc[:, 1:(num_probs + 2)].values
        self.ground_truth_metadata = self.model.training_set_metadata
        target_predictions = test_df[self.output_feature_name]
        self.ground_truth = np.asarray([
            self.ground_truth_metadata[self.output_feature_name]['str2idx'][
                test_row]
            for test_row in target_predictions
        ])
        self.prediction_raw = predictions.iloc[:, 0].tolist()
        self.prediction = np.asarray([
            self.ground_truth_metadata[self.output_feature_name]['str2idx'][
                pred_row]
            for pred_row in self.prediction_raw])

    def _create_model(self):
        """Configure and setup test model"""
        model_definition = {
            'input_features': self.input_features,
            'output_features': self.output_features,
            'combiner': {'type': 'concat', 'fc_size': 14},
            'training': {'epochs': 2}
        }
        return LudwigModel(model_definition)


def obtain_df_splits(data_csv):
    """Split input data csv file in to train, validation and test dataframes.

    :param data_csv: Input data CSV file.
    :return test_df, train_df, val_df: Train, validation and test dataframe
            splits
    """
    data_df = read_csv(data_csv)
    # Obtain data split array mapping data rows to split type
    # 0-train, 1-validation, 2-test
    data_split = get_split(data_df)
    train_split, test_split, val_split = split_dataset_ttv(data_df, data_split)
    # Splits are python dictionaries not dataframes- they need to be converted.
    test_df = pd.DataFrame(test_split)
    train_df = pd.DataFrame(train_split)
    val_df = pd.DataFrame(val_split)
    return test_df, train_df, val_df


def test_learning_curves_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    experiment = Experiment(csv_filename)
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = experiment.output_dir + '/*.{}'.format(
            viz_output)
        visualize.learning_curves(
            experiment.train_stats,
            output_feature_name=None,
            output_directory=experiment.output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 4 == len(figure_cnt)
    shutil.rmtree(experiment.output_dir, ignore_errors=True)


def test_compare_performance_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    experiment = Experiment(csv_filename)
    # extract test stats only
    test_stats = experiment.test_stats_full
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = experiment.output_dir + '/*.{}'.format(
            viz_output
        )
        visualize.compare_performance(
            [test_stats, test_stats],
            output_feature_name=None,
            model_names=['Model1', 'Model2'],
            output_directory=experiment.output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 1 == len(figure_cnt)
    shutil.rmtree(experiment.output_dir, ignore_errors=True)


def test_compare_classifier_performance_from_prob_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    experiment = Experiment(csv_filename)
    probability = experiment.probability
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = experiment.output_dir + '/*.{}'.format(
            viz_output)
        visualize.compare_classifiers_performance_from_prob(
            [probability, probability],
            experiment.ground_truth,
            top_n_classes=[0],
            labels_limit=0,
            model_namess=['Model1', 'Model2'],
            output_directory=experiment.output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 1 == len(figure_cnt)
    shutil.rmtree(experiment.output_dir, ignore_errors=True)


def test_compare_classifier_performance_from_pred_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    experiment = Experiment(csv_filename)
    prediction = experiment.prediction_raw
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = experiment.output_dir + '/*.{}'.format(
            viz_output)
        visualize.compare_classifiers_performance_from_pred(
            [prediction, prediction],
            experiment.ground_truth,
            experiment.ground_truth_metadata,
            experiment.output_feature_name,
            labels_limit=0,
            model_namess=['Model1', 'Model2'],
            output_directory=experiment.output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 1 == len(figure_cnt)
    shutil.rmtree(experiment.output_dir, ignore_errors=True)


def test_compare_classifiers_performance_subset_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    experiment = Experiment(csv_filename)
    probability = experiment.probability
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = experiment.output_dir + '/*.{}'.format(
            viz_output)
        visualize.compare_classifiers_performance_subset(
            [probability, probability],
            experiment.ground_truth,
            top_n_classes=[6],
            labels_limit=0,
            subset='ground_truth',
            model_namess=['Model1', 'Model2'],
            output_directory=experiment.output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 1 == len(figure_cnt)
    shutil.rmtree(experiment.output_dir, ignore_errors=True)


def test_compare_classifiers_performance_changing_k_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    experiment = Experiment(csv_filename)
    probability = experiment.probability
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = experiment.output_dir + '/*.{}'.format(
            viz_output
        )
        visualize.compare_classifiers_performance_changing_k(
            [probability, probability],
            experiment.ground_truth,
            top_k=3,
            labels_limit=0,
            model_namess=['Model1', 'Model2'],
            output_directory=experiment.output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 1 == len(figure_cnt)
    shutil.rmtree(experiment.output_dir, ignore_errors=True)


def test_compare_classifiers_multiclass_multimetric_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    experiment = Experiment(csv_filename)
    # extract test stats only
    test_stats = experiment.test_stats_full
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = experiment.output_dir + '/*.{}'.format(
            viz_output
        )
        visualize.compare_classifiers_multiclass_multimetric(
            [test_stats, test_stats],
            experiment.ground_truth_metadata,
            experiment.output_feature_name,
            top_n_classes=[6],
            model_namess=['Model1', 'Model2'],
            output_directory=experiment.output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 4 == len(figure_cnt)
    shutil.rmtree(experiment.output_dir, ignore_errors=True)


def test_compare_classifiers_predictions_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    experiment = Experiment(csv_filename)
    prediction = experiment.prediction
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = experiment.output_dir + '/*.{}'.format(
            viz_output
        )
        visualize.compare_classifiers_predictions(
            [prediction, prediction],
            experiment.ground_truth,
            labels_limit=0,
            model_names=['Model1', 'Model2'],
            output_directory=experiment.output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 1 == len(figure_cnt)
    shutil.rmtree(experiment.output_dir, ignore_errors=True)


def test_compare_classifiers_predictions_distribution_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    experiment = Experiment(csv_filename)
    prediction = experiment.prediction
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = experiment.output_dir + '/*.{}'.format(
            viz_output
        )
        visualize.compare_classifiers_predictions_distribution(
            [prediction, prediction],
            experiment.ground_truth,
            labels_limit=0,
            model_names=['Model1', 'Model2'],
            output_directory=experiment.output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 1 == len(figure_cnt)
    shutil.rmtree(experiment.output_dir, ignore_errors=True)


def test_confidence_thresholding_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    experiment = Experiment(csv_filename)
    probability = experiment.probability
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = experiment.output_dir + '/*.{}'.format(
            viz_output
        )
        visualize.confidence_thresholding(
            [probability, probability],
            experiment.ground_truth,
            labels_limit=0,
            model_names=['Model1', 'Model2'],
            output_directory=experiment.output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 1 == len(figure_cnt)
    shutil.rmtree(experiment.output_dir, ignore_errors=True)


def test_confidence_thresholding_data_vs_acc_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    experiment = Experiment(csv_filename)
    probability = experiment.probability
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = experiment.output_dir + '/*.{}'.format(
            viz_output
        )
        visualize.confidence_thresholding_data_vs_acc(
            [probability, probability],
            experiment.ground_truth,
            labels_limit=0,
            model_names=['Model1', 'Model2'],
            output_directory=experiment.output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 1 == len(figure_cnt)
    shutil.rmtree(experiment.output_dir, ignore_errors=True)


def test_confidence_thresholding_data_vs_acc_subset_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    experiment = Experiment(csv_filename)
    probability = experiment.probability
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = experiment.output_dir + '/*.{}'.format(
            viz_output
        )
        visualize.confidence_thresholding_data_vs_acc_subset(
            [probability, probability],
            experiment.ground_truth,
            top_n_classes=[3],
            labels_limit=0,
            subset='ground_truth',
            model_names=['Model1', 'Model2'],
            output_directory=experiment.output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 1 == len(figure_cnt)
    shutil.rmtree(experiment.output_dir, ignore_errors=True)


def test_confidence_thresholding_data_vs_acc_subset_per_class_vis_api(
        csv_filename
):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    experiment = Experiment(csv_filename)
    probability = experiment.probability
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = experiment.output_dir + '/*.{}'.format(
            viz_output
        )
        visualize.confidence_thresholding_data_vs_acc_subset_per_class(
            [probability, probability],
            experiment.ground_truth,
            experiment.ground_truth_metadata,
            experiment.output_feature_name,
            top_n_classes=[3],
            labels_limit=0,
            subset='ground_truth',
            model_names=['Model1', 'Model2'],
            output_directory=experiment.output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        # 3 figures should be saved because experiment setting top_n_classes = 3
        # hence one figure per class
        assert 3 == len(figure_cnt)
    shutil.rmtree(experiment.output_dir, ignore_errors=True)


def test_confidence_thresholding_2thresholds_2d_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
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
    encoder = 'parallel_cnn'
    # Generate test data
    data_csv = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = encoder
    model = run_api_experiment(input_features, output_features)
    test_df, train_df, val_df = obtain_df_splits(data_csv)
    _, _, output_dir = model.train(
        training_set=train_df,
        validation_set=val_df
    )
    test_stats, predictions, _ = model.evaluate(
        dataset=test_df,
        collect_predictions=True,
        output_dir=output_dir
    )

    output_feature_name1 = output_features[0]['name']
    output_feature_name2 = output_features[1]['name']
    # probabilities need to be list of lists containing each row data from the
    # probability columns ref: https://ludwig-ai.github.io/ludwig-docs/api/#test - Return
    probability1 = predictions.iloc[:, [2, 3, 4]].values
    probability2 = predictions.iloc[:, [7, 8, 9]].values

    ground_truth_metadata = model.training_set_metadata
    target_predictions1 = test_df[output_feature_name1]
    target_predictions2 = test_df[output_feature_name2]
    ground_truth1 = np.asarray([
        ground_truth_metadata[output_feature_name1]['str2idx'][prediction]
        for prediction in target_predictions1
    ])
    ground_truth2 = np.asarray([
        ground_truth_metadata[output_feature_name2]['str2idx'][prediction]
        for prediction in target_predictions2
    ])
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = os.path.join(output_dir, '*.{}').format(
            viz_output)
        visualize.confidence_thresholding_2thresholds_2d(
            [probability1, probability2],
            [ground_truth1, ground_truth2],
            [output_feature_name1, output_feature_name2],
            labels_limit=0,
            model_names=['Model1'],
            output_directory=output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 3 == len(figure_cnt)
    shutil.rmtree(output_dir, ignore_errors=True)


def test_confidence_thresholding_2thresholds_3d_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
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
    encoder = 'parallel_cnn'
    # Generate test data
    data_csv = generate_data(input_features, output_features, csv_filename)
    input_features[0]['encoder'] = encoder
    model = run_api_experiment(input_features, output_features)
    test_df, train_df, val_df = obtain_df_splits(data_csv)
    _, _, output_dir = model.train(
        training_set=train_df,
        validation_set=val_df
    )
    test_stats, predictions, _ = model.evaluate(
        dataset=test_df,
        collect_predictions=True,
        output_directory=output_dir
    )

    output_feature_name1 = output_features[0]['name']
    output_feature_name2 = output_features[1]['name']
    # probabilities need to be list of lists containing each row data from the
    # probability columns ref: https://ludwig-ai.github.io/ludwig-docs/api/#test - Return
    probability1 = predictions.iloc[:, [2, 3, 4]].values
    probability2 = predictions.iloc[:, [7, 8, 9]].values

    ground_truth_metadata = model.training_set_metadata
    target_predictions1 = test_df[output_feature_name1]
    target_predictions2 = test_df[output_feature_name2]
    ground_truth1 = np.asarray([
        ground_truth_metadata[output_feature_name1]['str2idx'][prediction]
        for prediction in target_predictions1
    ])
    ground_truth2 = np.asarray([
        ground_truth_metadata[output_feature_name2]['str2idx'][prediction]
        for prediction in target_predictions2
    ])
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = os.path.join(
            output_dir, '*.{}'.format(viz_output)
        )
        visualize.confidence_thresholding_2thresholds_3d(
            [probability1, probability2],
            [ground_truth1, ground_truth2],
            [output_feature_name1, output_feature_name2],
            labels_limit=0,
            output_directory=output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 1 == len(figure_cnt)
    shutil.rmtree(output_dir, ignore_errors=True)


def test_binary_threshold_vs_metric_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    experiment = Experiment(csv_filename)
    probability = experiment.probability
    viz_outputs = ('pdf', 'png')
    metrics = ['accuracy']
    positive_label = 2
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = experiment.output_dir + '/*.{}'.format(
            viz_output
        )
        visualize.binary_threshold_vs_metric(
            [probability, probability],
            experiment.ground_truth,
            metrics,
            positive_label,
            model_names=['Model1', 'Model2'],
            output_directory=experiment.output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 1 == len(figure_cnt)
    shutil.rmtree(experiment.output_dir, ignore_errors=True)


def test_roc_curves_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    experiment = Experiment(csv_filename)
    probability = experiment.probability
    viz_outputs = ('pdf', 'png')
    positive_label = 2
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = experiment.output_dir + '/*.{}'.format(
            viz_output
        )
        visualize.roc_curves(
            [probability, probability],
            experiment.ground_truth,
            positive_label,
            model_names=['Model1', 'Model2'],
            output_directory=experiment.output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 1 == len(figure_cnt)
    shutil.rmtree(experiment.output_dir, ignore_errors=True)


def test_roc_curves_from_test_statistics_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    input_features = [binary_feature(), bag_feature()]
    output_features = [binary_feature()]

    # Generate test data
    data_csv = generate_data(input_features, output_features, csv_filename)
    output_feature_name = output_features[0]['name']

    model = run_api_experiment(input_features, output_features)
    data_df = read_csv(data_csv)
    _, _, output_dir = model.train(dataset=data_df)
    # extract test metrics
    test_stats, _, _ = model.evaluate(dataset=data_df,
                                      collect_overall_stats=True,
                                      output_directory=output_dir)
    test_stats = test_stats
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = os.path.join(output_dir, '*.{}'.format(
            viz_output))
        visualize.roc_curves_from_test_statistics(
            [test_stats, test_stats],
            output_feature_name,
            model_names=['Model1', 'Model2'],
            output_directory=output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 1 == len(figure_cnt)
    shutil.rmtree(output_dir, ignore_errors=True)


def test_calibration_1_vs_all_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    experiment = Experiment(csv_filename)
    probability = experiment.probability
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = os.path.join(
            experiment.output_dir, '*.{}'.format(viz_output)
        )
        visualize.calibration_1_vs_all(
            [probability, probability],
            experiment.ground_truth,
            top_n_classes=[6],
            labels_limit=0,
            model_namess=['Model1', 'Model2'],
            output_directory=experiment.output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 7 == len(figure_cnt)
    shutil.rmtree(experiment.output_dir, ignore_errors=True)


def test_calibration_multiclass_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    experiment = Experiment(csv_filename)
    probability = experiment.probability
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = experiment.output_dir + '/*.{}'.format(
            viz_output
        )
        visualize.calibration_multiclass(
            [probability, probability],
            experiment.ground_truth,
            labels_limit=0,
            model_names=['Model1', 'Model2'],
            output_directory=experiment.output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 2 == len(figure_cnt)
    shutil.rmtree(experiment.output_dir, ignore_errors=True)


def test_confusion_matrix_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    experiment = Experiment(csv_filename)
    # extract test stats only
    test_stats = experiment.test_stats_full
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = experiment.output_dir + '/*.{}'.format(
            viz_output
        )
        visualize.confusion_matrix(
            [test_stats, test_stats],
            experiment.ground_truth_metadata,
            experiment.output_feature_name,
            top_n_classes=[0],
            normalize=False,
            model_names=['Model1', 'Model2'],
            output_directory=experiment.output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 4 == len(figure_cnt)
    shutil.rmtree(experiment.output_dir, ignore_errors=True)


def test_frequency_vs_f1_vis_api(csv_filename):
    """Ensure pdf and png figures can be saved via visualization API call.

    :param csv_filename: csv fixture from tests.fixtures.filenames.csv_filename
    :return: None
    """
    experiment = Experiment(csv_filename)
    # extract test stats
    test_stats = experiment.test_stats_full
    viz_outputs = ('pdf', 'png')
    for viz_output in viz_outputs:
        vis_output_pattern_pdf = experiment.output_dir + '/*.{}'.format(
            viz_output
        )
        visualize.frequency_vs_f1(
            [test_stats, test_stats],
            experiment.ground_truth_metadata,
            experiment.output_feature_name,
            top_n_classes=[0],
            model_names=['Model1', 'Model2'],
            output_directory=experiment.output_dir,
            file_format=viz_output
        )
        figure_cnt = glob.glob(vis_output_pattern_pdf)
        assert 2 == len(figure_cnt)
    shutil.rmtree(experiment.output_dir, ignore_errors=True)
