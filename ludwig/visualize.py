#! /usr/bin/env python
# coding=utf-8
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import sys

import numpy as np
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from tabulate import tabulate

from ludwig.constants import *
from ludwig.utils import visualization_utils
from ludwig.utils.data_utils import load_json, load_from_file
from ludwig.utils.print_utils import logging_level_registry


def compare_classifiers_performance(
        test_stats,
        field, algorithms=None,
        **kwargs
):
    if len(test_stats) < 1:
        logging.error('No test_stats provided')
        return

    test_stats_per_algorithm = [load_json(test_stats_f)
                                for test_stats_f in test_stats]

    fields_set = set()
    for ls in test_stats_per_algorithm:
        for key in ls:
            fields_set.add(key)
    fields = [field] if field is not None and len(field) > 0 else fields_set

    for field in fields:
        accuracies = []
        hits_at_ks = []
        edit_distances = []

        for test_stats in test_stats_per_algorithm:
            if ACCURACY in test_stats[field]:
                accuracies.append(test_stats[field][ACCURACY])
            if HITS_AT_K in test_stats[field]:
                hits_at_ks.append(test_stats[field][HITS_AT_K])
            if EDIT_DISTANCE in test_stats[field]:
                edit_distances.append(test_stats[field][EDIT_DISTANCE])

        measures = []
        measures_names = []
        if len(accuracies) > 0:
            measures.append(accuracies)
            measures_names.append(ACCURACY)
        if len(hits_at_ks) > 0:
            measures.append(hits_at_ks)
            measures_names.append(HITS_AT_K)
        if len(edit_distances) > 0:
            measures.append(edit_distances)
            measures_names.append(EDIT_DISTANCE)

        visualization_utils.compare_classifiers_plot(
            measures,
            measures_names,
            algorithms,
            title='Performance comparison on {}'.format(field)
        )


def compare_classifiers_performance_from_prob(
        probabilities,
        ground_truth,
        field,
        top_n_classes,
        labels_limit,
        algorithms=None,
        **kwargs
):
    if len(probabilities) < 1:
        logging.error('No probabilities provided')
        return

    k = top_n_classes[0]

    gt = load_from_file(ground_truth, field)
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit

    probs = [load_from_file(probs_fn, dtype=float)
             for probs_fn in probabilities]

    accuracies = []
    hits_at_ks = []
    mrrs = []

    for i, prob in enumerate(probs):

        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, :labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            prob = prob_limit

        prob = np.argsort(prob, axis=1)
        top1 = prob[:, -1]
        topk = prob[:, -k:]

        accuracies.append((gt == top1).sum() / len(gt))

        hits_at_k = 0
        for j in range(len(gt)):
            hits_at_k += np.in1d(gt[j], topk[i, :])
        hits_at_ks.append(np.asscalar(hits_at_k) / len(gt))

        mrr = 0
        for j in range(len(gt)):
            gt_pos_in_probs = prob[i, :] == gt[j]
            if np.any(gt_pos_in_probs):
                mrr += (1 / -(np.asscalar(np.argwhere(gt_pos_in_probs)) -
                              prob.shape[1]))
        mrrs.append(mrr / len(gt))

    visualization_utils.compare_classifiers_plot(
        [accuracies, hits_at_ks, mrrs],
        [ACCURACY, HITS_AT_K, 'mrr'],
        algorithms
    )


def compare_classifiers_performance_from_pred(
        predictions,
        ground_truth,
        ground_truth_metadata,
        field,
        labels_limit,
        algorithms=None,
        **kwargs
):
    if len(predictions) < 1:
        logging.error('No predictions provided')
        return

    metadata = load_json(ground_truth_metadata)

    gt = load_from_file(ground_truth, field)
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit

    preds = [load_from_file(preds_fn, dtype=str) for preds_fn in predictions]

    mapped_preds = []
    for pred in preds:
        mapped_preds.append([metadata[field]['str2idx'][val] for val in pred])
    preds = mapped_preds

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for i, pred in enumerate(preds):
        accuracies.append(sklearn.metrics.accuracy_score(gt, pred))
        precisions.append(
            sklearn.metrics.precision_score(gt, pred, average='macro')
        )
        recalls.append(sklearn.metrics.recall_score(gt, pred, average='macro'))
        f1s.append(sklearn.metrics.f1_score(gt, pred, average='macro'))

    visualization_utils.compare_classifiers_plot(
        [accuracies, precisions, recalls, f1s],
        [ACCURACY, 'precision', 'recall', 'f1'],
        algorithms
    )


def compare_classifiers_performance_subset(
        probabilities,
        ground_truth,
        field,
        top_n_classes,
        labels_limit,
        subset,
        algorithms=None,
        **kwargs
):
    if len(probabilities) < 1:
        logging.error('No probabilities provided')
        return

    k = top_n_classes[0]

    gt = load_from_file(ground_truth, field)
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit

    subset_indices = gt > 0
    gt_subset = gt
    if subset == 'ground_truth':
        subset_indices = gt < k
        gt_subset = gt[subset_indices]
        logging.info('Subset is {:.2f}% of the data'.format(
            len(gt_subset) / len(gt) * 100)
        )

    probs = [load_from_file(probs_fn, dtype=float)
             for probs_fn in probabilities]

    accuracies = []
    hits_at_ks = []

    for i, prob in enumerate(probs):

        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, :labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            prob = prob_limit

        if subset == PREDICTIONS:
            subset_indices = np.argmax(prob, axis=1) < k
            gt_subset = gt[subset_indices]
            logging.info(
                'Subset for algorithm {} is {:.2f}% of the data'.format(
                    algorithms[i] if algorithms and i < len(algorithms) else i,
                    len(gt_subset) / len(gt) * 100
                )
            )
            algorithms[i] = '{} ({:.2f}%)'.format(
                algorithms[i] if algorithms and i < len(algorithms) else i,
                len(gt_subset) / len(gt) * 100
            )

        prob_subset = prob[subset_indices]

        prob_subset = np.argsort(prob_subset, axis=1)
        top1_subset = prob_subset[:, -1]
        top3_subset = prob_subset[:, -3:]

        ## TODO - double check this - pycharm warns "Unresolved attribute reference"
        accuracies.append((gt_subset == top1_subset).sum() / len(gt_subset))

        hits_at_k = 0
        for j in range(len(gt_subset)):
            hits_at_k += np.in1d(gt_subset[j], top3_subset[i, :])
        hits_at_ks.append(np.asscalar(hits_at_k) / len(gt_subset))

    title = None
    if subset == 'ground_truth':
        title = 'Classifier performance on first {} '.format(
            k, 'es' if k > 1 else '', len(gt_subset) / len(gt) * 100
        )
    elif subset == PREDICTIONS:
        title = 'Classifier performance on first {} class{}'.format(
            k,
            'es' if k > 1 else ''
        )

    visualization_utils.compare_classifiers_plot(
        [accuracies, hits_at_ks],
        [ACCURACY, HITS_AT_K],
        algorithms,
        title=title
    )


def compare_classifiers_performance_changing_k(
        probabilities,
        ground_truth,
        field,
        top_k,
        labels_limit,
        algorithms=None,
        **kwargs
):
    if len(probabilities) < 1:
        logging.error('No probabilities provided')
        return

    k = top_k

    gt = load_from_file(ground_truth, field)
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit

    probs = [load_from_file(probs_fn, dtype=float)
             for probs_fn in probabilities]

    hits_at_ks = []

    for i, prob in enumerate(probs):

        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, :labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            prob = prob_limit

        prob = np.argsort(prob, axis=1)

        hits_at_k = [0.0] * k
        for g in range(len(gt)):
            for j in range(k):
                hits_at_k[j] += np.in1d(gt[g], prob[g, -j - 1:])
        hits_at_ks.append(np.array(hits_at_k) / len(gt))

    visualization_utils.compare_classifiers_line_plot(
        np.arange(1, k + 1),
        hits_at_ks, 'hits@k',
        algorithms,
        title='Classifier comparison (hits@k)'
    )


def compare_classifiers_predictions(
        predictions,
        ground_truth,
        field,
        labels_limit,
        algorithms=None,
        **kwargs
):
    if len(predictions) < 2:
        logging.error('No predictions provided')
        return

    ground_truth_fn = ground_truth
    ground_truth_field = field
    predictions_1_fn = predictions[0]
    predictions_2_fn = predictions[1]
    name_c1 = (algorithms[0] if algorithms is not None and len(algorithms) > 0
               else 'c1')
    name_c2 = (algorithms[1] if algorithms is not None and len(algorithms) > 1
               else 'c2')

    gt = load_from_file(ground_truth_fn, ground_truth_field)
    pred_c1 = load_from_file(predictions_1_fn, dtype=int)
    pred_c2 = load_from_file(predictions_2_fn, dtype=int)

    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit
        pred_c1[pred_c1 > labels_limit] = labels_limit
        pred_c2[pred_c2 > labels_limit] = labels_limit

    # DOTO all shadows built in name - come up with a more descriptive name
    all = len(gt)
    if all == 0:
        logging.error('No labels in the ground truth')
        return

    both_right = 0
    both_wrong_same = 0
    both_wrong_different = 0
    c1_right_c2_wrong = 0
    c1_wrong_c2_right = 0

    for i in range(all):
        if gt[i] == pred_c1[i] and gt[i] == pred_c2[i]:
            both_right += 1
        elif gt[i] != pred_c1[i] and gt[i] != pred_c2[i]:
            if pred_c1[i] == pred_c2[i]:
                both_wrong_same += 1
            else:
                both_wrong_different += 1
        elif gt[i] == pred_c1[i] and gt[i] != pred_c2[i]:
            c1_right_c2_wrong += 1
        elif gt[i] != pred_c1[i] and gt[i] == pred_c2[i]:
            c1_wrong_c2_right += 1

    one_right = c1_right_c2_wrong + c1_wrong_c2_right
    both_wrong = both_wrong_same + both_wrong_different

    logging.info('Test datapoints: {}'.format(all))
    logging.info(
        'Both right: {} {:.2f}%'.format(both_right, 100 * both_right / all))
    logging.info(
        'One right: {} {:.2f}%'.format(one_right, 100 * one_right / all))
    logging.info(
        '  {} right / {} wrong: {} {:.2f}% {:.2f}%'.format(
            name_c1,
            name_c2,
            c1_right_c2_wrong,
            100 * c1_right_c2_wrong / all,
            100 * c1_right_c2_wrong / one_right if one_right > 0 else 0
        )
    )
    logging.info(
        '  {} wrong / {} right: {} {:.2f}% {:.2f}%'.format(
            name_c1,
            name_c2,
            c1_wrong_c2_right,
            100 * c1_wrong_c2_right / all,
            100 * c1_wrong_c2_right / one_right if one_right > 0 else 0
        )
    )
    logging.info(
        'Both wrong: {} {:.2f}%'.format(both_wrong, 100 * both_wrong / all)
    )
    logging.info('  same prediction: {} {:.2f}% {:.2f}%'.format(
        both_wrong_same,
        100 * both_wrong_same / all,
        100 * both_wrong_same / both_wrong if both_wrong > 0 else 0
    )
    )
    logging.info('  different prediction: {} {:.2f}% {:.2f}%'.format(
        both_wrong_different,
        100 * both_wrong_different / all,
        100 * both_wrong_different / both_wrong if both_wrong > 0 else 0
    )
    )

    visualization_utils.donut(
        [both_right, one_right, both_wrong],
        ['both right', 'one right', 'both wrong'],
        [both_right, c1_right_c2_wrong, c1_wrong_c2_right, both_wrong_same,
         both_wrong_different],
        ['both right',
         '{} right / {} wrong'.format(name_c1, name_c2),
         '{} wrong / {} right'.format(name_c1, name_c2),
         'same prediction', 'different prediction'],
        [0, 1, 1, 2, 2],
        title='{} vs {}'.format(name_c1, name_c2)
    )


def compare_classifiers_predictions_distribution(
        predictions,
        ground_truth,
        field,
        labels_limit,
        algorithms=None,
        **kwargs
):
    if len(predictions) < 1:
        logging.error('No predictions provided')
        return

    ground_truth = load_from_file(ground_truth, field)
    if labels_limit > 0:
        ground_truth[ground_truth > labels_limit] = labels_limit

    predictions = [load_from_file(predictions_fn, dtype=int)
                   for predictions_fn in predictions]
    if labels_limit > 0:
        for i in range(len(predictions)):
            predictions[i][predictions[i] > labels_limit] = labels_limit

    counts_gt = np.bincount(ground_truth)
    prob_gt = counts_gt / counts_gt.sum()

    counts_predictions = [np.bincount(alg_predictions)
                          for alg_predictions in predictions]

    prob_predictions = [alg_count_prediction / alg_count_prediction.sum()
                        for alg_count_prediction in counts_predictions]

    visualization_utils.radar_chart(prob_gt, prob_predictions, algorithms)


def confidence_filtering(
        probabilities,
        ground_truth,
        field,
        labels_limit,
        algorithms=None,
        **kwargs
):
    if len(probabilities) < 1:
        logging.error('No probabilities provided')
        return

    gt = load_from_file(ground_truth, field)
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit

    probs = [load_from_file(probs_fn, dtype=float)
             for probs_fn in probabilities]

    thresholds = [t / 100 for t in range(0, 101, 5)]

    accuracies = []
    dataset_kept = []

    for i, prob in enumerate(probs):

        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, :labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            prob = prob_limit

        max_prob = np.max(prob, axis=1)
        predictions = np.argmax(prob, axis=1)

        accuracies_alg = []
        dataset_kept_alg = []

        for threshold in thresholds:
            threshold = threshold if threshold < 1 else 0.999
            filtered_indices = max_prob >= threshold
            filtered_gt = gt[filtered_indices]
            filtered_predictions = predictions[filtered_indices]
            accuracy = (
                    (filtered_gt == filtered_predictions).sum() /
                    len(filtered_gt)
            )

            accuracies_alg.append(accuracy)
            dataset_kept_alg.append(len(filtered_gt) / len(gt))

        accuracies.append(accuracies_alg)
        dataset_kept.append(dataset_kept_alg)

    visualization_utils.confidence_fitlering_plot(
        thresholds,
        accuracies,
        dataset_kept,
        algorithms,
        title='Confidence Filtering'
    )


def confidence_filtering_2d(
        probabilities,
        ground_truth,
        threshold_fields,
        labels_limit,
        algorithms=None,  # DOTO this param is unused?
        **kwargs
):
    if len(probabilities) < 2:
        logging.error('Not enough probabilities provided, two are needed')
        return
    if len(probabilities) > 2:
        logging.error('Too many probabilities provided, only two are needed')
        return
    if len(threshold_fields) < 2:
        logging.error('Not enough threshold fields provided, two are needed')
        return
    if len(threshold_fields) > 2:
        logging.error('Too many threshold fields provided, only two are needed')
        return

    gt_1 = load_from_file(ground_truth, threshold_fields[0])
    gt_2 = load_from_file(ground_truth, threshold_fields[1])

    if labels_limit > 0:
        gt_1[gt_1 > labels_limit] = labels_limit
        gt_2[gt_2 > labels_limit] = labels_limit

    probs = [load_from_file(probs_fn, dtype=float)
             for probs_fn in probabilities]

    thresholds = [t / 100 for t in range(0, 101, 5)]

    accuracies = []
    dataset_kept = []

    if labels_limit > 0 and probs[0].shape[1] > labels_limit + 1:
        prob_limit = probs[0][:, :labels_limit + 1]
        prob_limit[:, labels_limit] = probs[0][:, labels_limit:].sum(1)
        probs[0] = prob_limit

    if labels_limit > 0 and probs[1].shape[1] > labels_limit + 1:
        prob_limit = probs[1][:, :labels_limit + 1]
        prob_limit[:, labels_limit] = probs[1][:, labels_limit:].sum(1)
        probs[1] = prob_limit

    max_prob_1 = np.max(probs[0], axis=1)
    predictions_1 = np.argmax(probs[0], axis=1)

    max_prob_2 = np.max(probs[1], axis=1)
    predictions_2 = np.argmax(probs[1], axis=1)

    for threshold_1 in thresholds:
        threshold_1 = threshold_1 if threshold_1 < 1 else 0.999
        curr_accuracies = []
        curr_dataset_kept = []

        for threshold_2 in thresholds:
            threshold_2 = threshold_2 if threshold_2 < 1 else 0.999

            filtered_indices = np.logical_and(
                max_prob_1 >= threshold_1,
                max_prob_2 >= threshold_2
            )

            filtered_gt_1 = gt_1[filtered_indices]
            filtered_predictions_1 = predictions_1[filtered_indices]
            filtered_gt_2 = gt_2[filtered_indices]
            filtered_predictions_2 = predictions_2[filtered_indices]

            accuracy = (
                           np.logical_and(
                               filtered_gt_1 == filtered_predictions_1,
                               filtered_gt_2 == filtered_predictions_2
                           )
                       ).sum() / len(filtered_gt_1)

            curr_accuracies.append(accuracy)
            curr_dataset_kept.append(len(filtered_gt_1) / len(gt_1))

        accuracies.append(curr_accuracies)
        dataset_kept.append(curr_dataset_kept)

    visualization_utils.confidence_fitlering_3d_plot(
        thresholds,
        thresholds,
        accuracies,
        dataset_kept,
        threshold_fields,
        title='Confidence Filtering'
    )


def confidence_filtering_data_vs_acc(
        probabilities,
        ground_truth,
        field,
        labels_limit,
        algorithms=None,
        **kwargs
):
    if len(probabilities) < 1:
        logging.error('No probabilities provided')
        return

    gt = load_from_file(ground_truth, field)
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit

    probs = [load_from_file(probs_fn, dtype=float)
             for probs_fn in probabilities]

    thresholds = [t / 100 for t in range(0, 101, 5)]

    accuracies = []
    dataset_kept = []

    for i, prob in enumerate(probs):

        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, :labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            prob = prob_limit

        max_prob = np.max(prob, axis=1)
        predictions = np.argmax(prob, axis=1)

        accuracies_alg = []
        dataset_kept_alg = []

        for threshold in thresholds:
            threshold = threshold if threshold < 1 else 0.999
            filtered_indices = max_prob >= threshold
            filtered_gt = gt[filtered_indices]
            filtered_predictions = predictions[filtered_indices]
            accuracy = ((filtered_gt == filtered_predictions).sum() /
                        len(filtered_gt))

            accuracies_alg.append(accuracy)
            dataset_kept_alg.append(len(filtered_gt) / len(gt))

        accuracies.append(accuracies_alg)
        dataset_kept.append(dataset_kept_alg)

    visualization_utils.confidence_fitlering_data_vs_acc_plot(
        accuracies,
        dataset_kept,
        algorithms,
        title='Confidence Filtering (Data vs Accuracy)'
    )


def confidence_filtering_data_vs_acc_2d(
        probabilities,
        ground_truth,
        threshold_fields,
        labels_limit,
        algorithms=None,
        **kwargs
):
    if len(probabilities) < 2:
        logging.error('Not enough probabilities provided, two are needed')
        return
    if len(probabilities) > 2:
        logging.error('Too many probabilities provided, only two are needed')
        return
    if len(threshold_fields) < 2:
        logging.error('Not enough threshold fields provided, two are needed')
        return
    if len(threshold_fields) > 2:
        logging.error('Too many threshold fields provided, only two are needed')
        return

    gt_1 = load_from_file(ground_truth, threshold_fields[0])
    gt_2 = load_from_file(ground_truth, threshold_fields[1])

    if labels_limit > 0:
        gt_1[gt_1 > labels_limit] = labels_limit
        gt_2[gt_2 > labels_limit] = labels_limit

    probs = [load_from_file(probs_fn, dtype=float)
             for probs_fn in probabilities]

    thresholds = [t / 100 for t in range(0, 101, 5)]
    fixed_step_coverage = thresholds
    name_t1 = 'threshold_{}'.format(threshold_fields[0])
    name_t2 = 'threshold_{}'.format(threshold_fields[1])

    accuracies = []
    dataset_kept = []
    interps = []
    table = [[name_t1, name_t2, 'coverage', ACCURACY]]

    if labels_limit > 0 and probs[0].shape[1] > labels_limit + 1:
        prob_limit = probs[0][:, :labels_limit + 1]
        prob_limit[:, labels_limit] = probs[0][:, labels_limit:].sum(1)
        probs[0] = prob_limit

    if labels_limit > 0 and probs[1].shape[1] > labels_limit + 1:
        prob_limit = probs[1][:, :labels_limit + 1]
        prob_limit[:, labels_limit] = probs[1][:, labels_limit:].sum(1)
        probs[1] = prob_limit

    max_prob_1 = np.max(probs[0], axis=1)
    predictions_1 = np.argmax(probs[0], axis=1)

    max_prob_2 = np.max(probs[1], axis=1)
    predictions_2 = np.argmax(probs[1], axis=1)

    for threshold_1 in thresholds:
        threshold_1 = threshold_1 if threshold_1 < 1 else 0.999
        curr_accuracies = []
        curr_dataset_kept = []

        for threshold_2 in thresholds:
            threshold_2 = threshold_2 if threshold_2 < 1 else 0.999

            filtered_indices = np.logical_and(
                max_prob_1 >= threshold_1,
                max_prob_2 >= threshold_2
            )

            filtered_gt_1 = gt_1[filtered_indices]
            filtered_predictions_1 = predictions_1[filtered_indices]
            filtered_gt_2 = gt_2[filtered_indices]
            filtered_predictions_2 = predictions_2[filtered_indices]

            coverage = len(filtered_gt_1) / len(gt_1)
            accuracy = (
                           np.logical_and(
                               filtered_gt_1 == filtered_predictions_1,
                               filtered_gt_2 == filtered_predictions_2
                           )
                       ).sum() / len(filtered_gt_1)

            curr_accuracies.append(accuracy)
            curr_dataset_kept.append(coverage)
            table.append([threshold_1, threshold_2, coverage, accuracy])

        accuracies.append(curr_accuracies)
        dataset_kept.append(curr_dataset_kept)
        interps.append(
            np.interp(
                fixed_step_coverage,
                list(reversed(curr_dataset_kept)),
                list(reversed(curr_accuracies)),
                left=1,
                right=0
            )
        )

    logging.info('CSV table')
    for row in table:
        logging.info(','.join([str(e) for e in row]))

    # ===========#
    # Multiline #
    # ===========#
    visualization_utils.confidence_fitlering_data_vs_acc_multiline_plot(
        accuracies,
        dataset_kept,
        algorithms,
        title='Coverage vs Accuracy'
    )
    # ==========#
    # Max line #
    # ==========#
    max_accuracies = np.amax(np.array(interps), 0)
    visualization_utils.confidence_fitlering_data_vs_acc_plot(
        [max_accuracies],
        [thresholds],
        algorithms,
        title='Coverage vs Accuracy'
    )

    # ==========================#
    # Max line with thresholds #
    # ==========================#
    acc_matrix = np.array(accuracies)
    cov_matrix = np.array(dataset_kept)
    t1_maxes = [1]
    t2_maxes = [1]
    for i in range(len(fixed_step_coverage) - 1):
        lower = fixed_step_coverage[i]
        upper = fixed_step_coverage[i + 1]
        indices = np.logical_and(cov_matrix >= lower, cov_matrix < upper)
        selected_acc = acc_matrix.copy()
        selected_acc[np.logical_not(indices)] = -1
        threshold_indices = np.unravel_index(np.argmax(selected_acc, axis=None),
                                             selected_acc.shape)
        t1_maxes.append(thresholds[threshold_indices[
            0]])  # TODO pycharm warning - unexpected type
        t2_maxes.append(thresholds[threshold_indices[1]])
    algorithm_name = algorithms[0] if algorithms is not None and len(
        algorithms) > 0 else ''
    visualization_utils.confidence_fitlering_data_vs_acc_plot(
        [max_accuracies, t1_maxes, t2_maxes],
        [fixed_step_coverage, fixed_step_coverage, fixed_step_coverage],
        algorithm_names=[algorithm_name, name_t1, name_t2],
        dotted=[False, True, True],
        title='Coverage vs Accuracy'
    )


def confidence_filtering_data_vs_acc_subset(
        probabilities,
        ground_truth,
        field,
        top_n_classes,
        labels_limit,
        subset,
        algorithms=None,
        **kwargs
):
    if len(probabilities) < 1:
        logging.error('No probabilities provided')
        return

    k = top_n_classes[0]
    gt = load_from_file(ground_truth, field)
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit

    probs = [load_from_file(probs_fn, dtype=float)
             for probs_fn in probabilities]

    thresholds = [t / 100 for t in range(0, 101, 5)]

    accuracies = []
    dataset_kept = []

    subset_indices = gt > 0
    gt_subset = gt
    if subset == 'ground_truth':
        subset_indices = gt < k
        gt_subset = gt[subset_indices]
        logging.info('Subset is {:.2f}% of the data'.format(
            len(gt_subset) / len(gt) * 100)
        )

    for i, prob in enumerate(probs):

        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, :labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            prob = prob_limit

        if subset == PREDICTIONS:
            subset_indices = np.argmax(prob, axis=1) < k
            gt_subset = gt[subset_indices]
            logging.info(
                'Subset for algorithm {} is {:.2f}% of the data'.format(
                    algorithms[i] if algorithms and i < len(algorithms) else i,
                    len(gt_subset) / len(gt) * 100
                )
            )

        prob_subset = prob[subset_indices]

        max_prob = np.max(prob_subset, axis=1)
        predictions = np.argmax(prob_subset, axis=1)

        accuracies_alg = []
        dataset_kept_alg = []

        for threshold in thresholds:
            threshold = threshold if threshold < 1 else 0.999
            filtered_indices = max_prob >= threshold
            filtered_gt = gt_subset[filtered_indices]
            filtered_predictions = predictions[filtered_indices]
            accuracy = ((filtered_gt == filtered_predictions).sum() /
                        len(filtered_gt))

            accuracies_alg.append(accuracy)
            dataset_kept_alg.append(len(filtered_gt) / len(gt))

        accuracies.append(accuracies_alg)
        dataset_kept.append(dataset_kept_alg)

    visualization_utils.confidence_fitlering_data_vs_acc_plot(
        accuracies,
        dataset_kept,
        algorithms,
        title='Confidence Filtering (Data vs Accuracy)'
    )


def confidence_filtering_data_vs_acc_subset_per_class(
        probabilities,
        ground_truth,
        field,
        top_n_classes,
        labels_limit,
        subset,
        algorithms=None,
        **kwargs
):
    if len(probabilities) < 1:
        logging.error('No probabilities provided')
        return

    k = top_n_classes[0]
    gt = load_from_file(ground_truth, field)
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit

    probs = [load_from_file(probs_fn, dtype=float)
             for probs_fn in probabilities]

    thresholds = [t / 100 for t in range(0, 101, 5)]

    for curr_k in range(k):
        accuracies = []
        dataset_kept = []

        subset_indices = gt > 0
        gt_subset = gt
        if subset == 'ground_truth':
            subset_indices = gt == curr_k
            gt_subset = gt[subset_indices]
            logging.info('Subset is {:.2f}% of the data'.format(
                len(gt_subset) / len(gt) * 100)
            )

        for i, prob in enumerate(probs):

            if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
                prob_limit = prob[:, :labels_limit + 1]
                prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
                prob = prob_limit

            if subset == PREDICTIONS:
                subset_indices = np.argmax(prob, axis=1) == curr_k
                gt_subset = gt[subset_indices]
                logging.info(
                    'Subset for algorithm {} is {:.2f}% of the data'.format(
                        algorithms[i] if algorithms and i < len(
                            algorithms) else i,
                        len(gt_subset) / len(gt) * 100
                    )
                )

            prob_subset = prob[subset_indices]

            max_prob = np.max(prob_subset, axis=1)
            predictions = np.argmax(prob_subset, axis=1)

            accuracies_alg = []
            dataset_kept_alg = []

            for threshold in thresholds:
                threshold = threshold if threshold < 1 else 0.999
                filtered_indices = max_prob >= threshold
                filtered_gt = gt_subset[filtered_indices]
                filtered_predictions = predictions[filtered_indices]
                accuracy = ((filtered_gt == filtered_predictions).sum() /
                            len(filtered_gt) if len(filtered_gt) > 0 else 0)

                accuracies_alg.append(accuracy)
                dataset_kept_alg.append(len(filtered_gt) / len(gt))

            accuracies.append(accuracies_alg)
            dataset_kept.append(dataset_kept_alg)

        visualization_utils.confidence_fitlering_data_vs_acc_plot(
            accuracies, dataset_kept, algorithms,
            title='Confidence Filtering (Data vs Accuracy) '
                  'for class {}'.format(curr_k)
        )


def data_vs_acc_subset(
        predictions,
        ground_truth,
        field,
        top_n_classes,
        labels_limit,
        subset,
        algorithms=None,
        **kwargs
):
    if len(predictions) < 1:
        logging.error('No predictions provided')
        return

    k = top_n_classes[0]
    gt = load_from_file(ground_truth, field)
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit

    preds = [load_from_file(preds_fn, dtype=float) for preds_fn in predictions]

    subset_indices = gt > 0
    gt_subset = gt

    accuracies = []
    data_percentage = []

    if subset == 'ground_truth':
        subset_indices = gt < k
        gt_subset = gt[subset_indices]
        logging.info('Subset is {:.2f}% of the data'.format(
            len(gt_subset) / len(gt) * 100)
        )

    for i, pred in enumerate(preds):

        if labels_limit > 0:
            pred[pred > labels_limit + 1] = labels_limit + 1

        if subset == PREDICTIONS:
            subset_indices = pred < k
            gt_subset = gt[subset_indices]

        pred_subset = pred[subset_indices]
        accuracy = (gt_subset == pred_subset).sum() / len(gt_subset)

        accuracies.append(100 * accuracy)
        data_percentage.append(100 * len(gt_subset) / len(gt))

    table = [['model', '% accuracy', '% data']]
    for i in range(len(accuracies)):
        table.append(
            [algorithms[i] if algorithms and i < len(algorithms) else i,
             accuracies[i], data_percentage[i]]
        )

    logging.info(
        tabulate(
            table,
            headers='firstrow',
            tablefmt='orgtbl',
            floatfmt='.4f'
        )
    )

    visualization_utils.compare_classifiers_plot(
        [accuracies, data_percentage],
        ['% accuracy', '% data'],
        algorithms,
        adaptive=True,
        title='Accuracy and Data comparison'
    )


def data_vs_acc_subset_per_class(
        predictions,
        ground_truth,
        field,
        top_n_classes,
        labels_limit,
        subset,
        algorithms=None,
        **kwargs
):
    if len(predictions) < 1:
        logging.error('No predictions provided')
        return

    k = top_n_classes[0]
    gt = load_from_file(ground_truth, field)
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit

    preds = [load_from_file(preds_fn, dtype=float) for preds_fn in predictions]

    for curr_k in range(k):
        subset_indices = gt > 0
        gt_subset = gt

        accuracies = []
        data_percentage = []

        if subset == 'ground_truth':
            subset_indices = gt == curr_k
            gt_subset = gt[subset_indices]
            logging.info('Subset is {:.2f}% of the data'.format(
                len(gt_subset) / len(gt) * 100)
            )

        for i, pred in enumerate(preds):

            if labels_limit > 0:
                pred[pred > labels_limit + 1] = labels_limit + 1

            if subset == PREDICTIONS:
                subset_indices = pred == curr_k
                gt_subset = gt[subset_indices]

            pred_subset = pred[subset_indices]
            accuracy = (gt_subset == pred_subset).sum() / len(gt_subset)

            accuracies.append(100 * accuracy)
            data_percentage.append(100 * len(gt_subset) / len(gt))

        table = [['model', '% accuracy', '% data']]
        for i in range(len(accuracies)):
            table.append(
                [algorithms[i] if algorithms and i < len(algorithms) else i,
                 accuracies[i],
                 data_percentage[i]]
            )

        logging.info(
            tabulate(
                table,
                headers='firstrow',
                tablefmt='orgtbl',
                floatfmt='.4f'
            )
        )

        visualization_utils.compare_classifiers_plot(
            [accuracies, data_percentage],
            ['% accuracy', '% data'],
            algorithms,
            adaptive=True,
            title='Accuracy vs Data comparison for class {}'.format(curr_k)
        )


def binary_threshold_vs_metric(
        probabilities,
        ground_truth,
        field,
        metrics,
        positive_label=1,
        algorithms=None,
        **kwargs
):
    if len(probabilities) < 1:
        logging.error('No probabilities provided')
        return

    gt = load_from_file(ground_truth, field)

    probs = [load_from_file(probs_fn, dtype=float)
             for probs_fn in probabilities]

    thresholds = [t / 100 for t in range(0, 101, 5)]

    for metric in metrics:

        scores = []

        for i, prob in enumerate(probs):

            scores_alg = []

            if len(prob.shape) == 2:
                if prob.shape[1] > positive_label:
                    prob = prob[:, positive_label]
                else:
                    raise Exception(
                        'the specified positive label {} is not '
                        'present in the probabilities'.format(
                            positive_label
                        )
                    )

            for threshold in thresholds:
                threshold = threshold if threshold < 1 else 0.999

                predictions = prob >= threshold

                if metric == 'f1':
                    metric_score = sklearn.metrics.f1_score(gt, predictions)
                elif metric == 'precision':
                    metric_score = sklearn.metrics.precision_score(
                        gt,
                        predictions
                    )
                elif metric == 'recall':
                    metric_score = sklearn.metrics.recall_score(gt, predictions)
                else:  # if metric == ACCURACY:
                    metric_score = sklearn.metrics.accuracy_score(
                        gt,
                        predictions
                    )

                scores_alg.append(metric_score)

            scores.append(scores_alg)

        visualization_utils.threshold_vs_metric_plot(
            thresholds,
            scores,
            algorithms,
            title='Binary threshold vs {}'.format(metric)
        )


def roc_curves(
        probabilities,
        ground_truth,
        field,
        positive_label=1,
        algorithms=None,
        **kwargs
):
    if len(probabilities) < 1:
        logging.error('No probabilities provided')
        return

    gt = load_from_file(ground_truth, field)
    probs = [load_from_file(probs_fn, dtype=float)
             for probs_fn in probabilities]

    fpr_tprs = []

    for i, prob in enumerate(probs):
        if len(prob.shape) > 1:
            prob = prob[:, positive_label]
        fpr, tpr, _ = sklearn.metrics.roc_curve(
            gt, prob,
            pos_label=positive_label
        )
        fpr_tprs.append((fpr, tpr))

    visualization_utils.roc_curves(fpr_tprs, algorithms, title='ROC curves')


def roc_curves_from_test_stats(test_stats, field, algorithms=None, **kwargs):
    if len(test_stats) < 1:
        logging.error('No test_stats provided')
        return

    test_stats_per_algorithm = [load_json(test_stats_f)
                                for test_stats_f in test_stats]
    fpr_tprs = []
    for curr_test_stats in test_stats_per_algorithm:
        fpr = curr_test_stats[field]['roc_curve']['false_positive_rate']
        tpr = curr_test_stats[field]['roc_curve']['true_positive_rate']
        fpr_tprs.append((fpr, tpr))

    visualization_utils.roc_curves(fpr_tprs, algorithms, title='ROC curves')


def calibration_1_vs_all(
        probabilities,
        ground_truth,
        field,
        top_n_classes,
        labels_limit,
        algorithms=None,
        **kwargs
):
    if len(probabilities) < 1:
        logging.error('No probabilities provided')
        return

    gt = load_from_file(ground_truth, field)
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit
    probs = [load_from_file(probs_fn, dtype=float)
             for probs_fn in probabilities]
    for i, prob in enumerate(probs):
        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, :labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            probs[i] = prob_limit

    num_classes = max(gt)

    brier_scores = []

    classes = (min(num_classes, top_n_classes[0]) if top_n_classes[0] > 0
               else num_classes)

    for class_idx in range(classes):
        fraction_positives_class = []
        mean_predicted_vals_class = []
        probs_class = []
        brier_scores_class = []
        for prob in probs:
            # ground_truth is an vector of integers, each integer is a class
            # index to have a [0,1] vector we have to check if the value equals
            # the input class index and convert the resulting boolean vector
            # into an integer vector probabilities is a n x c matrix, n is the
            # number of datapoints and c number of classes; its values are the
            # probabilities of the ith datapoint to be classified as belonging
            # to the jth class according to the learned model. For this reason
            # we need to take only the column of predictions that is about the
            # class we are interested in, the input class index

            gt_class = (gt == class_idx).astype(int)
            prob_class = prob[:, class_idx]

            (
                curr_fraction_positives,
                curr_mean_predicted_vals
            ) = calibration_curve(gt_class, prob_class, n_bins=21)

            fraction_positives_class.append(curr_fraction_positives)
            mean_predicted_vals_class.append(curr_mean_predicted_vals)
            probs_class.append(prob[:, class_idx])
            brier_scores_class.append(
                brier_score_loss(
                    gt_class,
                    prob_class, pos_label=1
                )
            )

        brier_scores.append(brier_scores_class)
        visualization_utils.calibration_plot(
            fraction_positives_class,
            mean_predicted_vals_class,
            algorithms
        )
        visualization_utils.predictions_distribution_plot(
            probs_class,
            algorithms
        )

    visualization_utils.brier_plot(np.array(brier_scores), algorithms)


def calibration_multiclass(
        probabilities,
        ground_truth,
        field,
        labels_limit,
        algorithms=None,
        **kwargs
):
    if len(probabilities) < 1:
        logging.error('No probabilities provided')
        return

    gt = load_from_file(ground_truth, field)
    if labels_limit > 0:
        gt[gt > labels_limit] = labels_limit

    prob_classes = 0
    probs = [load_from_file(probs_fn, dtype=float)
             for probs_fn in probabilities]

    for i, prob in enumerate(probs):
        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, :labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            probs[i] = prob_limit
        if probs[i].shape[1] > prob_classes:
            prob_classes = probs[i].shape[1]

    gt_one_hot_dim_2 = max(prob_classes, max(gt) + 1)
    gt_one_hot = np.zeros((len(gt), gt_one_hot_dim_2))
    gt_one_hot[np.arange(len(gt)), gt] = 1
    gt_one_hot_flat = gt_one_hot.flatten()

    fraction_positives = []
    mean_predicted_vals = []
    brier_scores = []
    for prob in probs:
        # flatten probabilities to be compared to flatten ground truth
        prob_flat = prob.flatten()
        curr_fraction_positives, curr_mean_predicted_vals = calibration_curve(
            gt_one_hot_flat,
            prob_flat,
            n_bins=21
        )
        fraction_positives.append(curr_fraction_positives)
        mean_predicted_vals.append(curr_mean_predicted_vals)
        brier_scores.append(
            brier_score_loss(
                gt_one_hot_flat,
                prob_flat,
                pos_label=1
            )
        )

    visualization_utils.calibration_plot(
        fraction_positives,
        mean_predicted_vals,
        algorithms
    )
    visualization_utils.compare_classifiers_plot(
        [brier_scores],
        ['brier'],
        algorithms,
        adaptive=True,
        decimals=8
    )
    for i, brier_score in enumerate(brier_scores):
        if i < len(algorithms):
            # TODO not sure rstrip() avoids the newline
            logging.info('{}:'.format(algorithms[i]).rstrip())
        logging.info(brier_score)


def confusion_matrix(
        test_stats,
        ground_truth,  # TODO - this is unused
        field,
        top_n_classes,
        normalize,
        algorithms=None,
        **kwargs
):
    if len(test_stats) < 1:
        logging.error('No test_stats provided')
        return

    test_stats_per_algorithm = [load_json(test_stats_f)
                                for test_stats_f in test_stats]

    fields_set = set()
    for ls in test_stats_per_algorithm:
        for key in ls:
            fields_set.add(key)
    fields = [field] if field is not None and len(field) > 0 else fields_set

    for i, test_stats in enumerate(test_stats_per_algorithm):
        for field in fields:
            if 'confusion_matrix' in test_stats[field]:
                confusion_matrix = np.array(
                    test_stats[field]['confusion_matrix']
                )
                algorithm_name = algorithms[i] if (
                        algorithms is not None and i < len(algorithms)
                ) else ''

                for k in top_n_classes:
                    k = (min(k, confusion_matrix.shape[0])
                         if k > 0 else confusion_matrix.shape[0])
                    cm = confusion_matrix[0:k + 1, 0:k + 1]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        cm_norm = np.true_divide(cm, cm.sum(1)[:, np.newaxis])
                        cm_norm[cm_norm == np.inf] = 0
                        cm_norm = np.nan_to_num(cm_norm)
                    if normalize:
                        cm = cm_norm
                    visualization_utils.confusion_matrix_plot(
                        cm,
                        title='{} Confusion Matrix top {} {}'.format(
                            algorithm_name, k,
                            field
                        )
                    )

                    cm_entropy = - cm_norm * np.log2(cm_norm)
                    cm_entropy[cm_entropy == np.inf] = 0
                    cm_entropy = np.nan_to_num(cm_entropy)
                    class_entropy = cm_entropy.sum(axis=1)
                    class_desc_entropy = np.argsort(class_entropy)[::-1]
                    desc_entropy = class_entropy[class_desc_entropy]
                    visualization_utils.bar_plot(
                        class_desc_entropy,
                        desc_entropy,
                        title='Classes ranked by entropy of '
                              'Confusion Matrix row'
                    )


def multiclass_multimetric(
        test_stats,
        field,
        top_n_classes,
        algorithms=None,
        **kwargs
):
    if len(test_stats) < 1:
        logging.error('No test_stats provided')
        return

    test_stats_per_algorithm = [load_json(test_stats_f)
                                for test_stats_f in test_stats]

    fields_set = set()
    for ls in test_stats_per_algorithm:
        for key in ls:
            fields_set.add(key)
    fields = [field] if field is not None and len(field) > 0 else fields_set

    for i, test_stats in enumerate(test_stats_per_algorithm):
        for field in fields:
            algorithm_name = (
                algorithms[i]
                if algorithms is not None and i < len(algorithms)
                else ''
            )
            per_class_stats = test_stats[field]['per_class_stats']
            precisions = []
            recalls = []
            f1_scores = []
            labels = []
            for class_id in sorted(map(int, per_class_stats.keys())):
                class_stats = per_class_stats[str(class_id)]
                precisions.append(class_stats['precision'])
                recalls.append(class_stats['recall'])
                f1_scores.append(class_stats['f1_score'])
                labels.append(str(class_id))
            for k in top_n_classes:
                k = min(k, len(precisions)) if k > 0 else len(precisions)
                ps = precisions[0:k]
                rs = recalls[0:k]
                fs = f1_scores[0:k]
                ls = labels[0:k]
                visualization_utils.multiclass_multimetric_plot(
                    [ps, rs, fs],
                    ['precision', 'recall', 'f1 score'],
                    labels=ls,
                    title='{} Multiclass Precision / Recall / '
                          'F1 Score top {} {}'.format(algorithm_name, k, field)
                )

            p_np = np.nan_to_num(np.array(precisions, dtype=np.float32))
            r_np = np.nan_to_num(np.array(recalls, dtype=np.float32))
            f1_np = np.nan_to_num(np.array(f1_scores, dtype=np.float32))
            labels_np = np.nan_to_num(np.array(labels))

            sorted_indices = f1_np.argsort()
            higher_f1s = sorted_indices[-5:][::-1]
            visualization_utils.multiclass_multimetric_plot(
                [p_np[higher_f1s],
                 r_np[higher_f1s],
                 f1_np[higher_f1s]],
                ['precision', 'recall', 'f1 score'],
                labels=labels_np[higher_f1s].tolist(),
                title='{} Multiclass Precision / Recall / '
                      'F1 Score best 5 classes {}'.format(algorithm_name, field)
            )
            lower_f1s = sorted_indices[:5]
            visualization_utils.multiclass_multimetric_plot(
                [p_np[lower_f1s],
                 r_np[lower_f1s],
                 f1_np[lower_f1s]],
                ['precision', 'recall', 'f1 score'],
                labels=labels_np[lower_f1s].tolist(),
                title='{} Multiclass Precision / Recall / F1 Score worst '
                      '5 classes {}'.format(algorithm_name, field)
            )

            visualization_utils.multiclass_multimetric_plot(
                [p_np[sorted_indices[::-1]],
                 r_np[sorted_indices[::-1]],
                 f1_np[sorted_indices[::-1]]],
                ['precision', 'recall', 'f1 score'],
                labels=labels_np[sorted_indices[::-1]].tolist(),
                title='{} Multiclass Precision / Recall / F1 Score '
                      '{} sorted'.format(algorithm_name, field)
            )

            logging.info('\n')
            logging.info(algorithm_name)
            # TODO rstrip()
            logging.info('{0} best 5 classes: '.format(field).rstrip())
            logging.info(higher_f1s)
            logging.info(f1_np[higher_f1s])
            logging.info('{0} worst 5 classes: '.format(field).rstrip())
            logging.info(lower_f1s)
            logging.info(f1_np[lower_f1s])
            logging.info('{0} number of classes with f1 score > 0: '.format(
                field).rstrip())
            logging.info((f1_np > 0).sum())
            logging.info('{0} number of classes with f1 score = 0: '.format(
                field).rstrip())
            logging.info((f1_np == 0).sum())


def frequency_vs_f1(
        test_stats,
        ground_truth_metadata,
        field,
        top_n_classes,
        algorithms=None,
        **kwargs
):
    if len(test_stats) < 1:
        logging.error('No test_stats provided')
        return

    metadata = load_json(ground_truth_metadata)
    test_stats_per_algorithm = [load_json(test_stats_f)
                                for test_stats_f in test_stats]
    k = top_n_classes[0]

    fields_set = set()
    for ls in test_stats_per_algorithm:
        for key in ls:
            fields_set.add(key)
    fields = [field] if field is not None and len(field) > 0 else fields_set

    for i, test_stats in enumerate(test_stats_per_algorithm):
        for field in fields:
            algorithm_name = (algorithms[i]
                              if algorithms is not None and i < len(algorithms)
                              else '')
            per_class_stats = test_stats[field]['per_class_stats']
            f1_scores = []
            labels = []
            class_ids = sorted(map(int, per_class_stats.keys()))
            if k > 0:
                class_ids = class_ids[:k]
            for class_id in class_ids:
                class_stats = per_class_stats[str(class_id)]
                f1_scores.append(class_stats['f1_score'])
                labels.append(str(class_id))

            f1_np = np.nan_to_num(np.array(f1_scores, dtype=np.float32))
            f1_sorted_indices = f1_np.argsort()

            field_frequency_dict = {
                metadata[field]['str2idx'][key]: val
                for key, val in metadata[field]['str2freq'].items()
            }
            field_frequency_np = np.array(
                [field_frequency_dict[class_id]
                 for class_id in sorted(field_frequency_dict)],
                dtype=np.int32
            )

            field_frequency_reordered = field_frequency_np[
                                            f1_sorted_indices[::-1]
                                        ][:len(f1_sorted_indices)]
            f1_reordered = f1_np[f1_sorted_indices[::-1]][
                           :len(f1_sorted_indices)]

            visualization_utils.double_axis_line_plot(
                f1_reordered,
                field_frequency_reordered,
                'F1 score',
                'frequency',
                title='{} F1 Score vs Frequency {}'.format(
                    algorithm_name,
                    field
                )
            )

            frequency_sorted_indices = field_frequency_np.argsort()
            field_frequency_reordered = field_frequency_np[
                                            frequency_sorted_indices[::-1]
                                        ][:len(f1_sorted_indices)]

            f1_reordered = np.zeros(len(field_frequency_reordered))

            for idx in frequency_sorted_indices[::-1]:
                if idx < len(f1_np):
                    f1_reordered[idx] = f1_np[idx]

            visualization_utils.double_axis_line_plot(
                field_frequency_reordered,
                f1_reordered,
                'frequency',
                'F1 score',
                title='{} F1 Score vs Frequency {}'.format(
                    algorithm_name,
                    field
                )
            )


def learning_curves(training_stats, field, algorithms=None, **kwargs):
    if len(training_stats) < 1:
        logging.error('No training_stats provided')
        return

    training_stats_per_algorithm = [load_json(learning_stats_f)
                                    for learning_stats_f in training_stats]

    fields_set = set()
    for ls in training_stats_per_algorithm:
        for _, values in ls.items():
            for key in values:
                fields_set.add(key)
    fields = [field] if field is not None and len(field) > 0 else fields_set

    metrics = [LOSS, ACCURACY, HITS_AT_K, EDIT_DISTANCE]
    for field in fields:
        for metric in metrics:
            if metric in training_stats_per_algorithm[0]['train'][field]:
                visualization_utils.lerning_curves_plot(
                    [learning_stats['train'][field][metric]
                     for learning_stats in training_stats_per_algorithm],
                    [learning_stats['validation'][field][metric]
                     for learning_stats in training_stats_per_algorithm],
                    metric, algorithms,
                    title='Learning Curve {}'.format(field)
                )


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description='This script analyzes results and shows some nice plots.',
        prog='ludwig visualize',
        usage='%(prog)s [options]')

    parser.add_argument('-r', '--raw_data', help='raw data file')
    parser.add_argument('-g', '--ground_truth', help='ground truth file')
    parser.add_argument(
        '-gm',
        '--ground_truth_metadata',
        help='input metadata JSON file'
    )

    parser.add_argument(
        '-v',
        '--visualization',
        default='confidence_filtering',
        choices=['compare_classifiers_performance',
                 'compare_classifiers_performance_from_prob',
                 'compare_classifiers_performance_from_pred',
                 'compare_classifiers_performance_changing_k',
                 'compare_classifiers_performance_subset',
                 'compare_classifiers_predictions',
                 'compare_classifiers_predictions_distribution',
                 'confidence_filtering',
                 'confidence_filtering_2d',
                 'confidence_filtering_data_vs_acc',
                 'confidence_filtering_data_vs_acc_2d',
                 'confidence_filtering_data_vs_acc_subset',
                 'confidence_filtering_data_vs_acc_subset_per_class',
                 'binary_threshold_vs_metric',
                 'roc_curves',
                 'roc_curves_from_test_stats',
                 'data_vs_acc_subset',
                 'data_vs_acc_subset_per_class',
                 'calibration_1_vs_all',
                 'calibration_multiclass',
                 'confusion_matrix',
                 'multiclass_multimetric',
                 'frequency_vs_f1',
                 'learning_curves'],
        help='type of visualization'
    )

    parser.add_argument(
        '-f',
        '--field',
        default=[],
        help='field containing ground truth'
    )
    parser.add_argument(
        '-tf',
        '--threshold_fields',
        default=[],
        nargs='+',
        help='fields for 2d threshold'
    )
    parser.add_argument(
        '-pred',
        '--predictions',
        default=[],
        nargs='+',
        type=str,
        help='predictions files'
    )
    parser.add_argument(
        '-prob',
        '--probabilities',
        default=[],
        nargs='+',
        type=str,
        help='probabilities files'
    )
    parser.add_argument(
        '-tes',
        '--training_stats',
        default=[],
        nargs='+',
        type=str,
        help='training stats files'
    )
    parser.add_argument(
        '-trs',
        '--test_stats',
        default=[],
        nargs='+',
        type=str,
        help='test stats files'
    )
    parser.add_argument(
        '-alg',
        '--algorithms',
        default=[],
        nargs='+',
        type=str,
        help='names of the algorithms (for better graphs)'
    )
    parser.add_argument(
        '-tn',
        '--top_n_classes',
        default=[0],
        nargs='+',
        type=int,
        help='number of classes to plot'
    )
    parser.add_argument(
        '-k',
        '--top_k',
        default=3,
        type=int,
        help='number of elements in the ranklist to consider'
    )
    parser.add_argument(
        '-ll',
        '--labels_limit',
        default=0,
        type=int,
        help='maximum numbers of labels. '
             'If labels in dataset are higher than this number, "rare" label'
    )
    parser.add_argument(
        '-ss',
        '--subset',
        default='ground_truth',
        choices=['ground_truth', PREDICTIONS],
        help='type of subset filtering'
    )
    parser.add_argument(
        '-n',
        '--normalize',
        action='store_true',
        default=False,
        help='normalize rows in confusion matrix'
    )
    parser.add_argument(
        '-m',
        '--metrics',
        default=['f1'],
        nargs='+',
        type=str,
        help='metrics to dispay in threshold_vs_metric'
    )
    parser.add_argument(
        '-pl',
        '--positive_label',
        type=int,
        default=1,
        help='label of the positive class for the roc curve'
    )
    parser.add_argument(
        '-l',
        '--logging_level',
        default='info',
        help='the level of logging to use',
        choices=['critical', 'error', 'warning', 'info', 'debug', 'notset']
    )

    args = parser.parse_args(sys_argv)

    logging.basicConfig(
        stream=sys.stdout,
        # filename='log.log', TODO - remove these?
        # filemode='w',
        level=logging_level_registry[args.logging_level],
        format='%(message)s'
    )

    if args.visualization == 'compare_classifiers_performance':
        compare_classifiers_performance(**vars(args))
    elif args.visualization == 'compare_classifiers_performance_from_prob':
        compare_classifiers_performance_from_prob(**vars(args))
    elif args.visualization == 'compare_classifiers_performance_from_pred':
        compare_classifiers_performance_from_pred(**vars(args))
    elif args.visualization == 'compare_classifiers_performance_subset':
        compare_classifiers_performance_subset(**vars(args))
    elif args.visualization == 'compare_classifiers_performance_changing_k':
        compare_classifiers_performance_changing_k(**vars(args))
    elif args.visualization == 'compare_classifiers_predictions':
        compare_classifiers_predictions(**vars(args))
    elif args.visualization == 'compare_classifiers_predictions_distribution':
        compare_classifiers_predictions_distribution(**vars(args))
    elif args.visualization == 'confidence_filtering':
        confidence_filtering(**vars(args))
    elif args.visualization == 'confidence_filtering_2d':
        confidence_filtering_2d(**vars(args))
    elif args.visualization == 'confidence_filtering_data_vs_acc':
        confidence_filtering_data_vs_acc(**vars(args))
    elif args.visualization == 'confidence_filtering_data_vs_acc_2d':
        confidence_filtering_data_vs_acc_2d(**vars(args))
    elif args.visualization == 'confidence_filtering_data_vs_acc_subset':
        confidence_filtering_data_vs_acc_subset(**vars(args))
    elif (args.visualization ==
          'confidence_filtering_data_vs_acc_subset_per_class'):
        confidence_filtering_data_vs_acc_subset_per_class(**vars(args))
    elif args.visualization == 'binary_threshold_vs_metric':
        binary_threshold_vs_metric(**vars(args))
    elif args.visualization == 'roc_curves':
        roc_curves(**vars(args))
    elif args.visualization == 'roc_curves_from_test_stats':
        roc_curves_from_test_stats(**vars(args))
    elif args.visualization == 'data_vs_acc_subset':
        data_vs_acc_subset(**vars(args))
    elif args.visualization == 'data_vs_acc_subset_per_class':
        data_vs_acc_subset_per_class(**vars(args))
    elif args.visualization == 'calibration_1_vs_all':
        calibration_1_vs_all(**vars(args))
    elif args.visualization == 'calibration_multiclass':
        calibration_multiclass(**vars(args))
    elif args.visualization == 'confusion_matrix':
        confusion_matrix(**vars(args))
    elif args.visualization == 'multiclass_multimetric':
        multiclass_multimetric(**vars(args))
    elif args.visualization == 'frequency_vs_f1':
        frequency_vs_f1(**vars(args))
    elif args.visualization == 'learning_curves':
        learning_curves(**vars(args))
    else:
        logging.info('Visualization argument not recognized')


if __name__ == '__main__':
    cli(sys.argv[1:])
