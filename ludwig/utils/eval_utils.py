#! /usr/bin/env python
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
from collections import OrderedDict

import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


class ConfusionMatrix:
    def __init__(self, conditions, predictions, labels=None, sample_weight=None):
        # assert (len(predictions) == len(conditions))
        min_length = min(len(predictions), len(conditions))
        self.predictions = predictions[:min_length]
        self.conditions = conditions[:min_length]

        if labels is not None:
            self.label2idx = {label: idx for idx, label in enumerate(labels)}
            self.idx2label = {idx: label for idx, label in enumerate(labels)}
            labels = list(range(len(labels)))
        else:
            self.label2idx = {
                str(label): idx for idx, label in enumerate(np.unique([self.predictions, self.conditions]))
            }
            self.idx2label = {
                idx: str(label) for idx, label in enumerate(np.unique([self.predictions, self.conditions]))
            }
        self.cm = confusion_matrix(self.conditions, self.predictions, labels=labels, sample_weight=sample_weight)

        # if labels is not None:
        #     self.labels_dict = {label: idx for idx, label in enumerate(labels)}
        # else:
        #     if conditions.dtype.char == 'S':  # it's an array of strings
        #         self.labels_dict = {str(label): idx for idx, label in
        #                             enumerate(np.unique([predictions, conditions]))}
        #     else:  # number
        #         max_label = np.concatenate([predictions, conditions]).max()
        #         self.labels_dict = {str(i): i for i in range(max_label + 1)}
        #         labels = [str(i) for i in range(max_label + 1)]
        # self.cm = confusion_matrix(conditions, predictions, labels, sample_weight)

        self.sum_predictions = np.sum(self.cm, axis=0)
        self.sum_conditions = np.sum(self.cm, axis=1)
        self.all = np.sum(self.cm)

    def label_to_idx(self, label):
        return self.label2idx[label]

    def true_positives(self, idx):
        return self.cm[idx, idx]

    def true_negatives(self, idx):
        return self.all - self.sum_predictions[idx] - self.sum_conditions[idx] + self.true_positives(idx)

    def false_positives(self, idx):
        return self.sum_predictions[idx] - self.true_positives(idx)

    def false_negatives(self, idx):
        return self.sum_conditions[idx] - self.true_positives(idx)

    def true_positive_rate(self, idx):
        nom = self.true_positives(idx)
        den = self.sum_conditions[idx]
        if den == 0 or den == np.nan:
            return 0
        else:
            return nom / den

    def true_negative_rate(self, idx):
        nom = tn = self.true_negatives(idx)
        den = tn + self.false_positives(idx)
        if den == 0 or den == np.nan:
            return 0
        else:
            return nom / den

    def positive_predictive_value(self, idx):
        nom = self.true_positives(idx)
        den = self.sum_predictions[idx]
        if den == 0 or den == np.nan:
            return 0
        else:
            return nom / den

    def negative_predictive_value(self, idx):
        nom = tn = self.true_negatives(idx)
        den = tn + self.false_negatives(idx)
        if den == 0 or den == np.nan:
            return 0
        else:
            return nom / den

    def false_negative_rate(self, idx):
        return 1.0 - self.true_positive_rate(idx)

    def false_positive_rate(self, idx):
        return 1.0 - self.true_negative_rate(idx)

    def false_discovery_rate(self, idx):
        return 1.0 - self.positive_predictive_value(idx)

    def false_omission_rate(self, idx):
        return 1.0 - self.negative_predictive_value(idx)

    def accuracy(self, idx):
        nom = self.true_positives(idx) + self.true_negatives(idx)
        den = self.all
        if den == 0 or den == np.nan:
            return 0
        else:
            return nom / den

    def precision(self, idx):
        return self.positive_predictive_value(idx)

    def recall(self, idx):
        return self.true_positive_rate(idx)

    def fbeta_score(self, beta, idx):
        beta_2 = np.power(beta, 2)
        precision = self.precision(idx)
        recall = self.recall(idx)
        nom = (1 + beta_2) * precision * recall
        den = (beta_2 * precision) + recall
        if den == 0 or den == np.nan:
            return 0
        else:
            return nom / den

    def f1_score(self, idx):
        return self.fbeta_score(1, idx)

    def sensitivity(self, idx):
        return self.true_positive_rate(idx)

    def specificity(self, idx):
        return self.true_negative_rate(idx)

    def hit_rate(self, idx):
        return self.true_positive_rate(idx)

    def miss_rate(self, idx):
        return self.false_negative_rate(idx)

    def fall_out(self, idx):
        return self.false_positive_rate(idx)

    def matthews_correlation_coefficient(self, idx):
        tp = self.true_positives(idx)
        tn = self.true_negatives(idx)
        fp = self.false_positives(idx)
        fn = self.false_negatives(idx)
        nom = tp * tn - fp * fn
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if den == 0 or den == np.nan:
            return 0
        else:
            return nom / den

    def informedness(self, idx):
        return self.true_positive_rate(idx) + self.true_negative_rate(idx) - 1

    def markedness(self, idx):
        return self.positive_predictive_value(idx) + self.negative_predictive_value(idx) - 1

    def token_accuracy(self):
        return metrics.accuracy_score(self.conditions, self.predictions)

    def avg_precision(self, average="macro"):
        return metrics.precision_score(self.conditions, self.predictions, average=average)

    def avg_recall(self, average="macro"):
        return metrics.recall_score(self.conditions, self.predictions, average=average)

    def avg_f1_score(self, average="macro"):
        return metrics.f1_score(self.conditions, self.predictions, average=average)

    def avg_fbeta_score(self, beta, average="macro"):
        return metrics.fbeta_score(self.conditions, self.predictions, beta=beta, average=average)

    def kappa_score(self):
        return metrics.cohen_kappa_score(self.conditions, self.predictions)

    def class_stats(self, idx):
        return {
            "true_positives": self.true_positives(idx),
            "true_negatives": self.true_negatives(idx),
            "false_positives": self.false_positives(idx),
            "false_negatives": self.false_negatives(idx),
            "true_positive_rate": self.true_positive_rate(idx),
            "true_negative_rate": self.true_negative_rate(idx),
            "positive_predictive_value": self.positive_predictive_value(idx),
            "negative_predictive_value": self.negative_predictive_value(idx),
            "false_negative_rate": self.false_negative_rate(idx),
            "false_positive_rate": self.false_positive_rate(idx),
            "false_discovery_rate": self.false_discovery_rate(idx),
            "false_omission_rate": self.false_omission_rate(idx),
            "accuracy": self.accuracy(idx),
            "precision": self.precision(idx),
            "recall": self.recall(idx),
            "f1_score": self.f1_score(idx),
            "sensitivity": self.sensitivity(idx),
            "specificity": self.specificity(idx),
            "hit_rate": self.hit_rate(idx),
            "miss_rate": self.miss_rate(idx),
            "fall_out": self.fall_out(idx),
            "matthews_correlation_coefficient": self.matthews_correlation_coefficient(idx),
            "informedness": self.informedness(idx),
            "markedness": self.markedness(idx),
        }

    def per_class_stats(self):
        stats = OrderedDict()
        for idx in sorted(self.idx2label.keys()):
            stats[self.idx2label[idx]] = self.class_stats(idx)
        return stats

    def stats(self):
        return {
            "token_accuracy": self.token_accuracy(),
            "avg_precision_macro": self.avg_precision(average="macro"),
            "avg_recall_macro": self.avg_recall(average="macro"),
            "avg_f1_score_macro": self.avg_f1_score(average="macro"),
            "avg_precision_micro": self.avg_precision(average="micro"),
            "avg_recall_micro": self.avg_recall(average="micro"),
            "avg_f1_score_micro": self.avg_f1_score(average="micro"),
            "avg_precision_weighted": self.avg_precision(average="micro"),
            "avg_recall_weighted": self.avg_recall(average="micro"),
            "avg_f1_score_weighted": self.avg_f1_score(average="weighted"),
            "kappa_score": self.kappa_score(),
        }


def roc_curve(conditions, prediction_scores, pos_label=None, sample_weight=None):
    return metrics.roc_curve(conditions, prediction_scores, pos_label=pos_label, sample_weight=sample_weight)


def roc_auc_score(conditions, prediction_scores, average="micro", sample_weight=None):
    try:
        return metrics.roc_auc_score(conditions, prediction_scores, average=average, sample_weight=sample_weight)
    except ValueError as ve:
        logger.info(ve)


def precision_recall_curve(conditions, prediction_scores, pos_label=None, sample_weight=None):
    return metrics.precision_recall_curve(
        conditions, prediction_scores, pos_label=pos_label, sample_weight=sample_weight
    )


def average_precision_score(conditions, prediction_scores, average="micro", sample_weight=None):
    # average == [micro, macro, sampled, weidhted]
    return metrics.average_precision_score(conditions, prediction_scores, average=average, sample_weight=sample_weight)
