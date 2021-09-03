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
from abc import abstractmethod

import numpy as np

from ludwig.constants import *
from ludwig.constants import PREDICTIONS
from ludwig.modules.loss_modules import (
    #     BWCEWLoss,
    #     SequenceSoftmaxCrossEntropyLoss,
    #     SequenceSampledSoftmaxCrossEntropyLoss,
    #     SigmoidCrossEntropyLoss,
    #     SoftmaxCrossEntropyLoss,
    #     SampledSoftmaxCrossEntropyLoss,
    rmspe_loss, SoftmaxCrossEntropyLoss,
)
import torch
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    Metric,
    AUROC, Accuracy, AverageMeter,
)
#from ludwig.utils.tf_utils import sequence_length_2D, to_sparse

metrics = {
    ACCURACY,
    TOKEN_ACCURACY,
    HITS_AT_K,
    R2,
    JACCARD,
    EDIT_DISTANCE,
    MEAN_SQUARED_ERROR,
    MEAN_ABSOLUTE_ERROR,
    PERPLEXITY,
    ROC_AUC,
    ROOT_MEAN_SQUARED_ERROR,
    ROOT_MEAN_SQUARED_PERCENTAGE_ERROR,
}

max_metrics = {ACCURACY, TOKEN_ACCURACY, HITS_AT_K, R2, JACCARD}
min_metrics = {
    EDIT_DISTANCE,
    MEAN_SQUARED_ERROR,
    MEAN_ABSOLUTE_ERROR,
    LOSS,
    ROC_AUC,
    PERPLEXITY,
    ROOT_MEAN_SQUARED_ERROR,
    ROOT_MEAN_SQUARED_PERCENTAGE_ERROR,
}


class RMSEMetric(MeanSquaredError):
    def __init__(self, name="MSE", **kwargs):
        super().__init__(squared=False, **kwargs)

    def update(self, preds, target):
        super().update(
            preds[PREDICTIONS].detach(), target
        )


class ROCAUCMetric(AUROC):
    def __init__(self, curve="ROC", name="roc_auc"):
        super().__init__()

    def update(self, preds, target):
        super().update(
            preds[PREDICTIONS].detach(), target
        )


class MeanMetric(Metric):
    def __init__(self):
        super().__init__()
        self.avg = AverageMeter()

    def update(self, preds, target):
        self.avg.update(self.get_current_value(preds, target))

    def compute(self):
        return self.avg.compute()

    @abstractmethod
    def get_current_value(self, preds, target):
        raise NotImplementedError()


class RMSPEMetric(MeanMetric):
    def get_current_value(self, preds, target):
        if isinstance(preds, dict) and PREDICTIONS in preds:
            preds = preds[PREDICTIONS]
        return rmspe_loss(target, preds)


# class LudwigMetric:
#     def __init__(self, name):
#         self.variables = []
#         self.name = name
#
#     def reset_states(self, *args, **kwargs):
#         for var in variables:
#             var = 0
#
#     def add_tensor(self, size=1, name=None, dtype=None):
#         tensor = torch.zeros(size, dtype=dtype)
#         self.variables.append(tensor)
#         return tensor
#
#     @abstractmethod
#     def update_state(self, *args, **kwargs):
#         pass
#
#     @abstractmethod
#     def result(self):
#         pass


class R2Score(Metric):#(tf.keras.metrics.Metric):
    def __init__(self, name='r2_score'):
        #super().__init__(name=name)
        super().__init__()
        '''
        self.sum_y = self.add_weight(
            "sum_y", initializer="zeros", dtype=tf.float32
        )
        self.sum_y_squared = self.add_weight(
            "sum_y_squared", initializer="zeros", dtype=tf.float32
        )
        self.sum_y_hat = self.add_weight(
            "sum_y_hat", initializer="zeros", dtype=tf.float32
        )
        self.sum_y_hat_squared = self.add_weight(
            "sum_y_hat_squared", initializer="zeros", dtype=tf.float32
        )
        self.sum_y_hat = self.add_weight(
            "sum_y_y_hat", initializer="zeros", dtype=tf.float32
        )
        self.sum_y_y_hat = self.add_weight(
            "sum_y_y_hat", initializer="zeros", dtype=tf.float32
        )
        self.N = self.add_weight(
            'N', initializer='zeros',
            dtype=tf.float32
        )
        '''
        self.add_state("sum_y", default=torch.tensor(0, dtype=torch.float32))
        self.add_state("sum_y_squared", default=torch.tensor(0, dtype=torch.float32))
        self.add_state("sum_y_hat", default=torch.tensor(0, dtype=torch.float32))
        self.add_state("sum_y_hat_squared", default=torch.tensor(0, dtype=torch.float32))
        self.add_state("sum_y_y_hat", default=torch.tensor(0, dtype=torch.float32))
        self.add_state("N", default=torch.tensor(0, dtype=torch.float32))

    def update(self, y_hat, y):
        '''
        y = tf.cast(y, dtype=tf.float32)
        y_hat = tf.cast(y_hat, dtype=tf.float32)
        self.sum_y.assign_add(tf.reduce_sum(y))
        self.sum_y_squared.assign_add(tf.reduce_sum(y ** 2))
        self.sum_y_hat.assign_add(tf.reduce_sum(y_hat))
        self.sum_y_hat_squared.assign_add(tf.reduce_sum(y_hat ** 2))
        self.sum_y_y_hat.assign_add(tf.reduce_sum(y * y_hat))
        self.N.assign_add(y.shape[0])
        '''
        y = y.type(torch.float32)
        y_hat = y_hat.type(torch.float32)
        self.sum_y += torch.sum(y)
        self.sum_y_squared += torch.sum(y**2)
        self.sum_y_hat += torch.sum(y_hat)
        self.sum_y_hat_squared += torch.sum(y_hat ** 2)
        self.sum_y_y_hat += torch.sum(y * y_hat)
        self.N += y.shape[0]

    def compute(self):
        y_bar = self.sum_y / self.N
        tot_ss = (
            self.sum_y_squared
            - 2.0 * y_bar * self.sum_y
            + self.N * y_bar ** 2
        )
        res_ss = (
            self.sum_y_squared
            - 2.0 * self.sum_y_y_hat
            + self.sum_y_hat_squared
        )
        return 1.0 - res_ss / tot_ss


class ErrorScore(Metric):
#(tf.keras.metrics.Metric)
    def __init__(self, name='error_score'):
        #super().__init__(name=name)
        super().__init__()
        '''
        self.sum_error = self.add_weight(
            'sum_error', initializer='zeros',
            dtype=tf.float32
        )
        self.N = self.add_weight(
            'N', initializer='zeros',
            dtype=tf.float32
        )
        '''
        self.add_state("sum_error", default=torch.tensor(0, dtype=torch.float32))
        self.add_state("N", default=torch.tensor(0, dtype=torch.float32))

    def update(self, y_hat, y):
        '''
        y = tf.cast(y, tf.float32)
        y_hat = tf.cast(y_hat, tf.float32)
        self.sum_error.assign_add(tf.reduce_sum(y - y_hat))
        self.N.assign_add(y.shape[0])
        '''
        y = y.type(torch.float32)
        y_hat = y_hat.type(torch.float32)
        self.sum_error += torch.sum(y - y_hat)
        self.N += y.shape[0]

    def compute(self):
        return self.sum_error / self.N


# class BWCEWLMetric(tf.keras.metrics.Metric):
#     # Binary Weighted Cross Entropy Weighted Logits Score Metric
#     # See for additional info:
#     #   https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric
#
#     def __init__(
#             self,
#             positive_class_weight=1,
#             robust_lambda=0,
#             confidence_penalty=0,
#             name="binary_cross_entropy_weighted_loss_metric",
#     ):
#         super().__init__(name=name)
#
#         self.bwcew_loss_function = BWCEWLoss(
#             positive_class_weight=positive_class_weight,
#             robust_lambda=robust_lambda,
#             confidence_penalty=confidence_penalty,
#         )
#
#         self.sum_loss = self.add_weight(
#             "sum_loss", initializer="zeros", dtype=tf.float32
#         )
#         self.N = self.add_weight("N", initializer="zeros", dtype=tf.float32)
#
#     def update_state(self, y, y_hat):
#         loss = self.bwcew_loss_function(y, y_hat)
#         self.sum_loss.assign_add(loss)
#         self.N.assign_add(1)
#
#     def result(self):
#         return self.sum_loss / self.N


class SoftmaxCrossEntropyMetric(MeanMetric):
    def __init__(self):
        super().__init__()
        self.softmax_cross_entropy_function = SoftmaxCrossEntropyLoss()

    def get_current_value(self, preds, target):
        return self.softmax_cross_entropy_function(preds, target)


# class SampledSoftmaxCrossEntropyMetric(tf.keras.metrics.Mean):
#     def __init__(
#             self,
#             decoder_obj=None,
#             num_classes=0,
#             feature_loss=None,
#             name="sampled_softmax_cross_entropy_metric",
#     ):
#         super(SampledSoftmaxCrossEntropyMetric, self).__init__(name=name)
#
#         self.metric_function = SampledSoftmaxCrossEntropyLoss(
#             decoder_obj=decoder_obj,
#             num_classes=num_classes,
#             feature_loss=feature_loss,
#         )
#
#     def update_state(self, y, y_hat):
#         super().update_state(self.metric_function(y, y_hat))
#
#
# class SigmoidCrossEntropyMetric(tf.keras.metrics.Mean):
#     def __init__(self, feature_loss=None, name="sigmoid_cross_entropy_metric"):
#         super().__init__(name=name)
#         self.sigmoid_cross_entropy_function = SigmoidCrossEntropyLoss(
#             feature_loss
#         )
#
#     def update_state(self, y, y_hat):
#         super().update_state(self.sigmoid_cross_entropy_function(y, y_hat))
#
#
# class SequenceLossMetric(tf.keras.metrics.Mean):
#     def __init__(self, from_logits=True, name=None):
#         super().__init__(name=name)
#
#         self.loss_function = SequenceSoftmaxCrossEntropyLoss(
#             from_logits=from_logits
#         )
#
#     def update_state(self, y, y_hat):
#         loss = self.loss_function(y, y_hat)
#         super().update_state(loss)
#
#
# class SequenceSampledLossMetric(tf.keras.metrics.Mean):
#     def __init__(
#             self,
#             dec_dense_layer=None,
#             dec_num_layers=None,
#             num_classes=0,
#             feature_loss=None,
#             name=None,
#     ):
#         super(SequenceSampledLossMetric, self).__init__(name=name)
#
#         self.loss_function = SequenceSampledSoftmaxCrossEntropyLoss(
#             dec_dense_layer=dec_dense_layer,
#             dec_num_layers=dec_num_layers,
#             num_classes=num_classes,
#             feature_loss=feature_loss,
#         )
#
#     def update_state(self, y, y_hat):
#         loss = self.loss_function(y, y_hat)
#         super().update_state(loss)
#
#
# class SequenceLastAccuracyMetric(tf.keras.metrics.Accuracy):
#     """
#     Sequence accuracy based on last token in the sequence
#     """
#
#     def __init__(self, name=None):
#         super().__init__(name=name)
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true = tf.cast(y_true, dtype=tf.int64)
#         targets_sequence_length = sequence_length_2D(y_true)
#         last_targets = tf.gather_nd(
#             y_true,
#             tf.stack(
#                 [
#                     tf.range(tf.shape(y_true)[0]),
#                     tf.maximum(targets_sequence_length - 1, 0),
#                 ],
#                 axis=1,
#             ),
#         )
#         super().update_state(last_targets, y_pred, sample_weight=sample_weight)
#
#
# class PerplexityMetric(tf.keras.metrics.Mean):
#     def __init__(self, name=None):
#         super().__init__(name=name)
#         self.loss_function = SequenceSoftmaxCrossEntropyLoss(from_logits=False)
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         loss = self.loss_function(y_true, y_pred)
#         super().update_state(loss)
#
#     def result(self):
#         mean = super().result()
#         return np.exp(mean)
#
#
# class EditDistanceMetric(tf.keras.metrics.Mean):
#     def __init__(self, name=None):
#         super().__init__(name=name)
#
#     def update_state(self, y_true, y_pred):
#         # y_true: shape [batch_size, sequence_size]
#         # y_pred: shape [batch_size, sequence_size]
#
#         prediction_dtype = y_pred.dtype
#         prediction_sequence_length = sequence_length_2D(y_pred)
#         y_true_tensor = tf.cast(y_true, dtype=prediction_dtype)
#         target_sequence_length = sequence_length_2D(y_true_tensor)
#         edit_distance_val, _ = edit_distance(
#             y_true_tensor,
#             target_sequence_length,
#             y_pred,
#             prediction_sequence_length,
#         )
#         super().update_state(edit_distance_val)
#
#
# class TokenAccuracyMetric(tf.keras.metrics.Mean):
#     def __init__(self, name=None):
#         super().__init__(name=name)
#
#     def update_state(self, y_true, y_pred):
#         # y_true: shape [batch_size, sequence_size]
#         # y_pred: shape [batch_size, sequence_size]
#
#         prediction_dtype = y_pred.dtype
#         y_true_tensor = tf.cast(y_true, dtype=prediction_dtype)
#         target_sequence_length = sequence_length_2D(y_true_tensor)
#         masked_corrected_preds = masked_corrected_predictions(
#             y_true_tensor, y_pred, target_sequence_length
#         )
#
#         super().update_state(masked_corrected_preds)
#
#
# class SequenceAccuracyMetric(tf.keras.metrics.Mean):
#     def __init__(self, name=None):
#         super().__init__(name=name)
#
#     def update_state(self, y_true, y_pred):
#         # y_true: shape [batch_size, sequence_size]
#         # y_pred: shape [batch_size, sequence_size]
#
#         prediction_dtype = y_pred.dtype
#         y_true_tensor = tf.cast(y_true, dtype=prediction_dtype)
#         target_sequence_length = sequence_length_2D(y_true_tensor)
#         masked_sequence_corrected_preds = (
#             masked_sequence_corrected_predictions(
#                 y_true_tensor, y_pred, target_sequence_length
#             )
#         )
#
#         super().update_state(masked_sequence_corrected_preds)


class CategoryAccuracy(Accuracy):
    def __init__(self, name=None):
        super().__init__()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # make sure y_true is tf.int64
        super().update(preds, target.type(torch.LongTensor))


class HitsAtKMetric(Accuracy):
    def __init__(self, top_k: int = 3, name: str = None):
        super().__init__(top_k=top_k)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        super().update(preds[LOGITS], target)


'''
class MAEMetric(MeanAbsoluteErrorMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(
            y_true, y_pred[PREDICTIONS], sample_weight=sample_weight
        )
class MAEMetric(LudwigMetric):
    def __init__(self, name="MAE"):
        super(MAEMetric, self).__init__(name=name)
        self.absolute_error = self.add_tensor('absolute_error', dtype=torch.float32)
        self.N = self.add_tensor('N', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = y_true.type(torch.float32)
        y_hat = y_hat.type(torch.float32)
        self.absolute_error += torch.abs(y_true - y_abs)
        self.N += y_true.shape[0]

    def result(self):
        return self.absolute_error / self.N
'''
class MAEMetric(MeanAbsoluteError):
    def __init__(self, name="MAE", **kwargs):
        super().__init__(**kwargs)

    def update(self, preds, target):
        super().update(
            preds[PREDICTIONS].detach(), target
        )

'''
class MSEMetric(MeanSquaredErrorMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(
            y_true, y_pred[PREDICTIONS], sample_weight=sample_weight
        )
class MSEMetric(LudwigMetric):
    def __init__(self, name='MSE'):
        super(MSEMetric, self).__init__(name=name)
        self.square_error = self.add_tensor('square_error', dtype=torch.float32)
        self.N = self.add_tensor('N', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = y_true.type(torch.float32)
        y_hat = y_hat.type(torch.float32)
        self.square_error += (y_true - y_abs) ** 2
        self.N += y_true.shape[0]

    def result(self):
        return self.square_error / self.N
'''
class MSEMetric(MeanSquaredError):
    def __init__(self, name="MSE", **kwargs):
        super(MSEMetric, self).__init__(**kwargs)

    def update(self, preds, target):
        super().update(
            preds[PREDICTIONS].detach(), target
        )


# class JaccardMetric(tf.keras.metrics.Metric):
#     def __init__(self, name=None):
#         super().__init__(name=name)
#         self.jaccard_total = self.add_weight(
#             "jaccard_numerator", initializer="zeros", dtype=tf.float32
#         )
#         self.N = self.add_weight(
#             "jaccard_denomerator", initializer="zeros", dtype=tf.float32
#         )
#
#     def update_state(self, y_true, y_pred):
#         # notation: b is batch size and nc is number of unique elements
#         #           in the set
#         # y_true: shape [b, nc] bit-mapped set representation
#         # y_pred: shape [b, nc] bit-mapped set representation
#
#         batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
#
#         y_true_bool = tf.cast(y_true, tf.bool)
#         y_pred_bool = tf.cast(y_pred, tf.bool)
#
#         intersection = tf.reduce_sum(
#             tf.cast(tf.logical_and(y_true_bool, y_pred_bool), tf.float32),
#             axis=1,
#         )
#         union = tf.reduce_sum(
#             tf.cast(tf.logical_or(y_true_bool, y_pred_bool), tf.float32),
#             axis=1,
#         )
#
#         jaccard_index = intersection / union  # shape [b]
#
#         # update metric state tensors
#         self.jaccard_total.assign_add(tf.reduce_sum(jaccard_index))
#         self.N.assign_add(batch_size)
#
#     def result(self):
#         return self.jaccard_total / self.N


def get_improved_fun(metric):
    if metric in min_metrics:
        return lambda x, y: x < y
    else:
        return lambda x, y: x > y


def get_initial_validation_value(metric):
    if metric in min_metrics:
        return float("inf")
    else:
        return float("-inf")


def get_best_function(metric):
    if metric in min_metrics:
        return min
    else:
        return max


# def accuracy(targets, predictions, output_feature_name):
#     correct_predictions = tf.equal(
#         predictions,
#         targets,
#         name="correct_predictions_{}".format(output_feature_name),
#     )
#     accuracy = tf.reduce_mean(
#         tf.cast(correct_predictions, tf.float32),
#         name="accuracy_{}".format(output_feature_name),
#     )
#     return accuracy, correct_predictions
#
#
# def masked_corrected_predictions(
#         targets, predictions, targets_sequence_lengths
# ):
#     truncated_preds = predictions[:, : targets.shape[1]]
#     paddings = tf.stack(
#         [[0, 0], [0, tf.shape(targets)[1] - tf.shape(truncated_preds)[1]]]
#     )
#     padded_truncated_preds = tf.pad(truncated_preds, paddings, name="ptp")
#
#     correct_preds = tf.equal(padded_truncated_preds, targets)
#
#     mask = tf.sequence_mask(
#         targets_sequence_lengths, maxlen=correct_preds.shape[1], dtype=tf.int32
#     )
#
#     _, masked_correct_preds = tf.dynamic_partition(correct_preds, mask, 2)
#     masked_correct_preds = tf.cast(masked_correct_preds, dtype=tf.float32)
#
#     return masked_correct_preds
#
#
# def masked_sequence_corrected_predictions(
#         targets, predictions, targets_sequence_lengths
# ):
#     truncated_preds = predictions[:, : targets.shape[1]]
#     paddings = tf.stack(
#         [[0, 0], [0, tf.shape(targets)[1] - tf.shape(truncated_preds)[1]]]
#     )
#     padded_truncated_preds = tf.pad(truncated_preds, paddings, name="ptp")
#
#     correct_preds = tf.equal(padded_truncated_preds, targets)
#
#     mask = tf.sequence_mask(
#         targets_sequence_lengths, maxlen=correct_preds.shape[1], dtype=tf.int32
#     )
#
#     one_masked_correct_prediction = (
#         1.0
#         - tf.cast(mask, tf.float32)
#         + (tf.cast(mask, tf.float32) * tf.cast(correct_preds, tf.float32))
#     )
#     sequence_correct_preds = tf.reduce_prod(
#         one_masked_correct_prediction, axis=-1
#     )
#
#     return sequence_correct_preds
#
#
# def hits_at_k(targets, predictions_logits, top_k, output_feature_name):
#     with tf.device("/cpu:0"):
#         hits_at_k = tf.nn.in_top_k(
#             predictions_logits,
#             targets,
#             top_k,
#             name="hits_at_k_{}".format(output_feature_name),
#         )
#         mean_hits_at_k = tf.reduce_mean(
#             tf.cast(hits_at_k, tf.float32),
#             name="mean_hits_at_k_{}".format(output_feature_name),
#         )
#     return hits_at_k, mean_hits_at_k
#
#
# def edit_distance(
#         targets, target_seq_length, predictions_sequence,
#         predictions_seq_length
# ):
#     predicts = to_sparse(
#         predictions_sequence,
#         predictions_seq_length,
#         tf.shape(predictions_sequence)[1],
#     )
#     labels = to_sparse(targets, target_seq_length, tf.shape(targets)[1])
#     edit_distance = tf.edit_distance(predicts, labels)
#     mean_edit_distance = tf.reduce_mean(edit_distance)
#     return edit_distance, mean_edit_distance
#
#
# def perplexity(cross_entropy_loss):
#     # This seem weird but is correct:
#     # we are returning the cross entropy loss as it will be later summed,
#     # divided by the size of the dataset and finally exponentiated,
#     # because perplexity has a avg_exp aggregation strategy
#     # in the output config in SequenceOutputFeature.
#     # This implies that in Model update_output_stats_batch()
#     # the values read from the perplexity node will be summed
#     # and in Model update_output_stats() they will be divided
#     # by the set size first and exponentiated.
#     return cross_entropy_loss
#
#
# def error(targets, predictions, output_feature_name):
#     # return tf.get_variable('error_{}'.format(output_feature_name), initializer=tf.subtract(targets, predictions))
#     return tf.subtract(
#         targets, predictions, name="error_{}".format(output_feature_name)
#     )
#
#
# def absolute_error(targets, predictions, output_feature_name):
#     # error = tf.get_variable('error_{}'.format(output_feature_name), initializer=tf.subtract(targets, predictions))
#     error = tf.subtract(targets, predictions)
#     return tf.abs(error, name="absolute_error_{}".format(output_feature_name))
#
#
# def squared_error(targets, predictions, output_feature_name):
#     # error = tf.get_variable('error_{}'.format(output_feature_name), initializer=tf.subtract(targets, predictions))
#     error = tf.subtract(targets, predictions)
#     return tf.pow(
#         error, 2, name="squared_error_{}".format(output_feature_name)
#     )
#
#
# def r2(targets, predictions, output_feature_name):
#     y_bar = tf.reduce_mean(targets)
#     tot_ss = tf.reduce_sum(tf.pow(targets - y_bar, 2))
#     res_ss = tf.reduce_sum(tf.pow(targets - predictions, 2))
#     r2 = tf.subtract(
#         1.0, res_ss / tot_ss, name="r2_{}".format(output_feature_name)
#     )
#     return r2
