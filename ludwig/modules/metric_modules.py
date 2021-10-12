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
from typing import Callable, Optional

import torch
import torchmetrics.functional as metrics_F
from torch import Tensor
from torchmetrics import Accuracy, AUROC, AverageMeter, IoU, MeanAbsoluteError,\
    MeanSquaredError, Metric, R2Score as _R2Score

from ludwig.constants import *
from ludwig.modules.loss_modules import BWCEWLoss, SigmoidCrossEntropyLoss,\
    SoftmaxCrossEntropyLoss # SequenceSoftmaxCrossEntropyLoss,\
    # SequenceSampledSoftmaxCrossEntropyLoss, SampledSoftmaxCrossEntropyLoss
from ludwig.utils.loss_utils import rmspe_loss
from ludwig.utils.metric_utils import masked_corrected_predictions
from ludwig.utils.torch_utils import sequence_length_2D
# from ludwig.utils.metric_utils import masked_sequence_corrected_predictions,\
    # edit_distance
# from ludwig.utils.tf_utils import to_sparse

# TODO(shreya): See if these can be made into class properties.
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
max_metrics = {
    ACCURACY,
    TOKEN_ACCURACY,
    HITS_AT_K,
    R2,
    JACCARD
}
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
    """ Root mean squared error metric. """
    def __init__(self, **kwargs):
        super().__init__(squared=False, **kwargs)


class ROCAUCMetric(AUROC):
    """ Metric for area under ROC curve. """
    def update(self, preds: Tensor, target: Tensor) -> None:
        # Currently only supported for binary tasks.
        if preds.ndim > 1 or target.ndim > 1:
            raise RuntimeError(
                f'Only binary tasks supported, but received input of '
                f'{max(preds.ndim, target.ndim)} dimensions while expecting'
                f'1-dimensional input.')
        return super().update(preds, target)


class MeanMetric(Metric):
    """ Abstract class for computing mean of metrics. """
    def __init__(self):
        super().__init__()
        self.avg = AverageMeter()

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.avg.update(self.get_current_value(preds, target))

    def compute(self) -> Tensor:
        return self.avg.compute()

    @abstractmethod
    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError()


class RMSPEMetric(MeanMetric):
    """ Root mean squared percentage error metric. """
    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        return rmspe_loss(target, preds)


# TODO(shreya): Double check difference in computation.
class R2Score(_R2Score):
    """ R-squared metric. """
    pass


class BWCEWLMetric(MeanMetric):
    """ Binary Weighted Cross Entropy Weighted Logits Score Metric. """

    def __init__(
            self,
            positive_class_weight: Optional[Tensor] = None,
            robust_lambda: int = 0,
            confidence_penalty: int = 0
    ):
        super().__init__()

        self.loss_function = BWCEWLoss(
            positive_class_weight=positive_class_weight,
            robust_lambda=robust_lambda,
            confidence_penalty=confidence_penalty,
        )

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        return self.loss_function(preds, target)


class SoftmaxCrossEntropyMetric(MeanMetric):
    def __init__(self, **kwargs):
        super().__init__()
        self.softmax_cross_entropy_function = SoftmaxCrossEntropyLoss(**kwargs)

    def get_current_value(self, preds: Tensor, target: Tensor):
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
class SigmoidCrossEntropyMetric(MeanMetric):
    def __init__(self, **kwargs):
        super().__init__()
        self.sigmoid_cross_entropy_function = SigmoidCrossEntropyLoss(**kwargs)

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        return self.sigmoid_cross_entropy_function(preds, target)


# TODO(shreya): After Sequence Losses
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
# TODO(shreya): After Sequence Losses
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
# TODO(shreya): After Sequence Losses
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
# TODO(shreya): After Sequence Losses
# class PerplexityMetric(tf.keras.metrics.Mean):
#     def __init__(self, name=None):
#         super().__init__(name=name)
#         self.loss_function = SequenceSoftmaxCrossEntropyLoss(from_logits=False)

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         loss = self.loss_function(y_true, y_pred)
#         super().update_state(loss)

#     def result(self):
#         mean = super().result()
#         return np.exp(mean)


# TODO(shreya): No PyTorch CUDA implementation available
# class EditDistanceMetric(tf.keras.metrics.Mean):
#     def __init__(self, name=None):
#         super().__init__(name=name)

#     def update_state(self, y_true, y_pred):
#         # y_true: shape [batch_size, sequence_size]
#         # y_pred: shape [batch_size, sequence_size]

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


class TokenAccuracyMetric(MeanMetric):
    def __init__(self):
        super().__init__()

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        target = target.type(preds.dtype)
        target_sequence_length = sequence_length_2D(target)
        masked_corrected_preds = masked_corrected_predictions(
            target, preds, target_sequence_length)
        return torch.mean(masked_corrected_preds)


# TODO(shreya): After Sequence Losses
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
    def __init__(self):
        super().__init__()

    def update(self, preds: Tensor, target: Tensor) -> None:
        # make sure y_true is tf.int64
        super().update(preds, target.type(torch.LongTensor))


class HitsAtKMetric(Accuracy):
    def __init__(self, top_k: int = 3):
        super().__init__(top_k=top_k)

    def update(self, preds: Tensor, target: Tensor) -> None:
        super().update(preds, target)


class MAEMetric(MeanAbsoluteError):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, preds: Tensor, target: Tensor) -> None:
        super().update(preds.detach(), target)


class MSEMetric(MeanSquaredError):
    def __init__(self, **kwargs):
        super(MSEMetric, self).__init__(**kwargs)

    def update(self, preds: Tensor, target: Tensor) -> None:
        super().update(preds.detach(), target)


class JaccardMetric(MeanMetric):
    def __init__(self, **kwargs):
        super().__init__()
        self.jaccard_metric = IoU(num_classes=2, reduction='sum', **kwargs)
        self.add_state(name='loss', default=[], dist_reduce_fx='mean')

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        return self.jaccard_metric(
            preds.type(torch.bool), target.type(torch.bool)
        )


def get_improved_fun(metric: str) -> Callable:
    if metric in min_metrics:
        return lambda x, y: x < y
    else:
        return lambda x, y: x > y


def get_initial_validation_value(metric: str) -> float:
    if metric in min_metrics:
        return float("inf")
    else:
        return float("-inf")


def get_best_function(metric: str) -> Callable:
    if metric in min_metrics:
        return min
    else:
        return max


def accuracy(preds: Tensor, target: Tensor) -> Tensor:
    """
    Returns:
        Accuracy (float tensor of shape (1,)).
    """
    return metrics_F.accuracy(preds, target)
    # correct_predictions = predictions == target
    # accuracy = torch.mean(correct_predictions.type(torch.float32))
    # return accuracy, correct_predictions


def perplexity(cross_entropy_loss):
    # This seem weird but is correct:
    # we are returning the cross entropy loss as it will be later summed,
    # divided by the size of the dataset and finally exponentiated,
    # because perplexity has a avg_exp aggregation strategy
    # in the output config in SequenceOutputFeature.
    # This implies that in Model update_output_stats_batch()
    # the values read from the perplexity node will be summed
    # and in Model update_output_stats() they will be divided
    # by the set size first and exponentiated.
    return cross_entropy_loss


def error(preds: Tensor, target: Tensor) -> Tensor:
    return target - preds


def absolute_error(preds: Tensor, target: Tensor) -> Tensor:
    return torch.abs(target - preds)


def squared_error(preds: Tensor, target: Tensor) -> Tensor:
    return (target - preds) ** 2


def r2(preds: Tensor, target: Tensor) -> Tensor:
    return metrics_F.r2_score(preds, target)


# Registry to map loss name to the desired predicted input type.
METRICS_INPUTS_REGISTRY = {
    RMSEMetric: PREDICTIONS,
    ROCAUCMetric: PROBABILITIES,
    RMSPEMetric: PREDICTIONS,
    R2Score: PREDICTIONS,
    BWCEWLMetric: LOGITS,
    SoftmaxCrossEntropyMetric: LOGITS,
    SigmoidCrossEntropyMetric: LOGITS,
    TokenAccuracyMetric: PREDICTIONS, #double check
    CategoryAccuracy: PROBABILITIES, #double check
    HitsAtKMetric: LOGITS,
    MAEMetric: PREDICTIONS,
    MSEMetric: PREDICTIONS,
    JaccardMetric: PREDICTIONS,
    Accuracy: PREDICTIONS,
}