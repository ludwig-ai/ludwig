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
from abc import abstractmethod, ABC
from typing import Callable, Optional, Union, List

import torch
import torchmetrics.functional as metrics_F
from torch import Tensor
from torchmetrics import AUROC, IoU, MeanAbsoluteError,\
    MeanSquaredError, Metric, Accuracy as _Accuracy,\
    R2Score as _R2Score, MeanMetric as _MeanMetric

from ludwig.constants import *
from ludwig.modules.loss_modules import BWCEWLoss, SigmoidCrossEntropyLoss,\
    SoftmaxCrossEntropyLoss # SequenceSoftmaxCrossEntropyLoss,\
    # SequenceSampledSoftmaxCrossEntropyLoss, SampledSoftmaxCrossEntropyLoss
from ludwig.utils.loss_utils import rmspe_loss
from ludwig.utils.metric_utils import masked_correct_predictions
from ludwig.utils.registry import Registry
from ludwig.utils.torch_utils import sequence_length_2D
# from ludwig.utils.metric_utils import masked_sequence_corrected_predictions,\
    # edit_distance
# from ludwig.utils.tf_utils import to_sparse


metric_feature_registry = Registry()
metric_registry = Registry()


def register_metric(name: str, features: Union[str, List[str]]):
    if isinstance(features, str):
        features = [features]

    def wrap(cls):
        for feature in features:
            feature_registry = metric_feature_registry.get(feature, {})
            feature_registry[name] = cls
            metric_feature_registry[feature] = feature_registry
        metric_registry[name] = cls
        return cls
    return wrap


def get_metric_classes(feature: str):
    return metric_feature_registry[feature]


def get_metric_cls(feature: str, name: str):
    return metric_feature_registry[feature][name]


class LudwigMetric(ABC): 
    @classmethod
    def can_report(cls, feature: "OutputFeature") -> bool:
        return True

    @classmethod
    @abstractmethod
    def get_objective(cls):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_inputs(cls):
        raise NotImplementedError()


@register_metric(ROOT_MEAN_SQUARED_ERROR, [NUMERICAL])
class RMSEMetric(MeanSquaredError, LudwigMetric):
    """ Root mean squared error metric. """
    def __init__(self, **kwargs):
        super().__init__(squared=False, **kwargs)

    @classmethod
    def get_objective(cls):
        return MINIMIZE

    @classmethod
    def get_inputs(cls):
        return PREDICTIONS


@register_metric(ROC_AUC, [BINARY])
class ROCAUCMetric(AUROC, LudwigMetric):
    def __init__(self, **kwargs):
        super().__init__()

    """ Metric for area under ROC curve. """
    def update(self, preds: Tensor, target: Tensor) -> None:
        # Currently only supported for binary tasks.
        if preds.ndim > 1 or target.ndim > 1:
            raise RuntimeError(
                f'Only binary tasks supported, but received input of '
                f'{max(preds.ndim, target.ndim)} dimensions while expecting'
                f'1-dimensional input.')
        return super().update(preds, target)

    @classmethod
    def get_objective(cls):
        return MINIMIZE

    @classmethod
    def get_inputs(cls):
        return PROBABILITIES


class MeanMetric(Metric, LudwigMetric):
    """ Abstract class for computing mean of metrics. """
    def __init__(self, **kwargs):
        super().__init__()
        self.avg = _MeanMetric()

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.avg.update(self.get_current_value(preds, target))

    def compute(self) -> Tensor:
        return self.avg.compute()[0]

    @abstractmethod
    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError()


@register_metric(ROOT_MEAN_SQUARED_PERCENTAGE_ERROR, [NUMERICAL])
class RMSPEMetric(MeanMetric):
    def __init__(self, **kwargs):
        super().__init__()

    """ Root mean squared percentage error metric. """
    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        return rmspe_loss(target, preds)

    @classmethod
    def get_objective(cls):
        return MINIMIZE

    @classmethod
    def get_inputs(cls):
        return PREDICTIONS


# TODO(shreya): Double check difference in computation.
@register_metric(R2, [NUMERICAL, VECTOR])
class R2Score(_R2Score, LudwigMetric):
    """ R-squared metric. """
    def __init__(self, num_outputs: int = 1, **kwargs):
        super().__init__(num_outputs=num_outputs)

    @classmethod
    def get_objective(cls):
        return MAXIMIZE

    @classmethod
    def get_inputs(cls):
        return PREDICTIONS


@register_metric(LOSS, [])
class LossMetric(MeanMetric, ABC):
    @abstractmethod
    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError()

    @classmethod
    def get_objective(cls):
        return MINIMIZE

    @classmethod
    def get_inputs(cls):
        return LOGITS

    @classmethod
    def can_report(cls, feature: "OutputFeature") -> bool:
        return False


@register_metric('binary_weighted_cross_entropy', [BINARY])
class BWCEWLMetric(LossMetric):
    """ Binary Weighted Cross Entropy Weighted Logits Score Metric. """

    def __init__(
            self,
            positive_class_weight: Optional[Tensor] = None,
            robust_lambda: int = 0,
            confidence_penalty: int = 0,
            **kwargs
    ):
        super().__init__()

        self.loss_function = BWCEWLoss(
            positive_class_weight=positive_class_weight,
            robust_lambda=robust_lambda,
            confidence_penalty=confidence_penalty,
        )

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        return self.loss_function(preds, target)


@register_metric('softmax_cross_entropy', [CATEGORY])
class SoftmaxCrossEntropyMetric(LossMetric):
    def __init__(self, **kwargs):
        super().__init__()
        self.softmax_cross_entropy_function = SoftmaxCrossEntropyLoss(**kwargs)

    def get_current_value(self, preds: Tensor, target: Tensor):
        return self.softmax_cross_entropy_function(preds, target)


# @register_metric('sampled_softmax_cross_entropy', [CATEGORY])
# class SampledSoftmaxCrossEntropyMetric(tf.keras.metrics.Mean):
#     def __init__(
#             self,
#             decoder_obj=None,
#             num_classes=0,
#             feature_loss=None,
#             name="sampled_softmax_cross_entropy_metric",
#             **kwargs
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
#     @classmethod
#     def get_objective(cls):
#         return MINIMIZE
#
#     @classmethod
#     def get_inputs(cls):
#         return LOGITS


@register_metric('sigmoid_cross_entropy', [SET])
class SigmoidCrossEntropyMetric(LossMetric):
    def __init__(self, **kwargs):
        super().__init__()
        self.sigmoid_cross_entropy_function = SigmoidCrossEntropyLoss(**kwargs)

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        return self.sigmoid_cross_entropy_function(preds, target)


# TODO(shreya): After Sequence Losses
# class SequenceLossMetric(tf.keras.metrics.Mean):
#     def __init__(self, from_logits=True, name=None, **kwargs):
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
#             **kwargs
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
#     def __init__(self, name=None, **kwargs):
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
#     def __init__(self, name=None, **kwargs):
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
#     def __init__(self, name=None, **kwargs):
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


@register_metric(TOKEN_ACCURACY, [SEQUENCE, TEXT])
class TokenAccuracyMetric(MeanMetric):
    def __init__(self, **kwargs):
        super().__init__()

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        target = target.type(preds.dtype)
        target_sequence_length = sequence_length_2D(target)
        masked_correct_preds = masked_correct_predictions(
            target, preds, target_sequence_length)
        return torch.mean(masked_correct_preds)

    @classmethod
    def get_objective(cls):
        return MAXIMIZE

    @classmethod
    def get_inputs(cls):
        # TODO: double check
        return PREDICTIONS


# TODO(shreya): After Sequence Losses
# class SequenceAccuracyMetric(tf.keras.metrics.Mean):
#     def __init__(self, name=None, **kwargs):
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


@register_metric(ACCURACY, [BINARY])
class Accuracy(_Accuracy, LudwigMetric):
    """ R-squared metric. """
    def __init__(self, **kwargs):
        super().__init__()

    @classmethod
    def get_objective(cls):
        return MAXIMIZE

    @classmethod
    def get_inputs(cls):
        return PREDICTIONS


@register_metric(ACCURACY, [CATEGORY])
class CategoryAccuracy(_Accuracy, LudwigMetric):
    def __init__(self, **kwargs):
        super().__init__()

    def update(self, preds: Tensor, target: Tensor) -> None:
        # make sure y_true is tf.int64
        super().update(preds, target.type(torch.LongTensor))

    @classmethod
    def get_objective(cls):
        return MAXIMIZE

    @classmethod
    def get_inputs(cls):
        # TODO: double check
        return PREDICTIONS


@register_metric(HITS_AT_K, [CATEGORY])
class HitsAtKMetric(_Accuracy, LudwigMetric):
    def __init__(self, top_k: int = 3, **kwargs):
        super().__init__(top_k=top_k)

    def update(self, preds: Tensor, target: Tensor) -> None:
        super().update(preds, target)

    @classmethod
    def get_objective(cls):
        return MAXIMIZE

    @classmethod
    def get_inputs(cls):
        return LOGITS

    @classmethod
    def can_report(cls, feature: "OutputFeature") -> bool:
        return feature.decoder_obj.num_classes > feature.top_k


@register_metric(MEAN_ABSOLUTE_ERROR, [NUMERICAL, VECTOR])
class MAEMetric(MeanAbsoluteError, LudwigMetric):
    def __init__(self, **kwargs):
        super().__init__()

    def update(self, preds: Tensor, target: Tensor) -> None:
        super().update(preds.detach(), target)

    @classmethod
    def get_objective(cls):
        return MINIMIZE

    @classmethod
    def get_inputs(cls):
        return PREDICTIONS


@register_metric(MEAN_SQUARED_ERROR, [NUMERICAL, VECTOR])
class MSEMetric(MeanSquaredError, LudwigMetric):
    def __init__(self, **kwargs):
        super(MSEMetric, self).__init__()

    def update(self, preds: Tensor, target: Tensor) -> None:
        super().update(preds.detach(), target)

    @classmethod
    def get_objective(cls):
        return MINIMIZE

    @classmethod
    def get_inputs(cls):
        return PREDICTIONS


@register_metric(JACCARD, [SET])
class JaccardMetric(MeanMetric):
    def __init__(self, **kwargs):
        super().__init__()
        self.jaccard_metric = IoU(num_classes=2, reduction='sum')
        self.add_state(name='loss', default=[], dist_reduce_fx='mean')

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        return self.jaccard_metric(
            preds.type(torch.bool), target.type(torch.bool)
        )

    @classmethod
    def get_objective(cls):
        return MAXIMIZE

    @classmethod
    def get_inputs(cls):
        return PREDICTIONS


def get_improved_fun(metric: str) -> Callable:
    if metric_registry[metric].get_objective() == MINIMIZE:
        return lambda x, y: x < y
    else:
        return lambda x, y: x > y


def get_initial_validation_value(metric: str) -> float:
    if metric_registry[metric].get_objective() == MINIMIZE:
        return float("inf")
    else:
        return float("-inf")


def get_best_function(metric: str) -> Callable:
    if metric_registry[metric].get_objective() == MINIMIZE:
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
