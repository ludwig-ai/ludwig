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
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional

import torch
import torchmetrics.functional as metrics_F
from torch import Tensor
from torchmetrics import Accuracy as _Accuracy
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanMetric as _MeanMetric
from torchmetrics import MeanSquaredError, Metric
from torchmetrics import R2Score as _R2Score
from torchmetrics.metric import jit_distributed_available

from ludwig.constants import (
    ACCURACY,
    BINARY,
    CATEGORY,
    HITS_AT_K,
    JACCARD,
    LOGITS,
    LOSS,
    MAXIMIZE,
    MEAN_ABSOLUTE_ERROR,
    MEAN_SQUARED_ERROR,
    MINIMIZE,
    NUMBER,
    PREDICTIONS,
    PROBABILITIES,
    R2,
    ROC_AUC,
    ROOT_MEAN_SQUARED_ERROR,
    ROOT_MEAN_SQUARED_PERCENTAGE_ERROR,
    SEQUENCE,
    SET,
    TEXT,
    TOKEN_ACCURACY,
    VECTOR,
)
from ludwig.modules.loss_modules import (
    BWCEWLoss,
    SequenceSoftmaxCrossEntropyLoss,
    SigmoidCrossEntropyLoss,
    SoftmaxCrossEntropyLoss,
)
from ludwig.modules.metric_registry import metric_registry, register_metric
from ludwig.utils.horovod_utils import gather_all_tensors, is_distributed_available
from ludwig.utils.loss_utils import rmspe_loss
from ludwig.utils.metric_utils import masked_correct_predictions
from ludwig.utils.torch_utils import sequence_length_2D


class LudwigMetric(Metric, ABC):
    @classmethod
    def can_report(cls, feature: "OutputFeature") -> bool:  # noqa: F821
        return True

    @classmethod
    @abstractmethod
    def get_objective(cls):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_inputs(cls):
        """Returns the key of the tensor from the predictions() Dict that should be used for computing this metric.

        For example: PREDICTIONS would be used for accuracy metrics while LOGITS would be used for loss metrics.
        """
        raise NotImplementedError()

    @contextmanager
    def sync_context(
        self,
        dist_sync_fn: Optional[Callable] = None,
        process_group: Optional[Any] = None,
        should_sync: bool = True,
        should_unsync: bool = True,
        distributed_available: Optional[Callable] = jit_distributed_available,
    ) -> Generator:
        """Override the behavior of this in the base class to support Horovod."""
        self.sync(
            dist_sync_fn=gather_all_tensors,
            process_group=process_group,
            should_sync=should_sync,
            distributed_available=is_distributed_available,
        )

        yield

        self.unsync(should_unsync=self._is_synced and should_unsync)


@register_metric(ROOT_MEAN_SQUARED_ERROR, [NUMBER])
class RMSEMetric(MeanSquaredError, LudwigMetric):
    """Root mean squared error metric."""

    def __init__(self, **kwargs):
        super().__init__(squared=False, dist_sync_fn=gather_all_tensors, **kwargs)

    @classmethod
    def get_objective(cls):
        return MINIMIZE

    @classmethod
    def get_inputs(cls):
        return PREDICTIONS


@register_metric(ROC_AUC, [BINARY])
class ROCAUCMetric(LudwigMetric):
    """Fast implementation of metric for area under ROC curve."""

    def __init__(
        self,
        num_thresholds: int = 201,
        epsilon: float = 1e-7,
        **kwargs,
    ) -> None:
        super().__init__(dist_sync_fn=gather_all_tensors)
        self.num_thresholds = num_thresholds
        self.epsilon = epsilon
        self.add_state("summary_stats", torch.zeros(num_thresholds, 4), dist_reduce_fx="sum")

    def _get_thresholds(self, device, dtype) -> Tensor:
        thresholds = torch.linspace(0, 1, self.num_thresholds, device=device, dtype=dtype)
        thresholds[0] -= self.epsilon
        thresholds[-1] += self.epsilon
        return thresholds

    def update(self, preds: Tensor, target: Tensor) -> None:
        # Currently only supported for binary tasks.
        if preds.ndim > 1 or target.ndim > 1:
            raise RuntimeError(
                f"Only binary tasks supported, but received input of "
                f"{max(preds.ndim, target.ndim)} dimensions while expecting"
                f"1-dimensional input."
            )

        if torch.min(preds) < 0 or torch.max(preds) > 1:
            raise RuntimeError(
                f"Only binary tasks supported, but received predictions in range "
                f"({torch.min(preds)}, {torch.max(preds)})."
            )

        thresholds = self._get_thresholds(preds.device, preds.dtype)
        target = target.to(bool).type(preds.dtype)

        preds = preds.unsqueeze(1)
        target = target.unsqueeze(1)

        # Compute correct predictions at each threshold.
        correct_predictions = ((preds >= thresholds) == target).to(int)

        # Compute true positives, false positives, true negatives, false negatives.
        # overall_predictions is a tensor where each cell represents the type of prediction:
        # 0: false positive
        # 1: true negative
        # 2: false negative
        # 3: true positive
        overall_predictions = correct_predictions + (2 * target)

        # Sum up the number of true positives, false positives, true negatives, false negatives at each threshold.
        self.summary_stats += torch.eye(4, device=preds.device)[overall_predictions.T.long()].sum(dim=1, keepdim=False)

    def compute(self) -> Tensor:
        # Compute true positives, false positives, true negatives, false negatives.
        self.summary_stats = self.summary_stats.squeeze()
        false_positives = self.summary_stats[:, 0]
        true_negatives = self.summary_stats[:, 1]
        false_negatives = self.summary_stats[:, 2]
        true_positives = self.summary_stats[:, 3]

        true_positive_rate = true_positives / (true_positives + false_negatives)
        false_positive_rate = false_positives / (false_positives + true_negatives)

        # Compute area under ROC curve. Multiply by -1 because tpr and fpr are computed from the opposite direction.
        return -1 * torch.trapz(true_positive_rate, false_positive_rate)

    @classmethod
    def get_objective(cls):
        return MAXIMIZE

    @classmethod
    def get_inputs(cls):
        return PROBABILITIES


class MeanMetric(LudwigMetric):
    """Abstract class for computing mean of metrics."""

    def __init__(self, **kwargs):
        super().__init__(dist_sync_fn=gather_all_tensors)
        self.avg = _MeanMetric()

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.avg.update(self.get_current_value(preds, target))

    def compute(self) -> Tensor:
        return self.avg.compute()

    def reset(self):
        super().reset()
        self.avg.reset()

    @abstractmethod
    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError()


@register_metric(ROOT_MEAN_SQUARED_PERCENTAGE_ERROR, [NUMBER])
class RMSPEMetric(MeanMetric):
    def __init__(self, **kwargs):
        super().__init__(dist_sync_fn=gather_all_tensors)

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
@register_metric(R2, [NUMBER, VECTOR])
class R2Score(_R2Score, LudwigMetric):
    """R-squared metric."""

    def __init__(self, num_outputs: int = 1, **kwargs):
        super().__init__(num_outputs=num_outputs, dist_sync_fn=gather_all_tensors)

    @classmethod
    def get_objective(cls):
        return MAXIMIZE

    @classmethod
    def get_inputs(cls):
        return PREDICTIONS


@register_metric(LOSS, [])
class LossMetric(MeanMetric, ABC):
    def __init__(self):
        super().__init__()

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
    def can_report(cls, feature: "OutputFeature") -> bool:  # noqa: F821
        return False


@register_metric("binary_weighted_cross_entropy", [BINARY])
class BWCEWLMetric(LossMetric):
    """Binary Weighted Cross Entropy Weighted Logits Score Metric."""

    def __init__(
        self,
        positive_class_weight: Optional[Tensor] = None,
        robust_lambda: int = 0,
        confidence_penalty: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.loss_function = BWCEWLoss(
            positive_class_weight=positive_class_weight,
            robust_lambda=robust_lambda,
            confidence_penalty=confidence_penalty,
        )

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        return self.loss_function(preds, target)


@register_metric("softmax_cross_entropy", [CATEGORY])
class SoftmaxCrossEntropyMetric(LossMetric):
    def __init__(self, **kwargs):
        super().__init__()
        self.softmax_cross_entropy_function = SoftmaxCrossEntropyLoss(**kwargs)

    def get_current_value(self, preds: Tensor, target: Tensor):
        return self.softmax_cross_entropy_function(preds, target)


@register_metric("sequence_softmax_cross_entropy", [SEQUENCE, TEXT])
class SequenceSoftmaxCrossEntropyMetric(LossMetric):
    def __init__(self, **kwargs):
        super().__init__()
        self.sequence_softmax_cross_entropy_function = SequenceSoftmaxCrossEntropyLoss(**kwargs)

    def get_current_value(self, preds: Tensor, target: Tensor):
        return self.sequence_softmax_cross_entropy_function(preds, target)


@register_metric("sigmoid_cross_entropy", [SET])
class SigmoidCrossEntropyMetric(LossMetric):
    def __init__(self, **kwargs):
        super().__init__()
        self.sigmoid_cross_entropy_function = SigmoidCrossEntropyLoss(**kwargs)

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        return self.sigmoid_cross_entropy_function(preds, target)


@register_metric(TOKEN_ACCURACY, [SEQUENCE, TEXT])
class TokenAccuracyMetric(MeanMetric):
    def __init__(self, **kwargs):
        super().__init__(dist_sync_fn=gather_all_tensors)

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        target = target.type(preds.dtype)
        target_sequence_length = sequence_length_2D(target)
        masked_correct_preds = masked_correct_predictions(target, preds, target_sequence_length)
        return torch.mean(masked_correct_preds)

    @classmethod
    def get_objective(cls):
        return MAXIMIZE

    @classmethod
    def get_inputs(cls):
        return PREDICTIONS


@register_metric(ACCURACY, [BINARY])
class Accuracy(_Accuracy, LudwigMetric):
    """R-squared metric."""

    def __init__(self, **kwargs):
        super().__init__(dist_sync_fn=gather_all_tensors)

    @classmethod
    def get_objective(cls):
        return MAXIMIZE

    @classmethod
    def get_inputs(cls):
        return PREDICTIONS


@register_metric(ACCURACY, [CATEGORY])
class CategoryAccuracy(_Accuracy, LudwigMetric):
    def __init__(self, **kwargs):
        super().__init__(dist_sync_fn=gather_all_tensors)

    def update(self, preds: Tensor, target: Tensor) -> None:
        # make sure y_true is tf.int64
        super().update(preds, target.type(torch.long))

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
        super().__init__(top_k=top_k, dist_sync_fn=gather_all_tensors)

    def update(self, preds: Tensor, target: Tensor) -> None:
        super().update(preds, target)

    @classmethod
    def get_objective(cls):
        return MAXIMIZE

    @classmethod
    def get_inputs(cls):
        return LOGITS

    @classmethod
    def can_report(cls, feature: "OutputFeature") -> bool:  # noqa: F821
        return feature.decoder_obj.num_classes > feature.top_k


@register_metric(MEAN_ABSOLUTE_ERROR, [NUMBER, VECTOR])
class MAEMetric(MeanAbsoluteError, LudwigMetric):
    def __init__(self, **kwargs):
        super().__init__(dist_sync_fn=gather_all_tensors)

    def update(self, preds: Tensor, target: Tensor) -> None:
        super().update(preds.detach(), target)

    @classmethod
    def get_objective(cls):
        return MINIMIZE

    @classmethod
    def get_inputs(cls):
        return PREDICTIONS


@register_metric(MEAN_SQUARED_ERROR, [NUMBER, VECTOR])
class MSEMetric(MeanSquaredError, LudwigMetric):
    def __init__(self, **kwargs):
        super().__init__(dist_sync_fn=gather_all_tensors)

    def update(self, preds: Tensor, target: Tensor) -> None:
        super().update(preds, target)

    @classmethod
    def get_objective(cls):
        return MINIMIZE

    @classmethod
    def get_inputs(cls):
        return PREDICTIONS


@register_metric(JACCARD, [SET])
class JaccardMetric(MeanMetric):
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(dist_sync_fn=gather_all_tensors)
        self.threshold = threshold
        self.add_state(name="loss", default=[], dist_reduce_fx="mean")

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        # notation: b is batch size and nc is number of unique elements in the set
        # preds: shape [b, nc] probabilities for each class
        # target: shape [b, nc] bit-mapped set representation
        preds = torch.greater_equal(preds, self.threshold)  # now bit-mapped set
        target = target.type(torch.bool)

        intersection = torch.sum(torch.logical_and(target, preds).type(torch.float32), dim=-1)
        union = torch.sum(torch.logical_or(target, preds).type(torch.float32), dim=-1)

        return intersection / union  # shape [b]

    @classmethod
    def get_objective(cls):
        return MAXIMIZE

    @classmethod
    def get_inputs(cls):
        return PROBABILITIES


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
