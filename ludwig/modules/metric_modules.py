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
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import torch
import torchmetrics.functional as metrics_F
from torch import Tensor, tensor
from torchmetrics.classification import BinaryAccuracy
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanMetric as _MeanMetric
from torchmetrics import MeanSquaredError, Metric
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
from ludwig.utils.loss_utils import rmspe_loss
from ludwig.utils.metric_utils import masked_correct_predictions
from ludwig.utils.torch_utils import sequence_length_2D

logger = logging.getLogger(__name__)


_METRIC_INIT_KWARGS = {
    "compute_on_step", "dist_sync_on_step", "process_group", "dist_sync_fn",
    "distributed_available_fn", "sync_on_compute", "compute_with_cache",
}


class LudwigMetric(Metric, ABC):
    def __init__(self, **kwargs):
        # Filter out kwargs not recognized by torchmetrics.Metric to avoid ValueError
        # on unexpected keyword arguments (e.g., loss kwargs passed from OutputFeature).
        filtered = {k: v for k, v in kwargs.items() if k in _METRIC_INIT_KWARGS}
        super().__init__(**filtered)

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


@register_metric(ROOT_MEAN_SQUARED_ERROR, [NUMBER])
class RMSEMetric(MeanSquaredError, LudwigMetric):
    """Root mean squared error metric."""

    def __init__(self, **kwargs):
        super().__init__(squared=False, **kwargs)

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
        super().__init__()
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
        super().__init__()
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
        super().__init__(**kwargs)

    """ Root mean squared percentage error metric. """

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        return rmspe_loss(target, preds)

    @classmethod
    def get_objective(cls):
        return MINIMIZE

    @classmethod
    def get_inputs(cls):
        return PREDICTIONS


@register_metric(R2, [NUMBER, VECTOR])
class R2Score(LudwigMetric):
    """Custom R-squared metric implementation that modifies torchmetrics R-squared implementation to return Nan
    when there is only sample. This is because R-squared is only defined for two or more samples.

    Custom implementation uses code from torchmetrics v0.9.2's implementation of R2:
    https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/regression/r2.py
    """

    def __init__(
        self, num_outputs: int = 1, adjusted: int = 0, multioutput: str = "uniform_average", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.num_outputs = num_outputs

        if adjusted < 0 or not isinstance(adjusted, int):
            raise ValueError("`adjusted` parameter should be an integer larger or equal to 0.")
        self.adjusted = adjusted

        allowed_multioutput = ("raw_values", "uniform_average", "variance_weighted")
        if multioutput not in allowed_multioutput:
            raise ValueError(
                f"Invalid input to argument `multioutput`. Choose one of the following: {allowed_multioutput}"
            )
        self.multioutput = multioutput

        self.add_state("sum_squared_error", default=torch.zeros(self.num_outputs), dist_reduce_fx="sum")
        self.add_state("sum_error", default=torch.zeros(self.num_outputs), dist_reduce_fx="sum")
        self.add_state("residual", default=torch.zeros(self.num_outputs), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        n_obs = target.shape[0]
        sum_error = target.sum(dim=0)
        sum_squared_error = (target**2).sum(dim=0)
        residual = ((target - preds) ** 2).sum(dim=0)

        self.sum_squared_error += sum_squared_error
        self.sum_error += sum_error
        self.residual += residual
        self.total += n_obs

    def compute(self) -> Tensor:
        """Computes r2 score over the metric states."""

        # self.total maps to the number of observations in preds/target computed during update()
        if self.total <= 1:
            logger.warning(
                """R-squared (r2) is not defined for one sample. It needs at least two samples. Returning NaN."""
            )
            return torch.tensor(float("nan"))

        mean = self.sum_error / self.total
        ss_tot = self.sum_squared_error - self.total * mean**2
        ss_res = self.residual
        r2 = 1 - ss_res / ss_tot.clamp(min=1e-8)

        if self.adjusted:
            r2 = 1 - (1 - r2) * (self.total - 1) / (self.total - self.adjusted - 1)

        if self.multioutput == "raw_values":
            return r2
        if self.multioutput == "uniform_average":
            return r2.mean()
        # variance_weighted
        ss_tot_sum = ss_tot.sum()
        if ss_tot_sum == 0:
            return r2.mean()
        return (r2 * ss_tot / ss_tot_sum).sum()

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
        super().__init__(**kwargs)

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
class Accuracy(BinaryAccuracy, LudwigMetric):
    """Binary accuracy metric."""

    def __init__(self, **kwargs):
        # Filter out kwargs not recognized by BinaryAccuracy
        valid_keys = {"threshold", "multidim_average", "ignore_index", "validate_args"}
        valid_keys |= _METRIC_INIT_KWARGS
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
        super().__init__(**filtered)

    @classmethod
    def get_objective(cls):
        return MAXIMIZE

    @classmethod
    def get_inputs(cls):
        return PREDICTIONS


@register_metric(ACCURACY, [CATEGORY])
class CategoryAccuracy(LudwigMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        target = target.type(torch.long)
        preds = preds.type(torch.long)
        self.correct += (preds == target).sum()
        self.total += target.numel()

    def compute(self) -> Tensor:
        return self.correct.float() / self.total

    @classmethod
    def get_objective(cls):
        return MAXIMIZE

    @classmethod
    def get_inputs(cls):
        return PREDICTIONS


@register_metric(HITS_AT_K, [CATEGORY])
class HitsAtKMetric(LudwigMetric):
    def __init__(self, top_k: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        _, top_k_indices = preds.topk(self.top_k, dim=-1)
        target = target.long().unsqueeze(-1)
        self.correct += (top_k_indices == target).any(dim=-1).sum()
        self.total += target.size(0)

    def compute(self) -> Tensor:
        return self.correct.float() / self.total

    @classmethod
    def get_objective(cls):
        return MAXIMIZE

    @classmethod
    def get_inputs(cls):
        return LOGITS

    @classmethod
    def can_report(cls, feature: "OutputFeature") -> bool:  # noqa: F821
        return feature.num_classes > feature.top_k


@register_metric(MEAN_ABSOLUTE_ERROR, [NUMBER, VECTOR])
class MAEMetric(MeanAbsoluteError, LudwigMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        super().__init__(**kwargs)

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
        super().__init__(**kwargs)
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
