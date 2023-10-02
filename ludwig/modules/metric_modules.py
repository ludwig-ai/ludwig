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
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional, Type

import torch
from torch import Tensor, tensor
from torchmetrics import CharErrorRate, MeanAbsoluteError, MeanAbsolutePercentageError
from torchmetrics import MeanMetric as _MeanMetric
from torchmetrics import MeanSquaredError, Metric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    MulticlassAccuracy,
    MulticlassAUROC,
)
from torchmetrics.functional.regression.r2 import _r2_score_compute, _r2_score_update
from torchmetrics.metric import jit_distributed_available
from torchmetrics.text import BLEUScore, WordErrorRate
from torchmetrics.text.perplexity import Perplexity
from torchmetrics.text.rouge import ROUGEScore

from ludwig.constants import (
    ACCURACY,
    ACCURACY_MICRO,
    BINARY,
    BINARY_WEIGHTED_CROSS_ENTROPY,
    CATEGORY,
    CATEGORY_DISTRIBUTION,
    CORN,
    HITS_AT_K,
    HUBER,
    IGNORE_INDEX_TOKEN_ID,
    JACCARD,
    LOGITS,
    LOSS,
    MAXIMIZE,
    MEAN_ABSOLUTE_ERROR,
    MEAN_ABSOLUTE_PERCENTAGE_ERROR,
    MEAN_SQUARED_ERROR,
    MINIMIZE,
    NEXT_TOKEN_PERPLEXITY,
    NUMBER,
    PERPLEXITY,
    PRECISION,
    PREDICTIONS,
    PROBABILITIES,
    R2,
    RECALL,
    RESPONSE,
    ROC_AUC,
    ROOT_MEAN_SQUARED_ERROR,
    ROOT_MEAN_SQUARED_PERCENTAGE_ERROR,
    SEQUENCE,
    SEQUENCE_ACCURACY,
    SET,
    SPECIFICITY,
    TEXT,
    TIMESERIES,
    TOKEN_ACCURACY,
    VECTOR,
)
from ludwig.distributed import get_current_dist_strategy
from ludwig.modules.loss_modules import (
    BWCEWLoss,
    CORNLoss,
    HuberLoss,
    NextTokenSoftmaxCrossEntropyLoss,
    SequenceSoftmaxCrossEntropyLoss,
    SigmoidCrossEntropyLoss,
    SoftmaxCrossEntropyLoss,
)
from ludwig.modules.metric_registry import get_metric_objective, get_metric_registry, register_metric
from ludwig.schema.features.loss.loss import (
    BWCEWLossConfig,
    CORNLossConfig,
    HuberLossConfig,
    SequenceSoftmaxCrossEntropyLossConfig,
    SigmoidCrossEntropyLossConfig,
    SoftmaxCrossEntropyLossConfig,
)
from ludwig.utils.loss_utils import rmspe_loss
from ludwig.utils.metric_utils import masked_correct_predictions
from ludwig.utils.torch_utils import sequence_length_2D

logger = logging.getLogger(__name__)


class LudwigMetric(Metric, ABC):
    @classmethod
    def can_report(cls, feature: "OutputFeature") -> bool:  # noqa: F821
        return True

    @contextmanager
    def sync_context(
        self,
        dist_sync_fn: Optional[Callable] = None,
        process_group: Optional[Any] = None,
        should_sync: bool = True,
        should_unsync: bool = True,
        distributed_available: Optional[Callable] = jit_distributed_available,
    ) -> Generator:
        """Override the behavior of this in the base class to support custom distributed strategies."""
        dist_strategy = get_current_dist_strategy()
        self.sync(
            dist_sync_fn=dist_strategy.gather_all_tensors_fn(),
            process_group=process_group,
            should_sync=should_sync,
            distributed_available=dist_strategy.is_available,
        )

        yield

        self.unsync(should_unsync=self._is_synced and should_unsync)


@register_metric(ROOT_MEAN_SQUARED_ERROR, [NUMBER], MINIMIZE, PREDICTIONS)
class RMSEMetric(MeanSquaredError, LudwigMetric):
    """Root mean squared error metric."""

    def __init__(self, **kwargs):
        super().__init__(squared=False, **kwargs)


@register_metric(PRECISION, [BINARY], MAXIMIZE, PROBABILITIES)
class PrecisionMetric(BinaryPrecision, LudwigMetric):
    """Precision metric."""

    def __init__(self, **kwargs):
        super().__init__()


@register_metric(RECALL, [BINARY], MAXIMIZE, PROBABILITIES)
class RecallMetric(BinaryRecall, LudwigMetric):
    """Recall metric."""

    def __init__(self, **kwargs):
        super().__init__()


@register_metric(ROC_AUC, [BINARY], MAXIMIZE, PROBABILITIES)
class BinaryAUROCMetric(BinaryAUROC, LudwigMetric):
    """Area under the receiver operating curve."""

    def __init__(self, **kwargs):
        super().__init__()

    def update(self, preds: Tensor, target: Tensor) -> None:
        super().update(preds, target.type(torch.int8))


@register_metric(ROC_AUC, [CATEGORY, CATEGORY_DISTRIBUTION], MAXIMIZE, PROBABILITIES)
class CategoryAUROCMetric(MulticlassAUROC, LudwigMetric):
    """Area under the receiver operating curve."""

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes=num_classes)

    def update(self, preds: Tensor, target: Tensor) -> None:
        if len(target.shape) > 1:
            target = torch.argmax(target, dim=1)
        super().update(preds, target)


@register_metric(SPECIFICITY, [BINARY], MAXIMIZE, PROBABILITIES)
class SpecificityMetric(BinarySpecificity, LudwigMetric):
    """Specificity metric."""

    def __init__(self, **kwargs):
        super().__init__()


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


@register_metric(ROOT_MEAN_SQUARED_PERCENTAGE_ERROR, [NUMBER], MINIMIZE, PREDICTIONS)
class RMSPEMetric(MeanMetric):
    def __init__(self, **kwargs):
        super().__init__()

    """ Root mean squared percentage error metric. """

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        return rmspe_loss(target, preds)


@register_metric(R2, [NUMBER, VECTOR, TIMESERIES], MAXIMIZE, PREDICTIONS)
class R2Score(LudwigMetric):
    """Custom R-squared metric implementation that modifies torchmetrics R-squared implementation to return Nan
    when there is only sample. This is because R-squared is only defined for two or more samples.

    Custom implementation uses code from torchmetrics v0.9.2's implementation of R2: https://github.com/Lightning-
    AI/metrics/blob/master/src/torchmetrics/regression/r2.py
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
        sum_squared_error, sum_error, residual, n_obs = _r2_score_update(preds, target)

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

        return _r2_score_compute(
            self.sum_squared_error, self.sum_error, self.residual, self.total, self.adjusted, self.multioutput
        )


@register_metric(LOSS, [], MINIMIZE, LOGITS)
class LossMetric(MeanMetric, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError()

    @classmethod
    def can_report(cls, feature: "OutputFeature") -> bool:  # noqa: F821
        return False


@register_metric(BINARY_WEIGHTED_CROSS_ENTROPY, [BINARY], MINIMIZE, LOGITS)
class BWCEWLMetric(LossMetric):
    """Binary Weighted Cross Entropy Weighted Logits Score Metric."""

    def __init__(self, config: BWCEWLossConfig, **kwargs):
        super().__init__()
        self.loss_function = BWCEWLoss(config)

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        return self.loss_function(preds, target)


@register_metric("softmax_cross_entropy", [CATEGORY, CATEGORY_DISTRIBUTION], MINIMIZE, LOGITS)
class SoftmaxCrossEntropyMetric(LossMetric):
    def __init__(self, config: SoftmaxCrossEntropyLossConfig, **kwargs):
        super().__init__()
        self.softmax_cross_entropy_function = SoftmaxCrossEntropyLoss(config)

    def get_current_value(self, preds: Tensor, target: Tensor):
        return self.softmax_cross_entropy_function(preds, target)


@register_metric("sequence_softmax_cross_entropy", [SEQUENCE, TEXT], MINIMIZE, LOGITS)
class SequenceSoftmaxCrossEntropyMetric(LossMetric):
    def __init__(self, config: SequenceSoftmaxCrossEntropyLossConfig, **kwargs):
        super().__init__()
        self.sequence_softmax_cross_entropy_function = SequenceSoftmaxCrossEntropyLoss(config)

    def get_current_value(self, preds: Tensor, target: Tensor):
        return self.sequence_softmax_cross_entropy_function(preds, target)


@register_metric("next_token_softmax_cross_entropy", [SEQUENCE, TEXT], MINIMIZE, LOGITS)
class NextTokenSoftmaxCrossEntropyMetric(LossMetric):
    def __init__(self, config: SequenceSoftmaxCrossEntropyLossConfig, **kwargs):
        super().__init__()
        self.next_token_softmax_cross_entropy_function = NextTokenSoftmaxCrossEntropyLoss(config)

    def get_current_value(self, preds: Tensor, target: Tensor):
        return self.next_token_softmax_cross_entropy_function(preds, target)


@register_metric("sigmoid_cross_entropy", [SET], MINIMIZE, LOGITS)
class SigmoidCrossEntropyMetric(LossMetric):
    def __init__(self, config: SigmoidCrossEntropyLossConfig, **kwargs):
        super().__init__()
        self.sigmoid_cross_entropy_function = SigmoidCrossEntropyLoss(config)

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        return self.sigmoid_cross_entropy_function(preds, target)


@register_metric(TOKEN_ACCURACY, [SEQUENCE, TEXT], MAXIMIZE, PREDICTIONS)
class TokenAccuracyMetric(MeanMetric):
    def __init__(self, **kwargs):
        super().__init__()

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        target = target.type(preds.dtype)
        target_sequence_length = sequence_length_2D(target)
        masked_correct_preds = masked_correct_predictions(target, preds, target_sequence_length)
        return torch.mean(masked_correct_preds)


@register_metric(SEQUENCE_ACCURACY, [SEQUENCE, TEXT], MAXIMIZE, PREDICTIONS)
class SequenceAccuracyMetric(MeanMetric):
    def __init__(self, **kwargs):
        super().__init__()

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        return torch.sum(torch.all(preds == target, dim=1)) / target.size()[0]


@register_metric(PERPLEXITY, [SEQUENCE, TEXT], MINIMIZE, PROBABILITIES)
class PerplexityMetric(Perplexity, LudwigMetric):
    def __init__(self, **kwargs):
        super().__init__(ignore_index=IGNORE_INDEX_TOKEN_ID)

    def update(self, preds: Tensor, target: Tensor) -> None:
        super().update(preds, target.type(torch.int64))


@register_metric(NEXT_TOKEN_PERPLEXITY, [SEQUENCE, TEXT], MINIMIZE, PROBABILITIES)
class NextTokenPerplexityMetric(MeanMetric):
    def __init__(self, **kwargs):
        super().__init__()
        self.next_token_softmax_cross_entropy_function = NextTokenSoftmaxCrossEntropyLoss({})

    def get_current_value(self, preds: Tensor, target: Tensor):
        # Perplexity can be represented as the exponential of the cross-entropy loss.
        # https://towardsdatascience.com/perplexity-in-language-models-87a196019a94
        # We can't use torchmetrics perplexity because it calculates normal cross-entropy
        # loss as opposed to shifted cross entropy loss.
        shifted_loss = self.next_token_softmax_cross_entropy_function(preds, target)
        return torch.exp(shifted_loss)


@register_metric("bleu", [TEXT], MAXIMIZE, RESPONSE)
class BLEUScoreMetric(BLEUScore, LudwigMetric):
    def __init__(self, **kwargs):
        super().__init__()


@register_metric("rouge", [TEXT], MAXIMIZE, RESPONSE)
class ROUGEScoreMetric(ROUGEScore, LudwigMetric):
    def __init__(self, **kwargs):
        super().__init__()


@register_metric("word_error_rate", [TEXT], MINIMIZE, RESPONSE)
class WordErrorRateMetric(WordErrorRate, LudwigMetric):
    def __init__(self, **kwargs):
        super().__init__()


@register_metric("char_error_rate", [TEXT], MINIMIZE, RESPONSE)
class CharErrorRateMetric(CharErrorRate, LudwigMetric):
    def __init__(self, **kwargs):
        super().__init__()


@register_metric(ACCURACY, [BINARY], MAXIMIZE, PREDICTIONS)
class Accuracy(BinaryAccuracy, LudwigMetric):
    """R-squared metric."""

    def __init__(self, **kwargs):
        super().__init__()


@register_metric(ACCURACY, [CATEGORY, CATEGORY_DISTRIBUTION], MAXIMIZE, PREDICTIONS)
class CategoryAccuracy(MulticlassAccuracy, LudwigMetric):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes=num_classes)

    def update(self, preds: Tensor, target: Tensor) -> None:
        if len(target.shape) > 1:
            target = torch.argmax(target, dim=1)
        super().update(preds, target.type(torch.long))


@register_metric(ACCURACY_MICRO, [CATEGORY, CATEGORY_DISTRIBUTION], MAXIMIZE, PREDICTIONS)
class CategoryAccuracyMicro(MulticlassAccuracy, LudwigMetric):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes=num_classes, average="micro")

    def update(self, preds: Tensor, target: Tensor) -> None:
        if len(target.shape) > 1:
            target = torch.argmax(target, dim=1)
        super().update(preds, target.type(torch.long))


@register_metric(HITS_AT_K, [CATEGORY, CATEGORY_DISTRIBUTION], MAXIMIZE, LOGITS)
class HitsAtKMetric(MulticlassAccuracy, LudwigMetric):
    def __init__(self, num_classes: int, top_k: int, **kwargs):
        super().__init__(num_classes=num_classes, top_k=top_k, **kwargs)

    def update(self, preds: Tensor, target: Tensor) -> None:
        if len(target.shape) > 1:
            target = torch.argmax(target, dim=1)
        super().update(preds, target.type(torch.long))

    @classmethod
    def can_report(cls, feature: "OutputFeature") -> bool:  # noqa: F821
        return feature.num_classes > feature.top_k


@register_metric(MEAN_ABSOLUTE_ERROR, [NUMBER, VECTOR, TIMESERIES], MINIMIZE, PREDICTIONS)
class MAEMetric(MeanAbsoluteError, LudwigMetric):
    def __init__(self, **kwargs):
        super().__init__()

    def update(self, preds: Tensor, target: Tensor) -> None:
        super().update(preds.detach(), target)


@register_metric(MEAN_SQUARED_ERROR, [NUMBER, VECTOR, TIMESERIES], MINIMIZE, PREDICTIONS)
class MSEMetric(MeanSquaredError, LudwigMetric):
    def __init__(self, **kwargs):
        super().__init__()

    def update(self, preds: Tensor, target: Tensor) -> None:
        super().update(preds, target)


@register_metric(MEAN_ABSOLUTE_PERCENTAGE_ERROR, [NUMBER, VECTOR, TIMESERIES], MINIMIZE, PREDICTIONS)
class MAPEMetric(MeanAbsolutePercentageError, LudwigMetric):
    def __init__(self, **kwargs):
        super().__init__()

    def update(self, preds: Tensor, target: Tensor) -> None:
        super().update(preds, target)


@register_metric(JACCARD, [SET], MAXIMIZE, PROBABILITIES)
class JaccardMetric(MeanMetric):
    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__()
        self.threshold = threshold

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        # notation: b is batch size and nc is number of unique elements in the set
        # preds: shape [b, nc] probabilities for each class
        # target: shape [b, nc] bit-mapped set representation
        preds = torch.greater_equal(preds, self.threshold)  # now bit-mapped set
        target = target.type(torch.bool)

        intersection = torch.sum(torch.logical_and(target, preds).type(torch.float32), dim=-1)
        union = torch.sum(torch.logical_or(target, preds).type(torch.float32), dim=-1)

        return intersection / union  # shape [b]


@register_metric(HUBER, [NUMBER, VECTOR, TIMESERIES], MINIMIZE, PREDICTIONS)
class HuberMetric(LossMetric):
    def __init__(
        self,
        config: HuberLossConfig,
        **kwargs,
    ):
        super().__init__()
        self.loss_function = HuberLoss(config=config)

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        return self.loss_function(preds, target)


@register_metric(CORN, [CATEGORY], MINIMIZE, PREDICTIONS)
class CORNMetric(LossMetric):
    def __init__(
        self,
        config: CORNLossConfig,
        **kwargs,
    ):
        super().__init__()
        self.loss_function = CORNLoss(config=config)

    def get_current_value(self, preds: Tensor, target: Tensor) -> Tensor:
        return self.loss_function(preds, target)


def get_metric_cls(metric_name: str) -> Type[LudwigMetric]:
    return get_metric_registry()[metric_name]


def get_improved_fn(metric: str) -> Callable:
    if get_metric_objective(metric) == MINIMIZE:
        return lambda x, y: x < y
    else:
        return lambda x, y: x > y


def get_initial_validation_value(metric: str) -> float:
    if get_metric_objective(metric) == MINIMIZE:
        return float("inf")
    else:
        return float("-inf")


def get_best_function(metric: str) -> Callable:
    if get_metric_objective(metric) == MINIMIZE:
        return min
    else:
        return max
