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


from typing import List, Optional, Union

import torch
from torch import nn, Tensor
from torch.nn import L1Loss
from torch.nn import MSELoss as _MSELoss

import ludwig.utils.loss_utils as utils
from ludwig.constants import (
    BINARY,
    BINARY_WEIGHTED_CROSS_ENTROPY,
    CATEGORY,
    LOGITS,
    NUMBER,
    SEQUENCE,
    SEQUENCE_SOFTMAX_CROSS_ENTROPY,
    SET,
    SIGMOID_CROSS_ENTROPY,
    SOFTMAX_CROSS_ENTROPY,
    TEXT,
    TIMESERIES,
    VECTOR,
)
from ludwig.schema.features.loss.loss import (
    BWCEWLossConfig,
    MAELossConfig,
    MSELossConfig,
    RMSELossConfig,
    RMSPELossConfig,
    SequenceSoftmaxCrossEntropyLossConfig,
    SigmoidCrossEntropyLossConfig,
    SoftmaxCrossEntropyLossConfig,
)
from ludwig.utils import strings_utils
from ludwig.utils.registry import Registry

# used for Laplace smoothing for candidate samplers
EPSILON = 1.0e-10

loss_registry = Registry()


def register_loss(name: str, features: Union[str, List[str]]):
    if isinstance(features, str):
        features = [features]

    def wrap(cls):
        for feature in features:
            feature_registry = loss_registry.get(feature, {})
            feature_registry[name] = cls
            loss_registry[feature] = feature_registry
        return cls

    return wrap


def get_loss_cls(feature: str, name: str):
    return loss_registry[feature][name]


def get_loss_classes(feature: str):
    return loss_registry[feature]


class LogitsInputsMixin:
    @classmethod
    def get_loss_inputs(cls):
        """Maps loss to the desired predicted input type."""
        return LOGITS


@register_loss("mean_squared_error", [NUMBER, TIMESERIES, VECTOR])
class MSELoss(_MSELoss, LogitsInputsMixin):
    """Mean squared error."""

    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def get_schema_cls():
        return MSELossConfig


@register_loss("mean_absolute_error", [NUMBER, TIMESERIES, VECTOR])
class MAELoss(L1Loss, LogitsInputsMixin):
    """Mean absolute error."""

    def __init__(self, **kwargs):
        super().__init__()

    @staticmethod
    def get_schema_cls():
        return MAELossConfig


@register_loss("root_mean_squared_error", [NUMBER])
class RMSELoss(nn.Module, LogitsInputsMixin):
    """Root mean square error."""

    def __init__(self, **kwargs):
        super().__init__()
        self.mse = nn.MSELoss(**kwargs)

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        return torch.sqrt(self.mse(preds, target))

    @staticmethod
    def get_schema_cls():
        return RMSELossConfig


@register_loss("root_mean_squared_percentage_error", [NUMBER])
class RMSPELoss(nn.Module, LogitsInputsMixin):
    """Root mean square percentage error."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        loss = utils.rmspe_loss(target, preds)
        return loss

    @staticmethod
    def get_schema_cls():
        return RMSPELossConfig


@register_loss(BINARY_WEIGHTED_CROSS_ENTROPY, [BINARY])
class BWCEWLoss(nn.Module, LogitsInputsMixin):
    """Binary weighted cross entropy loss."""

    def __init__(
        self,
        positive_class_weight: Optional[Union[Tensor, int]] = None,
        robust_lambda: int = 0,
        confidence_penalty: int = 0,
        **kwargs,
    ):
        super().__init__()
        if positive_class_weight:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([positive_class_weight]), **kwargs)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_class_weight, **kwargs)
        self.robust_lambda = robust_lambda
        self.confidence_penalty = confidence_penalty

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        train_loss = self.loss_fn(preds, target.float())
        # robust lambda
        if self.robust_lambda > 0:
            train_loss = (1 - self.robust_lambda) * train_loss + self.robust_lambda / 2

        train_mean_loss = torch.mean(train_loss)

        # confidence penalty
        if self.confidence_penalty > 0:
            probabilities = torch.sigmoid(preds)
            mean_penalty = utils.mean_confidence_penalty(probabilities, 2)
            train_mean_loss += self.confidence_penalty * mean_penalty

        return train_mean_loss

    @staticmethod
    def get_schema_cls():
        return BWCEWLossConfig


@register_loss(SOFTMAX_CROSS_ENTROPY, [CATEGORY, VECTOR])
class SoftmaxCrossEntropyLoss(nn.Module, LogitsInputsMixin):
    def __init__(self, class_weights: Optional[Union[Tensor, List]] = None, **kwargs):
        """
        Params:
            class_weights: List or 1D tensor of length equal to number of classes.
        """
        super().__init__()
        if class_weights:
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Params:
            preds: Tensor of shape [batch x num_classes]
            target: Tensor of shape [batch], where each element is integral
                between 0 and num_classes.
        """
        target = target.long()
        return self.loss_fn(preds, target)

    @staticmethod
    def get_schema_cls():
        return SoftmaxCrossEntropyLossConfig


@register_loss(SEQUENCE_SOFTMAX_CROSS_ENTROPY, [SEQUENCE, TEXT])
class SequenceSoftmaxCrossEntropyLoss(nn.Module, LogitsInputsMixin):
    def __init__(self, **kwargs):
        """
        Params:
            class_weights: List or 1D tensor of length equal to number of classes.
        """
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=strings_utils.SpecialSymbol.PADDING.value)

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Params:
            preds: Tensor of shape [batch x sequence_length x vocab_size]
            target: Tensor of shape [batch x sequence_length], where each element is integral between 0 and vocab_size.
        """
        target = target.long()
        return self.loss_fn(preds[1:].view(-1, preds.size(-1)), target[1:].view(-1))

    @staticmethod
    def get_schema_cls():
        return SequenceSoftmaxCrossEntropyLossConfig


@register_loss(SIGMOID_CROSS_ENTROPY, [SET])
class SigmoidCrossEntropyLoss(nn.Module, LogitsInputsMixin):
    def __init__(self, class_weights: Optional[Union[Tensor, List]] = None, **kwargs):
        """
        Params:
            class_weights: List or 1D tensor of length equal to number of classes.
        """
        super().__init__()
        if class_weights:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(class_weights))
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        if preds.ndim != 2:
            raise RuntimeError("SigmoidCrossEntropyLoss currently supported for 2D tensors.")

        return self.loss_fn(preds.type(torch.float32), target.type(torch.float32))

    @staticmethod
    def get_schema_cls():
        return SigmoidCrossEntropyLossConfig
