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


from typing import Type

import torch
from torch import nn, Tensor
from torch.nn import HuberLoss as _HuberLoss
from torch.nn import L1Loss
from torch.nn import MSELoss as _MSELoss
from torchmetrics.functional import mean_absolute_percentage_error

import ludwig.utils.loss_utils as utils
from ludwig.constants import LOGITS
from ludwig.modules.loss_implementations.corn import corn_loss
from ludwig.schema.features.loss.loss import (
    BaseLossConfig,
    BWCEWLossConfig,
    CORNLossConfig,
    HuberLossConfig,
    MAELossConfig,
    MAPELossConfig,
    MSELossConfig,
    NextTokenSoftmaxCrossEntropyLossConfig,
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

loss_impl_registry = Registry[Type[nn.Module]]()


def register_loss(config_cls: Type[BaseLossConfig]):
    def wrap(cls: Type[nn.Module]):
        loss_impl_registry[config_cls] = cls
        return cls

    return wrap


def create_loss(config: BaseLossConfig) -> nn.Module:
    return loss_impl_registry[type(config)](config)


class LogitsInputsMixin:
    @classmethod
    def get_loss_inputs(cls):
        """Maps loss to the desired predicted input type."""
        return LOGITS


@register_loss(MSELossConfig)
class MSELoss(_MSELoss, LogitsInputsMixin):
    """Mean squared error."""

    def __init__(self, config: MSELossConfig):
        super().__init__()


@register_loss(MAELossConfig)
class MAELoss(L1Loss, LogitsInputsMixin):
    """Mean absolute error."""

    def __init__(self, config: MAELossConfig):
        super().__init__()


@register_loss(MAPELossConfig)
class MAPELoss(nn.Module, LogitsInputsMixin):
    """Mean absolute error."""

    def __init__(self, config: MAPELossConfig):
        super().__init__()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        return mean_absolute_percentage_error(preds, target)


@register_loss(RMSELossConfig)
class RMSELoss(nn.Module, LogitsInputsMixin):
    """Root mean square error."""

    def __init__(self, config: RMSELossConfig):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        return torch.sqrt(self.mse(preds, target))


@register_loss(RMSPELossConfig)
class RMSPELoss(nn.Module, LogitsInputsMixin):
    """Root mean square percentage error."""

    def __init__(self, config: RMSPELossConfig):
        super().__init__()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        loss = utils.rmspe_loss(target, preds)
        return loss


@register_loss(BWCEWLossConfig)
class BWCEWLoss(nn.Module, LogitsInputsMixin):
    """Binary weighted cross entropy loss."""

    def __init__(self, config: BWCEWLossConfig):
        super().__init__()
        if config.positive_class_weight:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([config.positive_class_weight]))
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=config.positive_class_weight)
        self.robust_lambda = config.robust_lambda
        self.confidence_penalty = config.confidence_penalty

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


@register_loss(SoftmaxCrossEntropyLossConfig)
class SoftmaxCrossEntropyLoss(nn.Module, LogitsInputsMixin):
    def __init__(self, config: SoftmaxCrossEntropyLossConfig):
        """
        Params:
            class_weights: List or 1D tensor of length equal to number of classes.
        """
        super().__init__()
        if config.class_weights:
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(config.class_weights))
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Params:
            preds: Tensor of shape [batch x num_classes]
            target: Tensor of shape [batch], where each element is integral
                between 0 and num_classes.
        """
        if len(target.shape) == 1:
            # Assumes we are providing the target as a single class, rather than a distribution
            target = target.long()
        return self.loss_fn(preds, target)


@register_loss(SequenceSoftmaxCrossEntropyLossConfig)
class SequenceSoftmaxCrossEntropyLoss(nn.Module, LogitsInputsMixin):
    def __init__(self, config: SequenceSoftmaxCrossEntropyLossConfig):
        """
        Params:
            class_weights: List or 1D tensor of length equal to number of classes.
        """
        super().__init__()
        if config.class_weights:
            self.loss_fn = nn.CrossEntropyLoss(
                weight=torch.Tensor(config.class_weights), ignore_index=strings_utils.SpecialSymbol.PADDING.value
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=strings_utils.SpecialSymbol.PADDING.value)

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Params:
            preds: Tensor of shape [batch x sequence_length x vocab_size]
            target: Tensor of shape [batch x sequence_length], where each element is integral between 0 and vocab_size.
        """
        target = target.long()
        return self.loss_fn(preds[1:].view(-1, preds.size(-1)), target[1:].view(-1))


@register_loss(NextTokenSoftmaxCrossEntropyLossConfig)
class NextTokenSoftmaxCrossEntropyLoss(nn.Module, LogitsInputsMixin):
    def __init__(self, config: NextTokenSoftmaxCrossEntropyLossConfig):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Params:
            preds: Tensor of shape [batch x sequence_length x vocab_size]
            target: Tensor of shape [batch x sequence_length], where each element is integral between 0 and vocab_size.

        Reference implementation:
        https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/bert/modeling_bert.py#LL1253C1-L1260C1 # noqa
        """
        target = target.long()
        _, _, vocab_size = preds.shape
        # logits for all tensors except n+1 since each logit tensor at position i represents the log probabilities for
        # the next token i+1 if we were to do argmax on the logits ensor at position i.
        shifted_predictions = preds[:, :-1, :]
        # Shift by 1 since the logits at position 0 in predictions represent the log likelihood of target token 1
        shifted_targets = target[:, 1:]
        return self.loss_fn(shifted_predictions.reshape(-1, vocab_size), shifted_targets.reshape(-1))


@register_loss(SigmoidCrossEntropyLossConfig)
class SigmoidCrossEntropyLoss(nn.Module, LogitsInputsMixin):
    def __init__(self, config: SigmoidCrossEntropyLossConfig):
        """
        Params:
            class_weights: List or 1D tensor of length equal to number of classes.
        """
        super().__init__()
        if config.class_weights:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(config.class_weights))
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        if preds.ndim != 2:
            raise RuntimeError("SigmoidCrossEntropyLoss currently only supported for 2D tensors.")

        return self.loss_fn(preds.type(torch.float32), target.type(torch.float32))


@register_loss(HuberLossConfig)
class HuberLoss(_HuberLoss, LogitsInputsMixin):
    """Huber loss."""

    def __init__(self, config: HuberLossConfig):
        super().__init__(delta=config.delta)


@register_loss(CORNLossConfig)
class CORNLoss(nn.Module, LogitsInputsMixin):
    """CORN loss."""

    def __init__(self, config: CORNLossConfig):
        super().__init__()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        num_classes = preds.shape[1]
        return corn_loss(preds, target, num_classes=num_classes)
