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


from typing import Optional, Union, List
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import (MSELoss as _MSELoss, L1Loss)

from ludwig.constants import LOGITS, PREDICTIONS
import ludwig.utils.loss_utils as utils
from ludwig.utils.torch_utils import sequence_length_2D

# used for Laplace smoothing for candidate samplers
EPSILON = 1.0e-10


class MSELoss(_MSELoss):
    """ Mean squared error. """


class MAELoss(L1Loss):
    """ Mean absolute error. """
    pass


class RMSELoss(nn.Module):
    """ Root mean square error. """

    def __init__(self, **kwargs):
        super().__init__()
        self.mse = nn.MSELoss(**kwargs)

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        return torch.sqrt(self.mse(preds, target))


class RMSPELoss(nn.Module):
    """ Root mean square percentage error. """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        loss = utils.rmspe_loss(target, preds)
        return loss


class BWCEWLoss(nn.Module):
    """ Binary weighted cross entropy loss. """

    def __init__(
            self,
            positive_class_weight: Optional[Tensor] = None,
            robust_lambda: int = 0,
            confidence_penalty: int = 0,
            **kwargs):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=positive_class_weight, **kwargs)
        self.robust_lambda = robust_lambda
        self.confidence_penalty = confidence_penalty

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        train_loss = self.loss_fn(preds, target.float())
        # robust lambda
        if self.robust_lambda > 0:
            train_loss = (1 - self.robust_lambda) * train_loss + \
                self.robust_lambda / 2

        train_mean_loss = torch.mean(train_loss)

        # confidence penalty
        if self.confidence_penalty > 0:
            probabilities = torch.sigmoid(preds)
            mean_penalty = utils.mean_confidence_penalty(probabilities, 2)
            train_mean_loss += self.confidence_penalty * mean_penalty

        return train_mean_loss


class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights: Optional[Union[Tensor, List]] = None, **kwargs):
        """
        Params:
            class_weights: List or 1D tensor of length equal to number of classes.
        """
        super().__init__()
        if class_weights:
            self.loss_fn = nn.CrossEntropyLoss(
                weight=torch.Tensor(class_weights))
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


# # For Categorical Output Features
# class SampledSoftmaxCrossEntropyLoss(tf.keras.losses.Loss):
#     def __init__(
#         self, decoder_obj=None, num_classes=0, feature_loss=None, name=None
#     ):
#         super().__init__(name=name)
#
#         self.decoder_obj = decoder_obj
#         self.num_classes = num_classes
#         self.feature_loss = feature_loss
#
#     def call(self, y, y_pred):
#         decoder_weights = self.decoder_obj.weights[0]
#         decoder_biases = self.decoder_obj.weights[1]
#
#         loss = sampled_softmax_cross_entropy(
#             y,
#             y_pred[PROJECTION_INPUT],
#             num_classes=self.num_classes,
#             decoder_weights=decoder_weights,
#             decoder_biases=decoder_biases,
#             **self.feature_loss
#         )
#
#         return loss
#
#
# # For Sequence Output Feature
# class SequenceSampledSoftmaxCrossEntropyLoss(tf.keras.losses.Loss):
#     def __init__(
#         self,
#         dec_dense_layer=None,
#         dec_num_layers=None,
#         num_classes=0,
#         feature_loss=None,
#         name=None,
#     ):
#         super(SequenceSampledSoftmaxCrossEntropyLoss, self).__init__(name=name)
#
#         self.num_classes = num_classes
#         self.feature_loss = feature_loss
#         self.dec_dense_layer = dec_dense_layer
#
#     def call(self, y, y_pred):
#
#         loss = sequence_sampled_softmax_cross_entropy(
#             y,  # targets
#             y_pred[PROJECTION_INPUT],
#             decoder_weights=self.dec_dense_layer.weights[0],
#             decoder_biases=self.dec_dense_layer.weights[1],
#             num_classes=self.num_classes,
#             **self.feature_loss
#         )
#
#         return loss


class SigmoidCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights: Optional[Union[Tensor, List]] = None, **kwargs):
        """
        Params:
            class_weights: List or 1D tensor of length equal to number of classes.
        """
        super().__init__()
        if class_weights:
            self.loss_fn = nn.BCEWithLogitsLoss(
                reduction='none',
                pos_weight=torch.Tensor(class_weights))
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        if preds.ndim != 2:
            raise RuntimeError(
                'SigmoidCrossEntropyLoss currently supported for 2D tensors.')

        element_loss = self.loss_fn(
            preds.type(torch.float32),
            target.type(torch.float32)
        )

        # Reduce by sum along column dimension, mean along batch dimension.
        loss = torch.sum(element_loss, dim=1)
        loss = torch.mean(loss)
        return loss


# TODO(shreya): To migrate from below here
################################################################################

# class SequenceSoftmaxCrossEntropyLoss(nn.Module):
#     def __init__(self, name=None, from_logits=True, **kwargs):
#         super().__init__(name=name)
#         self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
#             from_logits=from_logits, reduction="none"
#         )
#         self.from_logits = from_logits

#     def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
#         """
#         Params:
#             y_true: Labels of shape [batch x seq_size]
#             y_pred: Predictions of shape [batch x seq_size x num_classes]
#         """
#         # TODO(shreya): Make sure that the features using this are sending the correct tensor.
#         # if self.from_logits:
#         #     y_pred_tensor = y_pred[LOGITS]
#         # else:
#         #     y_pred_tensor = y_pred[PROBABILITIES]
#         y_pred_tensor = y_pred.type(torch.int64)
#         y_true_tensor = y_true.type(torch.int64)

#         # Pad both tensors so that they have the same sequence length.
#         if y_true_tensor.shape[1] > y_pred_tensor.shape[1]:
#             y_pred_tensor = F.pad(
#                 y_true_tensor,
#                 pad=(0, y_true_tensor.shape[1] - y_pred_tensor.shape[1], 0, 0))
#         elif y_pred_tensor.shape[1] > y_true_tensor.shape[1]:
#             y_true_tensor = F.pad(
#                 y_true_tensor,
#                 pad=(0, y_pred_tensor.shape[1] - y_true_tensor.shape[1]))

#         y_true_seq_len = sequence_length_2D(y_true_tensor)

#         mask = tf.sequence_mask(
#             y_true_seq_len + 1,  # this is for including the eos
#             # in case of generator and shouldn't impact
#             # negatively in case of tagger
#             maxlen=tf.shape(y_true_tensor)[1],
#             dtype=tf.float32,
#         )
#         # compute loss based on valid time steps
#         loss = self.loss_function(y_true_tensor, y_pred_tensor)
#         loss = loss * mask
#         loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
#         return loss
#


# Registry to map loss name to the desired predicted input type.
LOSS_INPUTS_REGISTRY = {
    MSELoss: LOGITS,
    MAELoss: LOGITS,
    RMSELoss: LOGITS,
    RMSPELoss: LOGITS,
    BWCEWLoss: LOGITS,
    SoftmaxCrossEntropyLoss: LOGITS,
    SigmoidCrossEntropyLoss: LOGITS,
}
