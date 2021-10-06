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


from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import (MSELoss as _MSELoss, L1Loss)

from ludwig.constants import LOGITS, PREDICTIONS
import ludwig.utils.loss_utils as utils
from ludwig.utils.torch_utils import sequence_length_2D

# used for Laplace smoothing for candidate samplers
EPSILON = 1.0e-10


loss_inputs_registry = {
    'MSELoss': PREDICTIONS, #double check
    'MAELoss': PREDICTIONS, #double check
    'RMSELoss': PREDICTIONS, #double check
    'RMSPELoss': PREDICTIONS, #double check
    'BWCEWLoss': LOGITS,
    'SoftmaxCrossEntropyLoss': LOGITS, #double check
    'SigmoidCrossEntropyLoss': LOGITS,
}


class MSELoss(_MSELoss):
    """ Mean squared error. """
    pass


class MAELoss(L1Loss):
    """ Mean absolute error. """
    pass


class RMSELoss(nn.Module):
    """ Root mean square error. """
    def __init__(self, **kwargs):
        super().__init__()
        self.mse = nn.MSELoss(**kwargs)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.sqrt(self.mse(input, target))


class RMSPELoss(nn.Module):
    """ Root mean square percentage error. """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = utils.rmspe_loss(target, input)
        return loss


class BWCEWLoss:
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(**kwargs)

    def mean_confidence_penalty(self, probabilities, num_classes):
        max_entropy = torch.log(torch.tensor(num_classes))
        # clipping needed for avoiding log(0) = -inf
        entropy_per_class = torch.maximum(
            -probabilities * torch.log(torch.clamp(probabilities, 1e-10, 1)),
            0
        )
        entropy = torch.sum(entropy_per_class, -1)
        penalty = (max_entropy - entropy) / max_entropy
        return torch.mean(penalty)

    def forward(self, predictions: torch.Tensor, target: torch.Tensor):
        logits = predictions[LOGITS]
        target = target.long()
        output = self.loss_fn(input, target)
        # robust lambda
        if self.robust_lambda > 0:
            train_loss = (
                                 1 - self.robust_lambda
                         ) * output + self.robust_lambda / 2

        train_mean_loss = torch.mean(train_loss)
        #
        # confidence penalty
        if self.confidence_penalty > 0:
            # need to define logits
            probabilities = torch.sigmoid(logits)
            mean_penalty = self.mean_confidence_penalty(probabilities, 2)
            train_mean_loss += self.confidence_penalty * mean_penalty

        return train_mean_loss

# TODO torch: test behavior parity with tf
class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights: Optional[Tensor]=None, **kwargs):
        """
        Params:
            class_weights: 1D tensor of length equal to number of classes.
        """
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = target.long()
        print(f'preds: {input.shape} {input.dtype} {input}')
        print(f'target: {target.shape} {target.dtype} {target}')
        return self.loss_fn(input, target)


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
#
#
class SigmoidCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights: Optional[Tensor] = None, **kwargs):
        """
        Params:
            class_weights: 1D tensor of length equal to number of classes.
        """
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(
            reduction='none',
            pos_weight=class_weights
        )
        self.class_weights = class_weights

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # TODO(shreya): Make sure y[LOGITS] is passed here
        if input.ndim != 2:
            raise RuntimeError(
                'SigmoidCrossEntropyLoss currently supported for 2D tensors.')

        element_loss = self.loss_fn(
            input.type(torch.float32),
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
#
# def mean_confidence_penalty(probabilities, num_classes):
#     max_entropy = tf.constant(np.log(num_classes), dtype=tf.float32)
#     # clipping needed for avoiding log(0) = -inf
#     entropy_per_class = tf.maximum(
#         -probabilities
#         * tf.math.log(tf.clip_by_value(probabilities, 1e-10, 1)),
#         0,
#     )
#     entropy = tf.reduce_sum(entropy_per_class, -1)
#     penalty = (max_entropy - entropy) / max_entropy
#     return tf.reduce_mean(penalty)
#
#
# # For categorical feature
# def sampled_softmax_cross_entropy(
#     labels,
#     last_hidden,
#     num_classes=1,
#     decoder_weights=None,
#     decoder_biases=None,
#     sampler=None,
#     negative_samples=0,
#     class_counts=0,
#     distortion=1,
#     unique=False,
#     **kwargs
# ):
#     labels = tf.cast(tf.expand_dims(labels, -1), tf.int64)
#
#     sampled_values = sample_values_from_classes(
#         labels,
#         sampler,
#         num_classes,
#         negative_samples,
#         unique,
#         class_counts,
#         distortion,
#     )
#     train_loss = tf.nn.sampled_softmax_loss(
#         weights=tf.transpose(decoder_weights),
#         biases=decoder_biases,
#         labels=labels,
#         inputs=last_hidden,
#         num_sampled=negative_samples,
#         num_classes=num_classes,
#         sampled_values=sampled_values,
#     )
#
#     return train_loss
#
#
# # custom class to support Laplace smoothing of Fixed Unigram candidate sampler
# # Required because of zeros returned in the true_expected_count for
# # <PAD> and <UNK> tokens in loss['class_counts'] list
# class FixedUnigramCandidateSampler(
#     collections.namedtuple(
#         "FixedUnigramCandidateSampler",
#         (
#             "sampled_candidates",
#             "true_expected_count",
#             "sampled_expected_count",
#         ),
#     )
# ):
#     pass
#
#
# # For sequence feature
# def sequence_sampled_softmax_cross_entropy(
#     targets, train_logits, decoder_weights, decoder_biases, num_classes, **loss
# ):
#     batch_max_targets_sequence_length = tf.shape(targets)[1]
#     targets_sequence_length = sequence_length_2D(tf.cast(targets, tf.int64))
#     batch_max_train_logits_sequence_length = tf.shape(train_logits)[1]
#
#     logits_pad_len = tf.maximum(
#         0,
#         batch_max_targets_sequence_length
#         - batch_max_train_logits_sequence_length,
#     )
#     targets_pad_len = tf.maximum(
#         0,
#         batch_max_train_logits_sequence_length
#         - batch_max_targets_sequence_length,
#     )
#
#     padded_logits = tf.pad(train_logits, [[0, 0], [0, logits_pad_len], [0, 0]])
#     padded_targets = tf.pad(targets, [[0, 0], [0, targets_pad_len]])
#
#     output_exp = tf.cast(tf.reshape(padded_targets, [-1, 1]), tf.int64)
#     sampled_values = sample_values_from_classes(
#         output_exp,
#         loss["sampler"],
#         num_classes,
#         loss["negative_samples"],
#         loss["unique"],
#         loss["class_counts"],
#         loss["distortion"],
#     )
#
#     if loss["sampler"] == "fixed_unigram":
#         # regenerate sampled_values structure for specified samplers
#         # to handle any zero values in true_expected_count tensor
#         sampled_values = FixedUnigramCandidateSampler(
#             sampled_values.sampled_candidates,
#             # add smoothing constant EPSILON to handle any zero values
#             tf.add(sampled_values.true_expected_count, EPSILON),
#             sampled_values.sampled_expected_count,
#         )
#
#     def _sampled_loss(labels, logits):
#         labels = tf.cast(labels, tf.int64)
#         labels = tf.reshape(labels, [-1, 1])
#         logits = tf.cast(logits, tf.float32)
#
#         return tf.cast(
#             tf.nn.sampled_softmax_loss(
#                 weights=tf.transpose(decoder_weights),
#                 biases=decoder_biases,
#                 labels=labels,
#                 inputs=logits,
#                 num_sampled=loss["negative_samples"],
#                 num_classes=num_classes,
#                 sampled_values=sampled_values,
#             ),
#             tf.float32,
#         )
#
#     train_loss = tfa.seq2seq.sequence_loss(
#         padded_logits,
#         padded_targets,
#         tf.sequence_mask(
#             targets_sequence_length,
#             tf.shape(padded_targets)[1],
#             dtype=tf.float32,
#         ),
#         average_across_timesteps=True,
#         average_across_batch=False,
#         softmax_loss_function=_sampled_loss,
#     )
#
#     return train_loss
#
#
