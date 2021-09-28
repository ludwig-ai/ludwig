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

import torch
from torch import nn
from torch.nn import (MSELoss as _MSELoss, L1Loss)

from ludwig.constants import LOGITS
# from ludwig.utils.tf_utils import sequence_length_2D

# used for Laplace smoothing for candidate samplers

EPSILON = 1.0e-10


class LogitsLoss(nn.Module):
    def __call__(self, input, target):
        if isinstance(input, dict) and LOGITS in input:
            input = input[LOGITS]
        return super().__call__(input, target)


class MSELoss(LogitsLoss, _MSELoss):
    pass


class MAELoss(LogitsLoss, L1Loss):
    pass


class RMSELoss(LogitsLoss):
    def __init__(self, **kwargs):
        super().__init__()
        self.mse = nn.MSELoss(**kwargs)

    def forward(self, input, target):
        return torch.sqrt(self.mse(input, target))


class RMSPELoss(LogitsLoss):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, input, target):
        loss = rmspe_loss(target, input)
        return loss


# class BWCEWLoss(tf.keras.losses.Loss):
#     def __init__(
#         self, positive_class_weight=1, robust_lambda=0, confidence_penalty=0
#     ):
#         super().__init__()
#
#         self.positive_class_weight = positive_class_weight
#         self.robust_lambda = robust_lambda
#         self.confidence_penalty = confidence_penalty
#
#     def call(self, y_true, y_pred):
#         logits = y_pred[LOGITS]
#
#         # weighted cross entropy
#         train_loss = tf.nn.weighted_cross_entropy_with_logits(
#             labels=tf.cast(y_true, tf.float32),
#             logits=logits,
#             pos_weight=self.positive_class_weight,
#         )
#
#         # robust lambda
#         if self.robust_lambda > 0:
#             train_loss = (
#                 1 - self.robust_lambda
#             ) * train_loss + self.robust_lambda / 2
#
#         train_mean_loss = tf.reduce_mean(train_loss)
#
#         # confidence penalty
#         if self.confidence_penalty > 0:
#             probabilities = tf.nn.sigmoid(logits)
#             mean_penalty = mean_confidence_penalty(probabilities, 2)
#             train_mean_loss += self.confidence_penalty * mean_penalty
#
#         return train_mean_loss


# TODO torch: test behavior parity with tf
class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(**kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        target = target.long()
        print(f'preds: {input.shape} {input.dtype} {input}')
        print(f'target: {target.shape} {target.dtype} {target}')
        return self.loss_fn(input, target)
        # vector_labels = tf.one_hot(
        #     tf.cast(y, dtype=tf.int64), self.num_classes
        # )
        #
        # loss = weighted_softmax_cross_entropy(
        #     y_pred[LOGITS], vector_labels, **self.feature_loss
        # )
        #
        # return loss


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
# class SigmoidCrossEntropyLoss(tf.keras.losses.Loss):
#     def __init__(self, feature_loss=None, name=None):
#         super().__init__(name=name)
#         self.feature_loss = feature_loss
#
#     def call(self, y, y_pred):
#         loss = weighted_sigmoid_cross_entropy(
#             y_pred[LOGITS], tf.cast(y, tf.float32), **self.feature_loss
#         )
#         return loss
#
#
# class SequenceSoftmaxCrossEntropyLoss(tf.keras.losses.Loss):
#     def __init__(self, name=None, from_logits=True, **kwargs):
#         super().__init__(name=name)
#         self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
#             from_logits=from_logits, reduction="none"
#         )
#         self.from_logits = from_logits
#
#     def call(self, y_true, y_pred):
#         # y_true: shape [batch_size, sequence_size]
#         # y_pred: shape [batch_size, sequence_size, num_classes]
#
#         if self.from_logits:
#             y_pred_tensor = y_pred[LOGITS]
#         else:
#             y_pred_tensor = y_pred[PROBABILITIES]
#         y_true_tensor = tf.cast(y_true, dtype=tf.int64)
#
#         # pad the shorter sequence (tensor shape 1)
#         y_pred_tensor_len = tf.shape(y_pred_tensor)[1]
#         y_true_tensor_len = tf.shape(y_true_tensor)[1]
#
#         y_pred_pad_len = tf.maximum(0, y_true_tensor_len - y_pred_tensor_len)
#         y_true_pad_len = tf.maximum(0, y_pred_tensor_len - y_true_tensor_len)
#
#         y_pred_tensor = tf.pad(
#             y_pred_tensor, [[0, 0], [0, y_pred_pad_len], [0, 0]]
#         )
#         y_true_tensor = tf.pad(y_true_tensor, [[0, 0], [0, y_true_pad_len]])
#
#         y_true_seq_len = sequence_length_2D(y_true_tensor)
#         # longest_sequence_length = tf.maximum(y_true_seq_len,
#         #                                     sequence_length_3D(y_pred_tensor))
#         # longest_sequence_length = tf.minimum(longest_sequence_length,
#         #                                     y_true_seq_len)
#         # longest_sequence_length += 2  # for EOS
#
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
# def softmax_cross_entropy_with_class_weighting(
#     logits, one_hot_labels, class_weights, labels_smoothing=0.0
# ):
#     class_weights_const = tf.expand_dims(
#         tf.constant(class_weights, dtype=tf.float32), 0
#     )
#     sample_weights = tf.reduce_sum(
#         tf.multiply(one_hot_labels, class_weights_const), 1
#     )
#     return tf.compat.v1.losses.softmax_cross_entropy(
#         onehot_labels=one_hot_labels,
#         logits=logits,
#         label_smoothing=labels_smoothing,
#         weights=sample_weights,
#         reduction=tf.losses.Reduction.NONE,
#     )
#
#
# def sigmoid_cross_entropy_with_class_weighting(
#     logits, multi_class_labels, class_weights, labels_smoothing=0.0
# ):
#     class_weights_const = tf.expand_dims(
#         tf.constant(class_weights, dtype=tf.float32), 0
#     )
#     sample_weights = tf.multiply(multi_class_labels, class_weights_const)
#     return tf.compat.v1.losses.sigmoid_cross_entropy(
#         multi_class_labels=multi_class_labels,
#         logits=logits,
#         label_smoothing=labels_smoothing,
#         weights=sample_weights,
#         reduction=tf.losses.Reduction.NONE,
#     )
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
# def weighted_softmax_cross_entropy(
#     logits, vector_labels, class_weights=1, labels_smoothing=0, **kwargs
# ):
#     use_class_weights = not isinstance(class_weights, (int, float))
#     if use_class_weights:
#         loss = softmax_cross_entropy_with_class_weighting(
#             logits, vector_labels, class_weights, labels_smoothing
#         )
#     else:
#         loss = tf.keras.losses.categorical_crossentropy(
#             y_true=vector_labels,
#             y_pred=logits,
#             from_logits=True,
#             label_smoothing=labels_smoothing,
#         )
#     return loss
#
#
# def weighted_sigmoid_cross_entropy(
#     logits, vector_labels, class_weights=1, labels_smoothing=0, **kwargs
# ):
#     use_class_weights = not isinstance(class_weights, (int, float))
#     if use_class_weights:
#         loss = sigmoid_cross_entropy_with_class_weighting(
#             logits, vector_labels, class_weights, labels_smoothing
#         )
#     else:
#         loss = tf.nn.sigmoid_cross_entropy_with_logits(
#             labels=vector_labels,
#             logits=logits,
#             # labels_smoothing=labels_smoothing  # todo reintroduce
#         )
#     return loss
#
#
# # used for categorical and sequence features
# def sample_values_from_classes(
#     labels,
#     sampler,
#     num_classes,
#     negative_samples,
#     unique,
#     class_counts,
#     distortion,
# ):
#     """returns sampled_values using the chosen sampler"""
#     if sampler == "fixed_unigram":
#         sampled_values = tf.random.fixed_unigram_candidate_sampler(
#             true_classes=labels,
#             num_true=1,
#             num_sampled=negative_samples,
#             unique=unique,
#             range_max=num_classes,
#             unigrams=class_counts,
#             distortion=distortion,
#         )
#     elif sampler == "uniform":
#         sampled_values = tf.random.uniform_candidate_sampler(
#             true_classes=labels,
#             num_true=1,
#             num_sampled=negative_samples,
#             unique=unique,
#             range_max=num_classes,
#         )
#     elif sampler == "log_uniform":
#         sampled_values = tf.random.log_uniform_candidate_sampler(
#             true_classes=labels,
#             num_true=1,
#             num_sampled=negative_samples,
#             unique=unique,
#             range_max=num_classes,
#         )
#     elif sampler == "learned_unigram":
#         sampled_values = tf.random.learned_unigram_candidate_sampler(
#             true_classes=labels,
#             num_true=1,
#             num_sampled=negative_samples,
#             unique=unique,
#             range_max=num_classes,
#         )
#     else:
#         raise ValueError("Unsupported sampler {}".format(sampler))
#     return sampled_values


def rmspe_loss(targets, predictions):
    loss = torch.sqrt(
        torch.mean(
            torch.square(torch.divide(
                torch.subtract(targets, predictions), targets))
        )
    )
    return loss


class SigmoidCrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, y: torch.Tensor, y_pred: torch.Tensor):
        return self.loss_fn(
            y_pred.type(torch.float32),
            y[LOGITS].type(torch.float32)
        )
