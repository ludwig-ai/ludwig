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
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa

from tensorflow.python.ops.losses.losses_impl import Reduction


#
# Custom classes to support Tensorflow 2
#
class BWCEWLoss(tf.keras.losses.Loss):
    def __init__(
            self,
            positive_class_weight=1,
            robust_lambda=0,
            confidence_penalty=0
    ):
        super(BWCEWLoss, self).__init__()

        self.positive_class_weight = positive_class_weight
        self.robust_lambda = robust_lambda
        self.confidence_penalty = confidence_penalty

    def call(self, y_true, y_pred):
        logits = y_pred

        # weighted cross entropy
        train_loss = tf.nn.weighted_cross_entropy_with_logits(
            targets=tf.cast(y_true, tf.float32),
            logits=logits,
            pos_weight=self.positive_class_weight
        )

        # robust lambda
        if self.robust_lambda > 0:
            train_loss = ((1 - self.robust_lambda) * train_loss +
                          self.robust_lambda / 2)

        train_mean_loss = tf.reduce_mean(
            train_loss
        )

        # confidence penalty
        if self.confidence_penalty > 0:
            probabilities = tf.nn.sigmoid(logits)
            mean_penalty = mean_confidence_penalty(probabilities, 2)
            train_mean_loss += self.confidence_penalty * mean_penalty

        return train_mean_loss

class SoftmaxCrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(
            self,
            num_classes=0,
            feature_loss=None,
            name=None
    ):
        super(SoftmaxCrossEntropyLoss, self).__init__(name=name)
        self.num_classes = num_classes
        self.feature_loss = feature_loss

    def call(self, y, y_pred):
        vector_labels = tf.one_hot(
            tf.cast(y, dtype=tf.int32),
            self.num_classes
        )

        loss = weighted_softmax_cross_entropy(
            y_pred,
            vector_labels,
            **self.feature_loss
        )

        return loss

# todo tf2: work-in-progress partial implementation
class SampledSoftmaxCrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(
            self,
            decoder_obj=None,
            num_classes=0,
            feature_loss=None,
            name=None
    ):
        super(SampledSoftmaxCrossEntropyLoss, self).__init__(name=name)

        self.decoder_obj = decoder_obj
        self.num_classes = num_classes
        self.feature_loss = feature_loss

    def call(self, y, y_pred):
        vector_labels = tf.one_hot(
            tf.cast(y, dtype=tf.int32),
            self.num_classes
        )

        decoder_weights = self.decoder_obj.get_weights()[0]
        decoder_biases = self.decoder_obj.get_weights()[1]

        loss, _ = sampled_softmax_cross_entropy(
            y,
            y_pred['logits'],
            num_classes=self.num_classes,
            decoder_weights=decoder_weights,
            decoder_biases=decoder_biases,
            last_hidden=y_pred['last_hidden'],
            **self.feature_loss
        )

        return loss


# end of custom classes


def softmax_cross_entropy_with_class_weighting(logits, one_hot_labels,
                                               class_weights,
                                               labels_smoothing=0.0):
    class_weights_const = tf.expand_dims(
        tf.constant(class_weights, dtype=tf.float32), 0)
    sample_weights = tf.reduce_sum(
        tf.multiply(one_hot_labels, class_weights_const), 1)
    return tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels,
                                           logits=logits,
                                           label_smoothing=labels_smoothing,
                                           weights=sample_weights,
                                           reduction=tf.losses.Reduction.NONE)


def sigmoid_cross_entropy_with_class_weighting(logits, multi_class_labels,
                                               class_weights,
                                               labels_smoothing=0.0):
    class_weights_const = tf.expand_dims(
        tf.constant(class_weights, dtype=tf.float32), 0)
    sample_weights = tf.reduce_sum(
        tf.multiply(multi_class_labels, class_weights_const), 1)
    return tf.losses.sigmoid_cross_entropy(
        multi_class_labels=multi_class_labels,
        logits=logits,
        label_smoothing=labels_smoothing,
        weights=sample_weights,
        reduction=tf.losses.Reduction.NONE)


def mean_confidence_penalty(probabilities, num_classes):
    max_entropy = tf.constant(np.log(num_classes), dtype=tf.float32)
    # clipping needed for avoiding log(0) = -inf
    entropy_per_class = tf.maximum(- probabilities *
                                   tf.log(tf.clip_by_value(probabilities, 1e-10,
                                                           1)), 0)
    entropy = tf.reduce_sum(entropy_per_class, -1)
    penalty = (max_entropy - entropy) / max_entropy
    return tf.reduce_mean(penalty)


def seq2seq_sequence_loss(targets, targets_sequence_length, logits,
                          softmax_function=None):
    batch_max_targets_sequence_length = tf.shape(targets)[1]
    batch_max_logits_sequence_length = tf.shape(logits)[1]
    difference = tf.maximum(0,
                            batch_max_targets_sequence_length - batch_max_logits_sequence_length)
    padded_logits = tf.pad(logits, [[0, 0], [0, difference], [0, 0]])
    padded_logits = padded_logits[:, :batch_max_targets_sequence_length, :]

    with tf.variable_scope('sequence_loss'):
        sequence_loss = tfa.seq2seq.sequence_loss(
            padded_logits,
            targets,
            tf.sequence_mask(targets_sequence_length,
                             batch_max_targets_sequence_length,
                             dtype=tf.float32),
            average_across_timesteps=True,
            average_across_batch=False,
            softmax_loss_function=softmax_function
        )

    # batch_max_seq_length = tf.shape(logits)[1]
    # unpadded_targets = targets[:, :tf.shape(logits)[1]]
    # with tf.variable_scope('sequence_loss'):
    #     sequence_loss = tfa.seq2seq.sequence_loss(
    #         logits,
    #         unpadded_targets,
    #         tf.sequence_mask(targets_sequence_length, batch_max_seq_length, dtype=tf.float32),
    #         average_across_timesteps=True,
    #         average_across_batch=False,
    #         softmax_loss_function=softmax_function
    #     )

    return sequence_loss


# manual implementation of sequence loss
def cross_entropy_sequence_loss(logits, targets, sequence_length):
    """Calculates the per-example cross-entropy loss for a sequence of logits and
      masks out all losses passed the sequence length.
    Args:
      logits: Logits of shape `[B, T, vocab_size]`
      targets: Target classes of shape `[B, T]`
      sequence_length: An int32 tensor of shape `[B]` corresponding
        to the length of each input
    Returns:
      A tensor of shape [T, B] that contains the loss per example, per time step.
    """
    with tf.variable_scope('sequence_loss'):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=targets)
        # Mask out the losses we don't care about
        loss_mask = tf.sequence_mask(
            tf.cast(sequence_length, tf.int32),
            tf.cast(tf.shape(targets)[1], tf.int32)
        )
        losses = losses * tf.cast(loss_mask, tf.float32)
        return losses


def sampled_softmax_cross_entropy(
        vector_labels,
        logits,
        num_classes=1,
        decoder_weights=None,
        decoder_biases=None,
        last_hidden=None,
        sampler=None,
        negative_samples=0,
        class_counts=0,
        distortion=1,
        unique=False,
        labels_smoothing=0,
        **kwargs
):
    output_exp = tf.cast(
        tf.expand_dims(vector_labels, -1),
        tf.int64
    )
    if sampler == 'fixed_unigram':
        sampled_values = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=output_exp,
            num_true=1,
            num_sampled=negative_samples,
            unique=unique,
            range_max=num_classes,
            unigrams=class_counts,
            distortion=distortion
        )
    elif sampler == 'uniform':
        sampled_values = tf.nn.uniform_candidate_sampler(
            true_classes=output_exp,
            num_true=1,
            num_sampled=negative_samples,
            unique=unique,
            range_max=num_classes
        )
    elif sampler == 'log_uniform':
        sampled_values = tf.nn.log_uniform_candidate_sampler(
            true_classes=output_exp,
            num_true=1,
            num_sampled=negative_samples,
            unique=unique,
            range_max=num_classes
        )
    elif sampler == 'learned_unigram':
        sampled_values = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=output_exp,
            num_true=1,
            num_sampled=negative_samples,
            unique=unique,
            range_max=num_classes,
            unigrams=class_counts,
            distortion=distortion
        )
    else:
        raise ValueError('Unsupported sampler {}'.format(sampler))

    train_loss = tf.nn.sampled_softmax_loss(weights=tf.transpose(decoder_weights),
                                            biases=decoder_biases,
                                            labels=output_exp,
                                            inputs=last_hidden,
                                            num_sampled=negative_samples,
                                            num_classes=num_classes,
                                            sampled_values=sampled_values)

    # todo tf2 need to determine how to handle this, possible in separate function
    # eval_loss = tf.losses.softmax_cross_entropy(
    #     onehot_labels=tf.cast(
    #         tf.cast(tf.one_hot(vector_labels, num_classes, axis=-1), dtype=tf.int16)
    #     ),
    #     logits=logits,
    #     label_smoothing=labels_smoothing,
    #     reduction=Reduction.NONE
    # )
    return train_loss, None #, eval_loss todo tf2 clean-up code


def sequence_sampled_softmax_cross_entropy(targets, targets_sequence_length,
                                           eval_logits, train_logits,
                                           class_weights,
                                           class_biases, loss,
                                           num_classes):
    batch_max_targets_sequence_length = tf.shape(targets)[1]

    batch_max_train_logits_sequence_length = tf.shape(train_logits)[1]
    difference_train = batch_max_targets_sequence_length - batch_max_train_logits_sequence_length
    padded_train_logits = tf.pad(train_logits,
                                 [[0, 0], [0, difference_train], [0, 0]])

    batch_max_eval_logits_sequence_length = tf.shape(eval_logits)[1]
    difference_eval = batch_max_targets_sequence_length - batch_max_eval_logits_sequence_length
    padded_eval_logits = tf.pad(eval_logits,
                                [[0, 0], [0, difference_eval], [0, 0]])

    # batch_max_seq_length = tf.shape(train_logits)[1]
    # unpadded_targets = targets[:, :batch_max_seq_length]
    # output_exp = tf.cast(tf.reshape(unpadded_targets, [-1, 1]), tf.int64)
    output_exp = tf.cast(tf.reshape(targets, [-1, 1]), tf.int64)

    if loss['sampler'] == 'fixed_unigram':
        sampled_values = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=output_exp,
            num_true=1,
            num_sampled=loss['negative_samples'],
            unique=loss['unique'],
            range_max=num_classes,
            unigrams=loss['class_counts'],
            distortion=loss['distortion']
        )
    elif loss['sampler'] == 'uniform':
        sampled_values = tf.nn.uniform_candidate_sampler(
            true_classes=output_exp,
            num_true=1,
            num_sampled=loss['negative_samples'],
            unique=loss['unique'],
            range_max=num_classes
        )
    elif loss['sampler'] == 'log_uniform':
        sampled_values = tf.nn.log_uniform_candidate_sampler(
            true_classes=output_exp,
            num_true=1,
            num_sampled=loss['negative_samples'],
            unique=loss['unique'],
            range_max=num_classes
        )
    elif loss['sampler'] == 'learned_unigram':
        sampled_values = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=output_exp,
            num_true=1,
            num_sampled=loss['negative_samples'],
            unique=loss['unique'],
            range_max=num_classes,
            unigrams=loss['class_counts'],
            distortion=loss['distortion']
        )
    else:
        raise ValueError('Unsupported sampler {}'.format(loss['sampler']))

    def _sampled_loss(labels, logits):
        labels = tf.cast(labels, tf.int64)
        labels = tf.reshape(labels, [-1, 1])
        logits = tf.cast(logits, tf.float32)

        return tf.cast(
            tf.nn.sampled_softmax_loss(weights=tf.transpose(class_weights),
                                       biases=class_biases,
                                       labels=labels,
                                       inputs=logits,
                                       num_sampled=loss['negative_samples'],
                                       num_classes=num_classes,
                                       sampled_values=sampled_values),
            tf.float32)

    train_loss = tfa.seq2seq.sequence_loss(
        padded_train_logits,
        targets,
        tf.sequence_mask(targets_sequence_length,
                         batch_max_targets_sequence_length, dtype=tf.float32),
        average_across_timesteps=True,
        average_across_batch=False,
        softmax_loss_function=_sampled_loss
    )

    # batch_max_seq_length_eval = tf.shape(eval_logits)[1]
    # unpadded_targets_eval = targets[:, :batch_max_seq_length_eval]

    eval_loss = tfa.seq2seq.sequence_loss(
        padded_eval_logits,
        targets,
        tf.sequence_mask(targets_sequence_length,
                         batch_max_targets_sequence_length, dtype=tf.float32),
        average_across_timesteps=True,
        average_across_batch=False
    )

    return train_loss, eval_loss


def weighted_softmax_cross_entropy(
        logits,
        vector_labels,
        class_weights=1,
        labels_smoothing=0,
        **kwargs
):
    use_class_weights = not isinstance(class_weights, (int, float))
    if use_class_weights:
        train_loss = softmax_cross_entropy_with_class_weighting(
            logits,
            vector_labels,
            class_weights,
            labels_smoothing
        )
    else:
        train_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=vector_labels,
            logits=logits,
            label_smoothing=labels_smoothing,
            reduction=Reduction.NONE)
    return train_loss


def loss_multilabel(logits, vector_labels, loss):
    # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
    # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
    # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    use_class_weights = not isinstance(loss['class_weights'], (int, float))
    if use_class_weights:
        train_loss = sigmoid_cross_entropy_with_class_weighting(
            logits,
            vector_labels,
            loss['class_weights'],
            loss['labels_smoothing']
        )
    else:
        train_loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=vector_labels,
            logits=logits,
            label_smoothing=loss[
                'labels_smoothing'],
            reduction=Reduction.NONE)
    return train_loss


def absolute_loss(y, y_hat):
    return tf.abs(tf.subtract(y, y_hat))


def squared_loss(y, y_hat):
    return tf.square(tf.subtract(y, y_hat))


def mean_absolute_error(y, y_hat, weight=1.0):
    return tf.reduce_mean(tf.multiply(absolute_loss(y, y_hat), weight))


def mean_squared_error(y, y_hat, weight=1.0):
    return tf.reduce_mean(tf.multiply(squared_loss(y, y_hat), weight))


# todo tf2: fix this!
# regularizer_registry = {'l1': tf2.keras.regularizers.l1,
#                         'l2': tf2.keras.regularizers.l2,
#                         'l1_l2': tf2.keras.regularizers.l1_l2,
#                         'None': lambda x: None,
#                         None: lambda x: None}
regularizer_registry = {'l1': lambda x: None,
                        'l2': lambda x: None,
                        'l1_l2': lambda x: None,
                        'None': lambda x: None,
                        None: lambda x: None}
