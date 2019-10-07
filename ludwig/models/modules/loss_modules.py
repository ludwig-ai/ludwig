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
import tensorflow as tf
from tensorflow.python.ops.losses.losses_impl import Reduction


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

    with tf.compat.v1.variable_scope('sequence_loss'):
        sequence_loss = tf.contrib.seq2seq.sequence_loss(
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
    # with tf.compat.v1.variable_scope('sequence_loss'):
    #     sequence_loss = tf.contrib.seq2seq.sequence_loss(
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
    with tf.compat.v1.variable_scope('sequence_loss'):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=targets)
        # Mask out the losses we don't care about
        loss_mask = tf.sequence_mask(
            tf.cast(sequence_length, tf.int32),
            tf.cast(tf.shape(targets)[1], tf.int32)
        )
        losses = losses * tf.cast(loss_mask, tf.float32)
        return losses


def sampled_softmax_cross_entropy(output_placeholder, feature_hidden, logits,
                                  vector_labels, class_weights,
                                  class_biases, loss, num_classes):
    output_exp = tf.cast(tf.expand_dims(output_placeholder, -1), tf.int64)
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

    train_loss = tf.nn.sampled_softmax_loss(weights=tf.transpose(class_weights),
                                            biases=class_biases,
                                            labels=output_exp,
                                            inputs=feature_hidden,
                                            num_sampled=loss[
                                                'negative_samples'],
                                            num_classes=num_classes,
                                            sampled_values=sampled_values)
    eval_loss = tf.losses.softmax_cross_entropy(onehot_labels=vector_labels,
                                                logits=logits,
                                                label_smoothing=loss[
                                                    'labels_smoothing'],
                                                reduction=Reduction.NONE)
    return train_loss, eval_loss


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

    train_loss = tf.contrib.seq2seq.sequence_loss(
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

    eval_loss = tf.contrib.seq2seq.sequence_loss(
        padded_eval_logits,
        targets,
        tf.sequence_mask(targets_sequence_length,
                         batch_max_targets_sequence_length, dtype=tf.float32),
        average_across_timesteps=True,
        average_across_batch=False
    )

    return train_loss, eval_loss


def weighted_softmax_cross_entropy(logits, vector_labels, loss):
    use_class_weights = not isinstance(loss['class_weights'], (int, float))
    if use_class_weights:
        train_loss = softmax_cross_entropy_with_class_weighting(
            logits,
            vector_labels,
            loss['class_weights'],
            loss['labels_smoothing']
        )
    else:
        train_loss = tf.compat.v1.losses.softmax_cross_entropy(
            onehot_labels=vector_labels,
            logits=logits,
            label_smoothing=loss[
                'labels_smoothing'],
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


regularizer_registry = {'l1': tf.contrib.layers.l1_regularizer,
                        'l2': tf.contrib.layers.l2_regularizer,
                        'sum': tf.contrib.layers.sum_regularizer,
                        'None': lambda x: None,
                        None: lambda x: None}
