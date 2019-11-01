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
import logging

import tensorflow as tf

from ludwig.models.modules.initializer_modules import get_initializer

logger = logging.getLogger(__name__)


def conv_1d(inputs, weights, biases,
            stride=1, padding='SAME',
            activation='relu', norm=None,
            dropout=False, dropout_rate=None,
            is_training=True):
    hidden = tf.nn.conv1d(tf.cast(inputs, tf.float32), weights, stride=stride,
                          padding=padding) + biases

    if norm is not None:
        if norm == 'batch':
            hidden = tf.layers.batch_normalization(
                hidden,
                training=is_training
            )
        elif norm == 'layer':
            hidden = tf.contrib.layers.layer_norm(hidden)

    if activation:
        hidden = getattr(tf.nn, activation)(hidden)

    if dropout and dropout_rate is not None:
        hidden = tf.layers.dropout(hidden, rate=dropout_rate,
                                   training=is_training)

    return hidden


def conv_2d(inputs, weights, biases,
            stride=1, padding='SAME',
            activation='relu', norm=None,
            dropout=False, dropout_rate=None,
            is_training=True):
    hidden = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1],
                          padding=padding) + biases

    if norm is not None:
        if norm == 'batch':
            hidden = tf.layers.batch_normalization(
                hidden,
                training=is_training
            )
        elif norm == 'layer':
            hidden = tf.contrib.layers.layer_norm(hidden)

    if activation:
        hidden = getattr(tf.nn, activation)(hidden)

    if dropout and dropout_rate is not None:
        hidden = tf.layers.dropout(hidden, rate=dropout_rate,
                                   training=is_training)

    return hidden


def conv_layer(inputs, kernel_shape, biases_shape,
               stride=1, padding='SAME', activation='relu', norm=None,
               dropout=False, dropout_rate=None, regularizer=None,
               initializer=None,
               dimensions=2, is_training=True):
    if initializer is not None:
        initializer_obj = get_initializer(initializer)
        weights = tf.compat.v1.get_variable(
            'weights',
            initializer=initializer_obj(kernel_shape),
            regularizer=regularizer
        )
    else:
        if activation == 'relu':
            initializer = get_initializer('he_uniform')
        elif activation == 'sigmoid' or activation == 'tanh':
            initializer = get_initializer('glorot_uniform')
        # if initializer is None, tensorFlow seems to be using
        # a glorot uniform initializer
        weights = tf.compat.v1.get_variable(
            'weights',
            kernel_shape,
            regularizer=regularizer,
            initializer=initializer
        )
    logger.debug('  conv_weights: {0}'.format(weights))

    biases = tf.compat.v1.get_variable('biases', biases_shape,
                             initializer=tf.constant_initializer(0.01))
    logger.debug('  conv_biases: {0}'.format(biases))

    if dimensions == 1:
        return conv_1d(inputs, weights, biases,
                       stride=stride,
                       padding=padding,
                       activation=activation,
                       norm=norm,
                       dropout=dropout,
                       dropout_rate=dropout_rate,
                       is_training=is_training)
    elif dimensions == 2:
        return conv_2d(inputs, weights, biases,
                       stride=stride,
                       padding=padding,
                       activation=activation,
                       norm=norm,
                       dropout=dropout,
                       dropout_rate=dropout_rate,
                       is_training=is_training)
    else:
        raise Exception('Unsupported number of dimensions', dimensions)


def conv_1d_layer(inputs, kernel_shape, biases_shape, stride=1, padding='SAME',
                  activation='relu', norm=None,
                  dropout=False, dropout_rate=None, regularizer=None,
                  initializer=None, is_training=True):
    return conv_layer(inputs, kernel_shape, biases_shape, stride=stride,
                      padding=padding, activation=activation,
                      norm=norm, dropout=dropout, dropout_rate=dropout_rate,
                      regularizer=regularizer,
                      initializer=initializer,
                      dimensions=1, is_training=is_training)


def conv_2d_layer(inputs, kernel_shape, biases_shape, stride=1, padding='SAME',
                  activation='relu', norm=None,
                  dropout=False, dropout_rate=None, regularizer=None,
                  initializer=None, is_training=True):
    return conv_layer(inputs, kernel_shape, biases_shape, stride=stride,
                      padding=padding, activation=activation,
                      norm=norm, dropout=dropout, dropout_rate=dropout_rate,
                      regularizer=regularizer,
                      initializer=initializer,
                      dimensions=2, is_training=is_training)


def flatten(hidden, skip_first=True):
    hidden_size = 1
    # if hidden is activation, the first dimension is the batch_size
    start = 1 if skip_first else 0
    for x in hidden.shape[start:]:
        if x.value is not None:
            hidden_size *= x.value
    hidden = tf.reshape(hidden, [-1, hidden_size], name='flatten')
    logger.debug('  flatten hidden: {0}'.format(hidden))
    return hidden, hidden_size


class ConvStack1D:

    def __init__(
            self,
            layers=None,
            num_layers=None,
            default_filter_size=3,
            default_num_filters=64,
            default_pool_size=None,
            default_activation='relu',
            default_norm=None,
            default_stride=1,
            default_pool_stride=1,
            default_dropout=False,
            default_initializer=None,
            default_regularize=True
    ):
        if layers is None:
            if num_layers is None:
                self.layers = [
                    {'filter_size': 7, 'pool_size': 3, 'regularize': False},
                    {'filter_size': 7, 'pool_size': 3, 'regularize': False},
                    {'filter_size': 3, 'pool_size': None, 'regularize': False},
                    {'filter_size': 3, 'pool_size': None, 'regularize': False},
                    {'filter_size': 3, 'pool_size': None, 'regularize': True},
                    {'filter_size': 3, 'pool_size': 3, 'regularize': True}
                ]
            else:
                self.layers = []
                for i in range(num_layers):
                    self.layers.append({
                        'filter_size': default_filter_size,
                        'num_filters': default_num_filters,
                        'pool_size': default_pool_size,
                        'pool_stride': default_pool_stride}
                    )
        else:
            self.layers = layers

        for layer in self.layers:
            if 'filter_size' not in layer:
                layer['filter_size'] = default_filter_size
            if 'num_filters' not in layer:
                layer['num_filters'] = default_num_filters
            if 'activation' not in layer:
                layer['activation'] = default_activation
            if 'norm' not in layer:
                layer['norm'] = default_norm
            if 'stride' not in layer:
                layer['stride'] = default_stride
            if 'pool_stride' not in layer:
                layer['pool_stride'] = default_pool_stride
            if 'dropout' not in layer:
                layer['dropout'] = default_dropout
            if 'initializer' not in layer:
                layer['initializer'] = default_initializer
            if 'regularize' not in layer:
                layer['regularize'] = default_regularize

    def __call__(
            self,
            input_sequence,
            input_size,
            regularizer,
            dropout_rate,
            is_training=True
    ):
        hidden = input_sequence
        prev_num_filters = input_size
        num_conv_layers = len(self.layers)

        for i in range(num_conv_layers):
            layer = self.layers[i]
            with tf.compat.v1.variable_scope('conv_{}'.format(i)):
                # Convolution Layer
                filter_shape = [
                    layer['filter_size'],
                    prev_num_filters,
                    layer['num_filters']
                ]
                layer_output = conv_1d_layer(hidden, filter_shape,
                                             [layer['num_filters']],
                                             stride=layer['stride'],
                                             padding='SAME',
                                             activation=layer['activation'],
                                             norm=layer['norm'],
                                             dropout=layer['dropout'],
                                             dropout_rate=dropout_rate,
                                             regularizer=regularizer if layer[
                                                 'regularize'] else None,
                                             is_training=is_training)
                prev_num_filters = layer['num_filters']
                logger.debug('  conv_{}: {}'.format(i, layer_output))

                # Pooling
                if layer['pool_size'] is not None:
                    layer_output = tf.layers.max_pooling1d(
                        layer_output,
                        pool_size=layer['pool_size'],
                        strides=layer['pool_stride'],
                        padding='VALID',
                        name='pool_{}'.format(layer['filter_size'])
                    )
                    logger.debug('  pool_{}: {}'.format(
                        i, layer_output))

            hidden = layer_output
            logger.debug('  hidden: {0}'.format(hidden))

        return hidden


class ParallelConv1D:

    def __init__(
            self,
            layers=None,
            default_filter_size=3,
            default_num_filters=256,
            default_pool_size=None,
            default_activation='relu',
            default_norm=None,
            default_stride=1,
            default_pool_stride=1,
            default_dropout=False,
            default_initializer=None,
            default_regularize=True
    ):
        if layers is None:
            self.layers = [
                {'filter_size': 2},
                {'filter_size': 3},
                {'filter_size': 4},
                {'filter_size': 5}
            ]
        else:
            self.layers = layers
        for layer in layers:
            if 'filter_size' not in layer:
                layer['filter_size'] = default_filter_size
            if 'num_filters' not in layer:
                layer['num_filters'] = default_num_filters
            if 'pool_size' not in layer:
                layer['pool_size'] = default_pool_size
            if 'activation' not in layer:
                layer['activation'] = default_activation
            if 'norm' not in layer:
                layer['norm'] = default_norm
            if 'stride' not in layer:
                layer['stride'] = default_stride
            if 'pool_stride' not in layer:
                layer['pool_stride'] = default_pool_stride
            if 'dropout' not in layer:
                layer['dropout'] = default_dropout
            if 'initializer' not in layer:
                layer['initializer'] = default_initializer
            if 'regularize' not in layer:
                layer['regularize'] = default_regularize

    def __call__(
            self,
            input_layer,
            input_size,
            regularizer,
            dropout_rate,
            is_training=True
    ):
        hidden = input_layer
        prev_output_size = input_size
        parallel_conv_layers = []

        for i, layer in enumerate(self.layers):
            with tf.compat.v1.variable_scope(
                    'conv_{}_fs{}'.format(i, layer['filter_size'])):
                # Convolution Layer
                filter_shape = [
                    layer['filter_size'],
                    prev_output_size,
                    layer['num_filters']
                ]
                layer_output = conv_1d_layer(hidden, filter_shape,
                                             [layer['num_filters']],
                                             stride=layer['stride'],
                                             padding='SAME',
                                             activation=layer['activation'],
                                             norm=layer['norm'],
                                             dropout=layer['dropout'],
                                             dropout_rate=dropout_rate,
                                             regularizer=regularizer if layer[
                                                 'regularize'] else None,
                                             is_training=is_training)

                logger.debug('  conv_{}_fs{}: {}'.format(
                    i,
                    layer['filter_size'],
                    layer_output)
                )

                # Pooling
                if layer['pool_size'] is not None:
                    layer_output = tf.layers.max_pooling1d(
                        layer_output,
                        pool_size=layer['pool_size'],
                        strides=layer['pool_stride'],
                        padding='VALID',
                        name='pool_{}'.format(layer['filter_size'])
                    )
                    logger.debug('  pool_{}_fs{}: {}'.format(
                        i,
                        layer['filter_size'],
                        layer_output
                    ))

                parallel_conv_layers.append(layer_output)

        hidden = tf.concat(parallel_conv_layers, 2)
        return hidden


class StackParallelConv1D:

    def __init__(
            self,
            stacked_layers=None,
            default_filter_size=3,
            default_num_filters=64,
            default_pool_size=None,
            default_activation='relu',
            default_norm=None,
            default_stride=1,
            default_pool_stride=1,
            default_dropout=False,
            default_initializer=None,
            default_regularize=True
    ):
        if stacked_layers is None:
            self.stacked_parallel_layers = [
                [
                    {'filter_size': 2},
                    {'filter_size': 3},
                    {'filter_size': 4},
                    {'filter_size': 5}
                ],
                [
                    {'filter_size': 2},
                    {'filter_size': 3},
                    {'filter_size': 4},
                    {'filter_size': 5}
                ],
                [
                    {'filter_size': 2},
                    {'filter_size': 3},
                    {'filter_size': 4},
                    {'filter_size': 5}
                ]
            ]

        else:
            self.stacked_parallel_layers = stacked_layers

        for i, parallel_layers in enumerate(self.stacked_parallel_layers):
            for j in range(len(parallel_layers)):
                layer = parallel_layers[j]
                if 'filter_size' not in layer:
                    layer['filter_size'] = default_filter_size
                if 'num_filters' not in layer:
                    layer['num_filters'] = default_num_filters
                if 'pool_size' not in layer:
                    layer['pool_size'] = default_pool_size
                if 'activation' not in layer:
                    layer['activation'] = default_activation
                if 'norm' not in layer:
                    layer['norm'] = default_norm
                if 'stride' not in layer:
                    layer['stride'] = default_stride
                if 'pool_stride' not in layer:
                    layer['pool_stride'] = default_pool_stride
                if i == len(self.stacked_parallel_layers) - 1:
                    layer['pool'] = False
                if 'dropout' not in layer:
                    layer['dropout'] = default_dropout
                if 'initializer' not in layer:
                    layer['initializer'] = default_initializer
                if 'regularize' not in layer:
                    layer['regularize'] = default_regularize

    def __call__(
            self,
            input_layer,
            input_size,
            regularizer,
            dropout_rate,
            is_training=True
    ):
        hidden = input_layer
        prev_num_filters = input_size

        for i in range(len(self.stacked_parallel_layers)):
            parallel_conv_layers = self.stacked_parallel_layers[i]
            with tf.compat.v1.variable_scope('parallel_conv_{}'.format(i)):
                parallel_conv_1d = ParallelConv1D(parallel_conv_layers)
                hidden = parallel_conv_1d(
                    hidden,
                    prev_num_filters,
                    regularizer=regularizer,
                    dropout_rate=dropout_rate,
                    is_training=is_training
                )
                logger.debug('  hidden: {}'.format(hidden))

            prev_num_filters = 0
            for layer in parallel_conv_layers:
                prev_num_filters += layer['num_filters']

        return hidden


class ConvStack2D:

    def __init__(
            self,
            layers=None,
            num_layers=None,
            default_filter_size=3,
            default_num_filters=256,
            default_pool_size=2,
            default_activation='relu',
            default_stride=2,
            default_pool_stride=None,
            default_norm=None,
            default_dropout=False,
            default_initializer=None,
            default_regularize=True
    ):
        if layers is None:
            if num_layers is None:
                self.layers = [
                    {'num_filters': 32},
                    {'num_filters': 64, 'dropout': True},
                ]
            else:
                self.layers = []
                for i in range(num_layers):
                    self.layers.append({
                        'filter_size': default_filter_size,
                        'num_filters': default_num_filters,
                        'pool_size': default_pool_size}
                    )
        else:
            self.layers = layers

        for layer in self.layers:
            if 'num_filters' not in layer:
                layer['num_filters'] = default_num_filters
            if 'filter_size' not in layer:
                layer['filter_size'] = default_filter_size
            if 'pool_size' not in layer:
                layer['pool_size'] = default_pool_size
            if 'activation' not in layer:
                layer['activation'] = default_activation
            if 'stride' not in layer:
                layer['stride'] = default_stride
            if 'pool_stride' not in layer:
                layer['pool_stride'] = default_pool_stride
            if 'norm' not in layer:
                layer['norm'] = default_norm
            if 'dropout' not in layer:
                layer['dropout'] = default_dropout
            if 'initializer' not in layer:
                layer['initializer'] = default_initializer
            if 'regularize' not in layer:
                layer['regularize'] = default_regularize

    def __call__(
            self,
            input_image,
            regularizer,
            dropout_rate,
            is_training=True
    ):
        num_layers = len(self.layers)
        hidden = input_image
        prev_num_filters = int(input_image.shape[-1])
        for i in range(num_layers):
            layer = self.layers[i]
            with tf.compat.v1.variable_scope('conv_{}'.format(i)):
                # Convolution Layer
                filter_shape = [
                    layer['filter_size'],
                    layer['filter_size'],
                    prev_num_filters,
                    layer['num_filters']
                ]
                layer_output = conv_2d_layer(hidden, filter_shape,
                                             [layer['num_filters']],
                                             stride=layer['stride'],
                                             padding='SAME',
                                             activation=layer['activation'],
                                             norm=layer['norm'],
                                             dropout=layer['dropout'],
                                             dropout_rate=dropout_rate,
                                             regularizer=regularizer if layer[
                                                 'regularize'] else None,
                                             initializer=layer['initializer'],
                                             is_training=is_training)
                prev_num_filters = layer['num_filters']
                logger.debug('  conv_{}: {}'.format(i, layer_output))

                # Pooling
                if layer['pool_size'] is not None:
                    pool_size = layer['pool_size']
                    pool_kernel_size = [1, pool_size, pool_size, 1]
                    pool_stride = layer['pool_stride']
                    if pool_stride is not None:
                        pool_kernel_stride = [1, pool_size, pool_size, 1]
                    else:
                        pool_kernel_stride = [1, pool_stride, pool_stride, 1]
                    layer_output = tf.nn.max_pool(
                        layer_output,
                        ksize=pool_kernel_size,
                        strides=pool_kernel_stride,
                        padding='SAME',
                        name='pool_{}'.format(i)
                    )

                    logger.debug('  pool_{}: {}'.format(
                        i, layer_output))

            hidden = layer_output
            logger.debug('  hidden: {0}'.format(hidden))

        return hidden


################################################################################
# The following code for ResNet is adapted from the TensorFlow implementation
# https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
################################################################################

################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def resnet_batch_norm(inputs, is_training,
                      batch_norm_momentum=0.9, batch_norm_epsilon=0.001):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    # Original implementation default values:
    # _BATCH_NORM_DECAY = 0.997
    # _BATCH_NORM_EPSILON = 1e-5
    # they lead to a big difference between the loss
    # at train and prediction time
    return tf.layers.batch_normalization(
        inputs=inputs, axis=3,
        momentum=batch_norm_momentum, epsilon=batch_norm_epsilon, center=True,
        scale=True, training=is_training, fused=True)


def fixed_padding(inputs, kernel_size):
    """Pads the input along the spatial dimensions independently of input size.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides,
                         regularizer=None):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size)

    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        kernel_regularizer=regularizer)


resnet_choices = {
    8: [1, 2, 2],
    14: [1, 2, 2],
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3]
}


def get_resnet_block_sizes(resnet_size):
    """Retrieve the size of each block_layer in the ResNet model.
    The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.
    Args:
      resnet_size: The number of convolutional layers needed in the model.
    Returns:
      A list of block sizes to use in building the model.
    Raises:
      KeyError: if invalid resnet_size is received.
    """
    try:
        return resnet_choices[resnet_size]
    except KeyError:
        err = (
            'Could not find layers for selected Resnet size.\n'
            'Size received: {}; sizes allowed: {}.'.format(
                resnet_size, resnet_choices.keys()
            )
        )
        raise ValueError(err)


################################################################################
# ResNet block definitions.
################################################################################
def resnet_block(inputs, filters, is_training, projection_shortcut, strides,
                 regularizer=None, batch_norm_momentum=0.9,
                 batch_norm_epsilon=0.001):
    """A single block for ResNet v2, without a bottleneck.
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the convolutions.
      is_training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
      projection_shortcut: The function to use for projection shortcuts
        (typically a 1x1 convolution when downsampling the input).
      strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
    Returns:
      The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs
    inputs = resnet_batch_norm(inputs, is_training,
                               batch_norm_momentum=batch_norm_momentum,
                               batch_norm_epsilon=batch_norm_epsilon)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        regularizer=regularizer)

    inputs = resnet_batch_norm(inputs, is_training,
                               batch_norm_momentum=batch_norm_momentum,
                               batch_norm_epsilon=batch_norm_epsilon)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        regularizer=regularizer)

    return inputs + shortcut


def resnet_bottleneck_block(inputs, filters, is_training, projection_shortcut,
                            strides, regularizer=None, batch_norm_momentum=0.9,
                            batch_norm_epsilon=0.001):
    """A single block for ResNet v2, with a bottleneck.
    Similar to _building_block_v2(), except using the "bottleneck" blocks
    described in:
      Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
    Adapted to the ordering conventions of:
      Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the convolutions.
      is_training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
      projection_shortcut: The function to use for projection shortcuts
        (typically a 1x1 convolution when downsampling the input).
      strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
    Returns:
      The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs
    inputs = resnet_batch_norm(inputs, is_training,
                               batch_norm_momentum=batch_norm_momentum,
                               batch_norm_epsilon=batch_norm_epsilon)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1,
        regularizer=regularizer)

    inputs = resnet_batch_norm(inputs, is_training,
                               batch_norm_momentum=batch_norm_momentum,
                               batch_norm_epsilon=batch_norm_epsilon)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        regularizer=regularizer)

    inputs = resnet_batch_norm(inputs, is_training,
                               batch_norm_momentum=batch_norm_momentum,
                               batch_norm_epsilon=batch_norm_epsilon)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
        regularizer=regularizer)

    return inputs + shortcut


def resnet_block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                       is_training, name, regularizer=None,
                       batch_norm_momentum=0.9, batch_norm_epsilon=0.001):
    """Creates one layer of blocks for the ResNet model.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the first convolution of the layer.
      bottleneck: Is the block created a bottleneck block.
      block_fn: The block to use within the model, either `building_block` or
        `bottleneck_block`.
      blocks: The number of blocks contained in the layer.
      strides: The stride to use for the first convolution of the layer. If
        greater than 1, this layer will ultimately downsample the input.
      is_training: Either True or False, whether we are currently training the
        model. Needed for batch norm.
      name: A string name for the tensor output of the block layer.
    Returns:
      The output tensor of the block layer.
    """

    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            regularizer=regularizer)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, is_training, projection_shortcut,
                      strides, regularizer=regularizer,
                      batch_norm_momentum=batch_norm_momentum,
                      batch_norm_epsilon=batch_norm_epsilon)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, is_training, None, 1,
                          regularizer=regularizer,
                          batch_norm_momentum=batch_norm_momentum,
                          batch_norm_epsilon=batch_norm_epsilon)

    return tf.identity(inputs, name)


class ResNet(object):
    """Base class for building the Resnet Model."""

    def __init__(self, resnet_size, bottleneck, num_filters,
                 kernel_size, conv_stride, first_pool_size, first_pool_stride,
                 block_sizes, block_strides, batch_norm_momentum=0.9,
                 batch_norm_epsilon=0.001):
        """Creates a model obtaining an image representation.

        Implements ResNet v2:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

        Args:
          resnet_size: A single integer for the size of the ResNet model.
          bottleneck: Use regular blocks or bottleneck blocks.
          num_filters: The number of filters to use for the first block layer
            of the model. This number is then doubled for each subsequent block
            layer.
          kernel_size: The kernel size to use for convolution.
          conv_stride: stride size for the initial convolutional layer
          first_pool_size: Pool size to be used for the first pooling layer.
            If none, the first pooling layer is skipped.
          first_pool_stride: stride size for the first pooling layer. Not used
            if first_pool_size is None.
          block_sizes: A list containing n values, where n is the number of sets of
            block layers desired. Each value should be the number of blocks in the
            i-th set.
          block_strides: List of integers representing the desired stride size for
            each of the sets of block layers. Should be same length as block_sizes.
        Raises:
          ValueError: if invalid version is selected.
        """
        self.resnet_size = resnet_size

        self.bottleneck = bottleneck
        if bottleneck:
            self.block_fn = resnet_bottleneck_block
        else:
            self.block_fn = resnet_block

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.block_sizes = block_sizes
        self.block_strides = block_strides
        self.pre_activation = True
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_epsilon = batch_norm_epsilon

    def __call__(
            self,
            input_image,
            regularizer,
            dropout_rate,
            is_training=True
    ):
        """Add operations to classify a batch of input images.
        Args:
          input_image: A Tensor representing a batch of input images.
          is_training: A boolean. Set to True to add operations required only when
            training the classifier.
        Returns:
          A logits Tensor with shape [<batch_size>, <final_channels>].
        """
        inputs = input_image

        with tf.compat.v1.variable_scope('resnet'):
            inputs = conv2d_fixed_padding(
                inputs=inputs, filters=self.num_filters,
                kernel_size=self.kernel_size,
                strides=self.conv_stride, regularizer=regularizer)
            inputs = tf.identity(inputs, 'initial_conv')

            if self.first_pool_size:
                inputs = tf.layers.max_pooling2d(
                    inputs=inputs, pool_size=self.first_pool_size,
                    strides=self.first_pool_stride, padding='SAME')
                inputs = tf.identity(inputs, 'initial_max_pool')

            for i, num_blocks in enumerate(self.block_sizes):
                num_filters = self.num_filters * (2 ** i)
                inputs = resnet_block_layer(
                    inputs=inputs, filters=num_filters,
                    bottleneck=self.bottleneck,
                    block_fn=self.block_fn, blocks=num_blocks,
                    strides=self.block_strides[i], is_training=is_training,
                    name='block_layer{}'.format(i + 1),
                    regularizer=regularizer,
                    batch_norm_momentum=self.batch_norm_momentum,
                    batch_norm_epsilon=self.batch_norm_epsilon
                )

            # Only apply the BN and ReLU for model that does pre_activation in each
            # building/bottleneck block, eg resnet V2.
            if self.pre_activation:
                inputs = resnet_batch_norm(
                    inputs, is_training,
                    batch_norm_momentum=self.batch_norm_momentum,
                    batch_norm_epsilon=self.batch_norm_epsilon)
                inputs = tf.nn.relu(inputs)

            # The current top layer has shape
            # `batch_size x pool_size x pool_size x final_size`.
            # ResNet does an Average Pooling layer over pool_size,
            # but that is the same as doing a reduce_mean. We do a reduce_mean
            # here because it performs better than AveragePooling2D.
            axes = [1, 2]
            inputs = tf.reduce_mean(inputs, axes, keepdims=True)
            inputs = tf.identity(inputs, 'final_reduce_mean')

            inputs = tf.squeeze(inputs, axes)
            return inputs
