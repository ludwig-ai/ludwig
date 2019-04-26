# -*- coding: utf-8 -*-
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

from ludwig.models.modules.image_encoders import Stacked2DCNN, ResNetEncoder
from ludwig.models.modules.loss_modules import regularizer_registry

L1_REGULARIZER = regularizer_registry['l1'](0.1)
L2_REGULARIZER = regularizer_registry['l2'](0.1)
NO_REGULARIZER = None
DROPOUT_RATE = 0.5


def create_encoder(encoder_type, encoder_args={}):
    encoder = encoder_type(**encoder_args)
    return encoder


def _generate_image(image_size):
    return np.random.randint(0, 1, image_size).astype(np.float32)


def generate_images(image_size, num_images):
    return np.array([_generate_image(image_size) for _ in range(num_images)])


def encoder_test(
        encoder,
        input_data,
        regularizer,
        dropout_rate,
        output_dtype,
        output_shape,
        output_data=None,
):
    """
    Helper method to test different kinds of encoders
    :param encoder: encoder object
    :param input_data: data to encode
    :param regularizer: regularizer
    :param dropout_rate: dropout rate
    :param output_dtype: expected data type of the output (optional)
    :param output_shape: expected shape of the encoder output (optional)
    :param output_data: expected output data (optional)
    :return: returns the encoder object for the caller to run extra checks
    """

    # Run the encoder
    input_data = tf.convert_to_tensor(input_data)
    dropout_rate = tf.convert_to_tensor(dropout_rate)
    is_training = tf.convert_to_tensor(False)

    hidden, _ = encoder(
        input_data,
        regularizer,
        dropout_rate,
        is_training=is_training
    )

    # Check output shape and type
    assert hidden.dtype == output_dtype
    assert hidden.shape.as_list() == output_shape

    if output_data is not None:
        # TODO the hidden output is actually a tensor. May need modification
        assert np.allclose(hidden, output_data)


def test_image_encoders_resnet():
    tf.reset_default_graph()
    # Test the resnet encoder for images
    encoder_args = {'resnet_size': 8, 'num_filters': 8, 'fc_size': 28}
    image_size = (10, 10, 3)

    output_shape = [1, 28]
    input_image = generate_images(image_size, 1)

    encoder = create_encoder(ResNetEncoder, encoder_args)
    encoder_test(
        encoder=encoder,
        input_data=input_image,
        regularizer=L1_REGULARIZER,
        dropout_rate=DROPOUT_RATE,
        output_dtype=np.float,
        output_shape=output_shape,
        output_data=None
    )

    output_shape = [5, 28]
    input_images = generate_images(image_size, 5)

    tf.reset_default_graph()
    encoder_test(
        encoder=encoder,
        input_data=input_images,
        regularizer=L1_REGULARIZER,
        dropout_rate=DROPOUT_RATE,
        output_dtype=np.float,
        output_shape=output_shape,
        output_data=None
    )

    assert encoder is not None
    assert encoder.resnet is not None
    assert encoder.resnet.kernel_size == 3
    assert encoder.fc_stack.layers[0]['fc_size'] == 28
    assert len(encoder.fc_stack.layers) == 1
    assert encoder.fc_stack.layers[0]['activation'] == 'relu'
    assert encoder.resnet.num_filters == 8
    assert encoder.resnet.resnet_size == 8


def test_image_encoders_stacked_2dcnn():
    tf.reset_default_graph()
    # Test the resnet encoder for images
    encoder_args = {'num_conv_layers': 2, 'num_filters': 16, 'fc_size': 28}
    image_size = (10, 10, 3)

    encoder = create_encoder(Stacked2DCNN, encoder_args)

    assert encoder is not None
    assert encoder.conv_stack_2d is not None
    assert encoder.conv_stack_2d.layers[0]['filter_size'] == 3
    assert encoder.fc_stack.layers[0]['fc_size'] == 28
    assert len(encoder.fc_stack.layers) == 1
    assert encoder.conv_stack_2d.layers[0]['num_filters'] == 16
    assert encoder.conv_stack_2d.layers[0]['pool_size'] == 2
    assert encoder.conv_stack_2d.layers[0]['stride'] == 1
    assert encoder.conv_stack_2d.layers[0]['pool_stride'] == 2
    assert encoder.conv_stack_2d.layers[0]['norm'] is None
    assert encoder.fc_stack.layers[0]['activation'] == 'relu'
    assert encoder.conv_stack_2d.layers[-1]['dropout'] is True

    output_shape = [1, 28]
    input_image = generate_images(image_size, 1)

    encoder_test(
        encoder=encoder,
        input_data=input_image,
        regularizer=L1_REGULARIZER,
        dropout_rate=DROPOUT_RATE,
        output_dtype=np.float,
        output_shape=output_shape,
        output_data=None
    )

    output_shape = [5, 28]
    input_images = generate_images(image_size, 5)

    tf.reset_default_graph()
    encoder_test(
        encoder=encoder,
        input_data=input_images,
        regularizer=L1_REGULARIZER,
        dropout_rate=DROPOUT_RATE,
        output_dtype=np.float,
        output_shape=output_shape,
        output_data=None
    )
