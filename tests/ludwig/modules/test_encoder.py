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
import random

import numpy as np
import tensorflow as tf

from ludwig.data.dataset_synthesizer import build_vocab
from ludwig.encoders.image_encoders import ResNetEncoder, Stacked2DCNN
from ludwig.encoders.sequence_encoders import ParallelCNN
from ludwig.encoders.sequence_encoders import SequenceEmbedEncoder
from ludwig.encoders.sequence_encoders import StackedCNN
from ludwig.encoders.sequence_encoders import StackedCNNRNN
from ludwig.encoders.sequence_encoders import StackedParallelCNN
from ludwig.encoders.sequence_encoders import StackedRNN

L1_REGULARIZER = 'l1'
L2_REGULARIZER = 'l2'
NO_REGULARIZER = None
DROPOUT = 0.5


def create_encoder(encoder_type, encoder_args={}):
    encoder = encoder_type(**encoder_args)
    return encoder


def _generate_image(image_size):
    return np.random.randint(0, 1, image_size).astype(np.float32)


def generate_images(image_size, num_images):
    return np.array([_generate_image(image_size) for _ in range(num_images)])


def _generate_sentence(vocab_size, max_len):
    sentence = np.zeros(max_len, dtype=np.int32)
    random_length = random.randint(1, max_len)
    sentence[:random_length] = [random.randint(0, vocab_size - 1) for _ in
                                range(random_length)]

    return sentence


def generate_random_sentences(num_sentences=10, max_len=10, vocab_size=10):
    # Generate some random text
    vocab = build_vocab(vocab_size)

    text = np.array([_generate_sentence(vocab_size, max_len)
                     for _ in range(num_sentences)])

    return text, vocab


def encoder_test(
        encoder,
        input_data,
        output_dtype,
        output_shape,
        output_data=None,
):
    """
    Helper method to test different kinds of encoders
    :param encoder: encoder object
    :param input_data: data to encode
    :param output_dtype: expected data type of the output (optional)
    :param output_shape: expected shape of the encoder output (optional)
    :param output_data: expected output data (optional)
    :return: returns the encoder object for the caller to run extra checks
    """
    # Run the encoder
    input_data = tf.convert_to_tensor(input_data)

    hidden = encoder(
        input_data,
        training=False
    )['encoder_output']

    # Check output shape and type
    assert hidden.dtype == output_dtype
    assert hidden.shape.as_list() == output_shape

    if output_data is not None:
        # todo the hidden output is actually a tensor. May need modification
        assert np.allclose(hidden, output_data)


def test_image_encoders_resnet():
    # Test the resnet encoder for images
    encoder_args = {
        'resnet_size': 8, 'num_filters': 8, 'fc_size': 28,
        'weights_regularizer': L1_REGULARIZER,
        'bias_regularizer': L1_REGULARIZER,
        'activity_regularizer': L1_REGULARIZER,
        'dropout': DROPOUT
    }
    image_size = (10, 10, 3)

    output_shape = [1, 28]
    input_image = generate_images(image_size, 1)

    encoder = create_encoder(ResNetEncoder, encoder_args)
    encoder_test(
        encoder=encoder,
        input_data=input_image,
        output_dtype=np.float,
        output_shape=output_shape,
        output_data=None
    )

    output_shape = [5, 28]
    input_images = generate_images(image_size, 5)

    encoder_test(
        encoder=encoder,
        input_data=input_images,
        output_dtype=np.float,
        output_shape=output_shape,
        output_data=None
    )

    assert encoder is not None
    assert encoder.resnet.__class__.__name__ == 'ResNet2'
    assert encoder.resnet.num_filters == 8
    assert encoder.resnet.resnet_size == 8
    assert encoder.resnet.filter_size == 3
    assert encoder.flatten.__class__.__name__ == 'Flatten'
    assert encoder.fc_stack.__class__.__name__ == 'FCStack'
    assert len(encoder.fc_stack.layers) == 1
    assert encoder.fc_stack.layers[0]['fc_size'] == 28
    assert encoder.fc_stack.layers[0]['activation'] == 'relu'


def test_image_encoders_stacked_2dcnn():
    # Test the resnet encoder for images
    encoder_args = {
        'num_conv_layers': 2, 'num_filters': 16, 'fc_size': 28,
        'conv_activity_regularizer': L1_REGULARIZER,
        'conv_weights_regularizer': L1_REGULARIZER,
        'conv_bias_regularizer': L1_REGULARIZER,
        'fc_activity_regularizer': L1_REGULARIZER,
        'fc_weights_regularizer': L1_REGULARIZER,
        'fc_bias_regularizer': L1_REGULARIZER,
        'dropout': DROPOUT

    }
    image_size = (10, 10, 3)

    encoder = create_encoder(Stacked2DCNN, encoder_args)

    assert encoder is not None
    assert encoder.conv_stack_2d is not None
    assert encoder.conv_stack_2d.layers[0]['filter_size'] == 3
    assert encoder.fc_stack.layers[0]['fc_size'] == 28
    assert len(encoder.fc_stack.layers) == 1
    assert encoder.conv_stack_2d.layers[0]['num_filters'] == 16
    assert encoder.conv_stack_2d.layers[0]['pool_size'] == (2, 2)
    assert encoder.conv_stack_2d.layers[0]['strides'] == (1, 1)
    assert encoder.conv_stack_2d.layers[0]['pool_strides'] is None
    assert encoder.conv_stack_2d.layers[0]['norm'] is None
    assert encoder.fc_stack.layers[0]['activation'] == 'relu'
    assert encoder.conv_stack_2d.layers[-1]['dropout'] == 0

    output_shape = [1, 28]
    input_image = generate_images(image_size, 1)

    encoder_test(
        encoder=encoder,
        input_data=input_image,
        output_dtype=np.float,
        output_shape=output_shape,
        output_data=None
    )

    output_shape = [5, 28]
    input_images = generate_images(image_size, 5)

    encoder_test(
        encoder=encoder,
        input_data=input_images,
        output_dtype=np.float,
        output_shape=output_shape,
        output_data=None
    )


def test_sequence_encoder_embed():
    num_sentences = 4
    embedding_size = 5
    max_len = 6

    # Generate data
    text, vocab = generate_random_sentences(
        num_sentences=num_sentences,
        max_len=max_len,
    )

    encoder_args = {'embedding_size': embedding_size, 'vocab': vocab}

    # Different values for reduce_output and the corresponding expected size
    reduce_outputs = ['sum', None, 'concat']
    output_shapes = [
        [num_sentences, embedding_size],
        [num_sentences, max_len, embedding_size],
        [num_sentences, max_len * embedding_size]
    ]

    for reduce_output, output_shape in zip(reduce_outputs, output_shapes):
        for trainable in [True, False]:
            encoder_args['reduce_output'] = reduce_output
            encoder_args['embeddings_trainable'] = trainable
            encoder_args['weights_regularizer'] = L1_REGULARIZER
            encoder_args['dropout'] = DROPOUT
            encoder = create_encoder(SequenceEmbedEncoder, encoder_args)

            encoder_test(
                encoder=encoder,
                input_data=text,
                output_dtype=np.float,
                output_shape=output_shape,
                output_data=None
            )

            embed = encoder.embed_sequence.embeddings
            assert embed.trainable == trainable
            assert encoder.embed_sequence.dropout is not None


def test_sequence_encoders():
    num_sentences = 4
    embedding_size = 5
    max_len = 7
    fc_size = 3

    # Generate data
    text, vocab = generate_random_sentences(
        num_sentences=num_sentences,
        max_len=max_len,
    )

    encoder_args = {
        'embedding_size': embedding_size,
        'vocab': vocab,
        'fc_size': fc_size,
        'num_fc_layers': 1,
        'filter_size': 3,
        'num_filters': 8,
        'state_size': fc_size
    }

    # Different values for reduce_output and the corresponding expected size
    # todo figure out the output size for parallel 1d conv
    reduce_outputs = ['sum', 'max']
    output_shapes = [
        [num_sentences, fc_size],
        [num_sentences, fc_size],
        [num_sentences, max_len, fc_size]
    ]

    for reduce_output, output_shape in zip(reduce_outputs, output_shapes):
        for trainable in [True, False]:
            for encoder_type in [ParallelCNN,
                                 StackedCNN,
                                 StackedParallelCNN,
                                 StackedRNN,
                                 StackedCNNRNN]:
                encoder_args['reduce_output'] = reduce_output
                encoder_args['embeddings_trainable'] = trainable
                encoder_args['weights_regularizer'] = L1_REGULARIZER
                encoder_args['bias_regularizer'] = L1_REGULARIZER
                encoder_args['activity_regularizer'] = L1_REGULARIZER
                encoder_args['dropout'] = DROPOUT
                encoder_args['dropout'] = DROPOUT
                encoder_args['recurrent_dropout'] = DROPOUT
                encoder_args['fc_dropout'] = DROPOUT
                encoder = create_encoder(encoder_type, encoder_args)

                encoder_test(
                    encoder=encoder,
                    input_data=text,
                    output_dtype=np.float,
                    output_shape=output_shape,
                    output_data=None
                )

                assert isinstance(encoder, encoder_type)
