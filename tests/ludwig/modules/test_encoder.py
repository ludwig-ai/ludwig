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
import pytest
import torch

from ludwig.data.dataset_synthesizer import build_vocab
from ludwig.encoders.base import Encoder
from ludwig.encoders.image.base import MLPMixerEncoder, Stacked2DCNN
from ludwig.encoders.sequence_encoders import (
    ParallelCNN,
    SequenceEmbedEncoder,
    StackedCNN,
    StackedCNNRNN,
    StackedParallelCNN,
    StackedRNN,
)
from ludwig.utils.torch_utils import get_torch_device
from tests.integration_tests.parameter_update_utils import check_module_parameters_updated

DROPOUT = 0.5
DEVICE = get_torch_device()
RANDOM_SEED = 1919


def create_encoder(encoder_type, **encoder_kwargs):
    encoder = encoder_type(**encoder_kwargs)
    return encoder


def _generate_image(image_size):
    return np.random.randint(0, 1, image_size).astype(np.float32)


def generate_images(image_size, num_images):
    return np.array([_generate_image(image_size) for _ in range(num_images)])


def _generate_sentence(vocab_size, max_len):
    sentence = np.zeros(max_len, dtype=np.int32)
    random_length = random.randint(1, max_len)
    sentence[:random_length] = [random.randint(0, vocab_size - 1) for _ in range(random_length)]

    return sentence


def generate_random_sentences(num_sentences=10, max_len=10, vocab_size=10):
    # Generate some random text
    vocab = build_vocab(vocab_size)

    text = np.array([_generate_sentence(vocab_size, max_len) for _ in range(num_sentences)])

    return text, vocab


def encoder_test(
    encoder,
    input_data,
    output_dtype,
    output_shape,
    output_data=None,
):
    """Helper method to test different kinds of encoders.

    :param encoder: encoder object
    :param input_data: data to encode
    :param output_dtype: expected data type of the output (optional)
    :param output_shape: expected shape of the encoder output (optional)
    :param output_data: expected output data (optional)
    :return: returns the encoder object for the caller to run extra checks
    """
    encoder = encoder.to(DEVICE)

    # Run the encoder
    input_data = torch.from_numpy(input_data).to(DEVICE)

    hidden = encoder(input_data)["encoder_output"]

    # Check output shape and type
    assert hidden.dtype == output_dtype
    assert list(hidden.shape) == output_shape

    if output_data is not None:
        # todo the hidden output is actually a tensor. May need modification
        assert np.allclose(hidden, output_data)


def test_image_encoders_stacked_2dcnn():
    # make repeatable
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Test the resnet encoder for images
    encoder_kwargs = {"num_conv_layers": 2, "num_filters": 16, "output_size": 28, "dropout": DROPOUT}
    image_size = (3, 10, 10)

    encoder = create_encoder(
        Stacked2DCNN, num_channels=image_size[0], height=image_size[1], width=image_size[2], **encoder_kwargs
    )

    assert encoder is not None
    assert encoder.conv_stack_2d is not None
    assert list(encoder.conv_stack_2d.output_shape) == [32, 1, 1]
    assert len(encoder.fc_stack.layers) == 1
    assert encoder.conv_stack_2d.layers[0]["pool_kernel_size"] == 2
    assert encoder.conv_stack_2d.layers[0]["stride"] == 1
    assert encoder.conv_stack_2d.layers[0]["pool_stride"] == 2
    assert encoder.conv_stack_2d.layers[0]["norm"] is None
    assert encoder.conv_stack_2d.layers[0]["activation"] == "relu"
    assert encoder.conv_stack_2d.layers[0]["dropout"] == 0

    output_shape = [1, 28]
    input_image = generate_images(image_size, 1)

    encoder_test(
        encoder=encoder, input_data=input_image, output_dtype=torch.float32, output_shape=output_shape, output_data=None
    )

    output_shape = [5, 28]
    input_images = generate_images(image_size, 5)

    encoder_test(
        encoder=encoder,
        input_data=input_images,
        output_dtype=torch.float32,
        output_shape=output_shape,
        output_data=None,
    )

    # test for parameter updates
    # generate tensors for parameter update test
    target = torch.rand(output_shape)
    image_tensor = torch.rand(input_image.shape)

    # check for parameter updates
    fpc, tpc, upc, not_updated = check_module_parameters_updated(encoder, (image_tensor,), target)
    assert upc == tpc, (
        f"Not all trainable parameters updated.  Parameters not updated: {not_updated}."
        f"  Module structure\n{encoder}"
    )


def test_image_encoders_mlpmixer():
    # make repeatable
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Test the resnet encoder for images
    encoder_kwargs = {
        "patch_size": 5,
        "embed_size": 8,
        "token_size": 32,
        "channel_dim": 16,
        "num_layers": 2,
        "dropout": DROPOUT,
    }
    image_size = (3, 10, 10)

    output_shape = [1, 8]
    input_image = generate_images(image_size, 1)

    encoder = create_encoder(
        MLPMixerEncoder, num_channels=image_size[0], height=image_size[1], width=image_size[2], **encoder_kwargs
    )
    encoder_test(
        encoder=encoder, input_data=input_image, output_dtype=torch.float32, output_shape=output_shape, output_data=None
    )

    output_shape = [5, 8]
    input_images = generate_images(image_size, 5)

    encoder_test(
        encoder=encoder,
        input_data=input_images,
        output_dtype=torch.float32,
        output_shape=output_shape,
        output_data=None,
    )

    assert encoder is not None
    assert encoder.mlp_mixer.__class__.__name__ == "MLPMixer"
    assert len(encoder.mlp_mixer.mixer_blocks) == 2
    assert list(encoder.mlp_mixer.mixer_blocks[0].mlp1.output_shape) == [4]
    assert encoder.mlp_mixer.patch_conv.__class__.__name__ == "Conv2d"
    assert encoder.mlp_mixer.patch_conv.kernel_size == (5, 5)

    # test for parameter updates
    # generate tensors for parameter update test
    target = torch.rand(output_shape)
    image_tensor = torch.rand(input_image.shape)

    # check for parameter updates
    fpc, tpc, upc, not_updated = check_module_parameters_updated(encoder, (image_tensor,), target)
    assert upc == tpc, (
        f"Not all trainable parameters updated.  Parameters not updated: {not_updated}."
        f"  Module structure\n{encoder}"
    )


def test_sequence_encoder_embed():
    num_sentences = 4
    embedding_size = 5
    max_len = 6

    # make repeatable
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Generate data
    text, vocab = generate_random_sentences(
        num_sentences=num_sentences,
        max_len=max_len,
    )

    encoder_kwargs = {"embedding_size": embedding_size, "vocab": vocab}

    # Different values for reduce_output and the corresponding expected size
    reduce_outputs = ["sum", None, "concat"]
    output_shapes = [
        [num_sentences, embedding_size],
        [num_sentences, max_len, embedding_size],
        [num_sentences, max_len * embedding_size],
    ]

    for reduce_output, output_shape in zip(reduce_outputs, output_shapes):
        for trainable in [True, False]:
            encoder_kwargs["reduce_output"] = reduce_output
            encoder_kwargs["embeddings_trainable"] = trainable
            encoder_kwargs["dropout"] = DROPOUT
            encoder = create_encoder(SequenceEmbedEncoder, max_sequence_length=max_len, **encoder_kwargs)

            encoder_test(
                encoder=encoder,
                input_data=text,
                output_dtype=torch.float32,
                output_shape=output_shape,
                output_data=None,
            )

            assert encoder.embed_sequence.dropout is not None

            # test for parameter updates
            # generate tensors for parameter update test
            target = torch.rand(output_shape)

            # check for parameter updates
            fpc, tpc, upc, not_updated = check_module_parameters_updated(
                encoder, (torch.tensor(text, dtype=torch.int32),), target
            )
            assert upc == tpc, (
                f"Not all trainable parameters updated.  Parameters not updated: {not_updated}."
                f"  Module structure\n{encoder}"
            )


@pytest.mark.parametrize("encoder_type", [ParallelCNN, StackedCNN, StackedParallelCNN, StackedRNN, StackedCNNRNN])
@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("reduce_output", ["sum", "max"])
def test_sequence_encoders(encoder_type: Encoder, trainable: bool, reduce_output: str):
    num_sentences = 4
    embedding_size = 5
    max_len = 7
    output_size = 3

    # make repeatable
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Generate data
    text, vocab = generate_random_sentences(
        num_sentences=num_sentences,
        max_len=max_len,
    )

    encoder_kwargs = {
        "embedding_size": embedding_size,
        "vocab": vocab,
        "output_size": output_size,
        "num_fc_layers": 1,
        "filter_size": 3,
        "num_filters": 8,
        "state_size": output_size,
    }

    # todo figure out the output size for parallel 1d conv
    output_shape = [num_sentences, output_size]

    encoder_kwargs["embeddings_trainable"] = trainable
    encoder_kwargs["dropout"] = DROPOUT
    encoder_kwargs["recurrent_dropout"] = DROPOUT
    encoder_kwargs["fc_dropout"] = DROPOUT
    encoder_kwargs["reduce_output"] = reduce_output
    encoder = create_encoder(encoder_type, max_sequence_length=max_len, **encoder_kwargs)

    encoder_test(
        encoder=encoder, input_data=text, output_dtype=torch.float32, output_shape=output_shape, output_data=None
    )

    assert isinstance(encoder, encoder_type)

    # test for parameter updates
    # generate tensors for parameter update test
    target = torch.rand(output_shape)

    # check for parameter updates
    fpc, tpc, upc, not_updated = check_module_parameters_updated(
        encoder, (torch.tensor(text, dtype=torch.int32),), target
    )

    if trainable:
        assert fpc == 0, "Embedding layer expected to be trainable but found to be frozen"
    else:
        assert fpc == 1, "Embedding layer expected to be frozen, but found to be trainable."

    # for given random seed and configuration and non-zero dropout updated parameter counts
    # could take on different values
    assert (upc == tpc) or (upc == 0), (
        f"Not all trainable parameters updated.  Parameters not updated: {not_updated}."
        f"  Module structure\n{encoder}"
    )
