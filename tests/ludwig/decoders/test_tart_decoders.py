import copy

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2Model

from ludwig.decoders.tart_decoders import BinaryTARTDecoder, get_embedding_protocol
from ludwig.schema.decoders.base import TARTDecoderConfig
from ludwig.utils.torch_utils import Dense


@pytest.fixture(scope="module")
def default_tart_decoder_schema():
    return TARTDecoderConfig()


class TestBinaryTARTDecoder:
    def create_sample_input(self, decoder_config):
        return np.random.rand(decoder_config.max_sequence_length, 1024).astype(np.float32)

    def create_decoder_from_config(self, decoder_config):
        return BinaryTARTDecoder(
            1024,
            use_bias=True,
            weights_initializer="xavier_uniform",
            bias_initializer="zeros",
            decoder_config=decoder_config,
        )

    def test__init__(self, default_tart_decoder_schema: TARTDecoderConfig):
        decoder = BinaryTARTDecoder(
            1024,
            use_bias=True,
            weights_initializer="xavier_uniform",
            bias_initializer="zeros",
            decoder_config=default_tart_decoder_schema,
        )

        assert decoder.decoder_config is default_tart_decoder_schema
        assert not decoder.pca_is_fit
        assert decoder.embedding_protocol is get_embedding_protocol(decoder.decoder_config.embedding_protocol)

        assert isinstance(decoder.pca, Dense)
        assert decoder.pca.dense.in_features == decoder.decoder_config.max_sequence_length
        assert decoder.pca.dense.out_features == decoder.decoder_config.num_pca_components

        assert isinstance(decoder.dense1, Dense)
        assert decoder.dense1.dense.in_features == decoder.decoder_config.num_pca_components
        assert decoder.dense1.dense.out_features == decoder.decoder_config.embedding_size

        assert isinstance(decoder._backbone_config, GPT2Config)
        assert decoder._backbone_config.model_type == "gpt2"
        assert decoder._backbone_config.n_embd == decoder.decoder_config.embedding_size
        assert decoder._backbone_config.n_head == decoder.decoder_config.num_heads
        assert decoder._backbone_config.n_layer == decoder.decoder_config.num_layers

        assert isinstance(decoder.reasoning_module, GPT2Model)

        assert isinstance(decoder.dense2, Dense)
        assert decoder.dense2.dense.in_features == decoder.decoder_config.embedding_size
        assert decoder.dense2.dense.out_features == 1

    def test_fit_pca(self, default_tart_decoder_schema: TARTDecoderConfig):
        decoder = BinaryTARTDecoder(
            default_tart_decoder_schema.max_sequence_length,
            use_bias=True,
            weights_initializer="xavier_uniform",
            bias_initializer="zeros",
            decoder_config=default_tart_decoder_schema,
        )

        input = self.create_sample_input(default_tart_decoder_schema)

        original_pca_weights = copy.deepcopy(decoder.pca.dense.weight)

        decoder.fit_pca(input)

        assert torch.all(torch.not_equal(original_pca_weights, decoder.pca.dense.weight))

    def test_get_schema_cls(self):
        assert BinaryTARTDecoder.get_schema_cls() is TARTDecoderConfig

    def test_input_shape(self, default_tart_decoder_schema: TARTDecoderConfig):
        decoder = BinaryTARTDecoder(
            1024,
            use_bias=True,
            weights_initializer="xavier_uniform",
            bias_initializer="zeros",
            decoder_config=default_tart_decoder_schema,
        )

        assert decoder.input_shape == decoder.decoder_config.max_sequence_length

    def test_forward(self, default_tart_decoder_schema: TARTDecoderConfig):
        decoder = BinaryTARTDecoder(
            1024,
            use_bias=True,
            weights_initializer="xavier_uniform",
            bias_initializer="zeros",
            decoder_config=default_tart_decoder_schema,
        )

        input = self.create_sample_input(default_tart_decoder_schema)
        labels = torch.FloatTensor(1, default_tart_decoder_schema.max_sequence_length, 1)

        # For simplicity, assume batch size 1
        with pytest.raises(RuntimeError):
            decoder(torch.unsqueeze(torch.from_numpy(input), 0), labels)

        decoder.fit_pca(input)
        predictions = decoder(torch.unsqueeze(torch.from_numpy(input), 0), labels)

        assert predictions.size() == (1, 1)
