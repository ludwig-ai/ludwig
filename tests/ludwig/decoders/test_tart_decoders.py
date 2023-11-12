import pytest
from transformers import GPT2Config, GPT2Model

from ludwig.decoders.tart_decoders import BinaryTARTDecoder, loo_embeddings  # , vanilla_embeddings
from ludwig.schema.decoders.base import TARTDecoderConfig
from ludwig.utils.torch_utils import Dense


@pytest.fixture(scope="module")
def default_tart_decoder_schema():
    return TARTDecoderConfig()


class TestBinaryTARTDecoder:
    def test__init__(self, default_tart_decoder_schema: TARTDecoderConfig):
        decoder = BinaryTARTDecoder(
            default_tart_decoder_schema.max_sequence_length,
            use_bias=True,
            weights_initializer="xavier_uniform",
            bias_initializer="zeros",
            decoder_config=default_tart_decoder_schema,
        )

        assert decoder.decoder_config is default_tart_decoder_schema
        assert not decoder.pca_is_fit
        assert decoder.embedding_protocol is loo_embeddings

        assert isinstance(decoder.pca, Dense)
        # assert decoder.pca.size() == (
        #     decoder.decoder_config.max_sequence_length,
        #     decoder.decoder_config.num_pca_components,
        # )

        assert isinstance(decoder.dense1, Dense)
        # assert decoder.dense1.size() == (
        #     decoder.decoder_config.num_pca_components,
        #     decoder.decoder_config.embedding_size,
        # )

        assert isinstance(decoder._backbone_config, GPT2Config)
        assert isinstance(decoder.reasoning_module, GPT2Model)

        assert isinstance(decoder.dense2, Dense)
        # assert decoder.dense2.size() == (decoder.decoder_config.embedding_size, 1)

    def test_fit_pca(self, default_tart_decoder_schema: TARTDecoderConfig):
        pass

    def test_get_schema_cls(self):
        assert BinaryTARTDecoder.get_schema_cls() is TARTDecoderConfig

    def test_input_shape(self, default_tart_decoder_schema: TARTDecoderConfig):
        pass

    def test_forward(self, default_tart_decoder_schema: TARTDecoderConfig):
        pass
