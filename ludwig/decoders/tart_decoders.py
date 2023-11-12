import logging
from typing import Callable, List

import torch
from sklearn.decomposition import PCA
from transformers import GPT2Config, GPT2Model

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BINARY
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder
from ludwig.schema.decoders.base import TARTDecoderConfig
from ludwig.utils.registry import Registry
from ludwig.utils.torch_utils import Dense

logger = logging.getLogger(__name__)


_embedding_protocol_registry = Registry()


@DeveloperAPI
def get_embedding_protocol(name: str) -> Callable:
    """Get a registered embedding protocol by name.
    Args:
        name: The name of the embedding protocol

    Returns:
        The embedding protocol function registered to `name`.
    """
    try:
        protocol = _embedding_protocol_registry[name]
    except KeyError as e:
        raise ValueError(
            f"The TART embedding protocol {name} does not exist. Please update your configuration to use one of the "
            f"following: {', '.join(_embedding_protocol_registry.keys())}."
        ) from e

    return protocol


@DeveloperAPI
def register_embedding_protocol(name: str) -> Callable:
    """Register an embedding protocol function by name.

    Args:
        name: The name to register the protocol under.

    Returns:
        An inner function to use as a decorator.
    """

    def wrap(func):
        """Register an embedding protocol function by name.

        Args:
            func: The function to register

        Returns:
            `func` unaltered.
        """
        _embedding_protocol_registry[name] = func
        return func

    return wrap


@DeveloperAPI
@register_decoder("tart", [BINARY])
class BinaryTARTDecoder(Decoder):
    """"""

    def __init__(
        self,
        input_size: int,
        use_bias: bool = True,
        weights_initializer: str = "xavier_uniform",
        bias_initializer: str = "zeros",
        decoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.decoder_config = decoder_config

        self.pca_is_fit = False

        # The embedding protocol determines how the inputs are averaged for processing by the reasoning module.
        self.embedding_protocol = get_embedding_protocol(self.decoder_config.embedding_protocol)

        # Combiner/LLM output is potentially very large, so it is reduced with PCA.
        self.pca = Dense(
            input_size,
            self.decoder_config.num_pca_components,
            use_bias=False,
            weights_initializer=weights_initializer,
            bias_initializer=bias_initializer,
        )

        # Transform the reduced input to work with the reasoning module.
        self.dense1 = Dense(
            self.decoder_config.num_pca_components,
            self.decoder_config.embedding_size,
            use_bias=use_bias,
            weights_initializer=weights_initializer,
            bias_initializer=bias_initializer,
        )

        # Set up the encoder/backbone of the reasoning head. We use
        self._backbone_config = GPT2Config(
            n_positions=2 * self.decoder_config.max_sequence_length,
            n_embd=self.decoder_config.embedding_size,
            n_layer=self.decoder_config.num_layers,
            n_head=self.decoder_config.num_heads,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )

        self.reasoning_module = GPT2Model(self._backbone_config)

        # Transform the embeddings to the output feature shape.
        self.dense2 = Dense(
            self.decoder_config.embedding_size,
            1,
            use_bias=use_bias,
            weights_initializer=weights_initializer,
            bias_initializer=bias_initializer,
        )

    def fit_pca(self, inputs: List[torch.Tensor]):
        """Fit a PCA model to vanilla or LOO embedded inputs.

        Args:
            inputs: Base model output embedded with one of the embedding protocols.
        """
        pca = PCA(n_components=self.decoder_config.num_pca_components, whiten=True)
        pca.fit(inputs)

        state_dict = {"weight": torch.from_numpy(pca.components_)}
        self.pca.load_state_dict(state_dict)

        self.pca_is_fit = True

    @staticmethod
    def get_schema_cls():
        return TARTDecoderConfig

    @property
    def input_shape(self):
        return self.pca.input_shape

    def forward(self, inputs, mask=None):
        if not self.pca_is_fit:
            raise RuntimeError(
                "Attempting to use a TART decoder without first fitting it to the data. Please run `ludwig train` "
                "with this config before predicting."
            )
        x = self.pca(inputs)
        x = self.dense1(x)
        x = self.reasoning_module(x)
        y = self.dense2(x)
        return y


@register_embedding_protocol("vanilla")
def vanilla_embeddings(inputs: List[torch.Tensor]) -> torch.Tensor:
    pass


@register_embedding_protocol("loo")
def loo_embeddings(inputs: List[torch.Tensor]) -> torch.Tensor:
    pass
