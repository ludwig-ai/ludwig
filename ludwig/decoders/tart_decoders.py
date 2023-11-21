import logging
from typing import Callable, List, Union

import numpy as np
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

    def wrap(func: Callable) -> Callable:
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

        print(f"PCA SHAPE: {self.pca}")

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

    def fit_pca(self, inputs: Union[np.ndarray, List[np.ndarray]]):
        """Fit a PCA model to vanilla or LOO embedded inputs.

        Args:
            inputs: Base model output embedded with one of the embedding protocols.
        """
        pca = PCA(n_components=self.decoder_config.num_pca_components, whiten=True)
        pca.fit(inputs)

        state_dict = {"dense.weight": torch.from_numpy(pca.components_)}
        self.pca.load_state_dict(state_dict)

        self.pca_is_fit = True

    @staticmethod
    def get_schema_cls():
        return TARTDecoderConfig

    @property
    def input_shape(self) -> torch.Size:
        return self.pca.dense.in_features

    def _combine_gen(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Interleave the inputs and labels into a single sequence.

        Args:
            x: Input examples reduced by PCA, shape (batch_size, sequence_length, num_pca_components)
            y: Label for each example, shape (batch_size, 1)

        Returns:
            The inputs and labels stacked, shape (batch_size, 2 * sequence_length, num_pca_components)
        """
        batch_size, num_examples, reduced_size = x.shape
        y_list = []

        # Assume one output per input
        y_i = y[0, ::]
        y_i_wide = torch.cat(
            (
                y_i.view(batch_size, num_examples, 1),
                torch.zeros(batch_size, num_examples, reduced_size - 1, device=y.device),
            ),
            axis=2,
        )
        y_list.append(y_i_wide)
        zs = torch.stack((x, *y_list), dim=2)
        zs = zs.view(batch_size, (2) * num_examples, reduced_size)

        return zs

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, mask=None) -> torch.Tensor:
        if not self.pca_is_fit:
            raise RuntimeError(
                "Attempting to use a TART decoder without first fitting it to the data. Please run `ludwig train` "
                "with this config before predicting."
            )

        # Reduce the size of the input representations
        x = self.pca(inputs)

        # Stack the x and y examples
        inds = torch.arange(labels.shape[-1])
        stacked = self._combine_gen(x, labels)

        # Transform the inputs to match the GPT2 dimensions
        embeds = self.dense1(stacked)

        # Compute the embeddings
        embeds = self.reasoning_module(inputs_embeds=embeds).last_hidden_state

        # Generate class predictions
        prediction = self.dense2(embeds)

        preds = []
        preds.append(prediction[:, 0::1][:, inds])
        preds = torch.cat(preds, dim=0)

        return preds


@register_embedding_protocol("vanilla")
def vanilla_embeddings(inputs: List[torch.Tensor]) -> torch.Tensor:
    pass


@register_embedding_protocol("loo")
def loo_embeddings(inputs: List[torch.Tensor]) -> torch.Tensor:
    pass
