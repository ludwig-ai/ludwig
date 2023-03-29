import logging
from typing import Dict

import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import HIDDEN, LOGITS, PREDICTIONS, PROBABILITIES, SEQUENCE, TEXT
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder
from ludwig.modules.attention_modules import MultiHeadSelfAttention
from ludwig.schema.decoders.sequence_decoders import SequenceTaggerDecoderConfig
from ludwig.utils.torch_utils import Dense

logger = logging.getLogger(__name__)


@DeveloperAPI
@register_decoder("tagger", [SEQUENCE, TEXT])
class SequenceTaggerDecoder(Decoder):
    def __init__(
        self,
        input_size: int,
        vocab_size: int,
        max_sequence_length: int,
        use_attention: bool = False,
        use_bias: bool = True,
        attention_embedding_size: int = 256,
        attention_num_heads: int = 8,
        decoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = decoder_config

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.input_size = input_size
        self.use_attention = use_attention
        if use_attention:
            logger.debug("  MultiHeadSelfAttention")
            self.self_attention = MultiHeadSelfAttention(
                input_size=input_size, hidden_size=attention_embedding_size, num_heads=attention_num_heads
            )
            # Adjust the input size to the final projection layer.
            input_size = self.self_attention.output_shape[0]
        self.projection_layer = Dense(input_size=input_size, output_size=vocab_size, use_bias=use_bias)

    def forward(self, inputs: Dict[str, torch.Tensor], target: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Decodes the inputs into a sequence.

        Args:
            inputs: Dictionary of tensors from the outputs of the combiner and other output features.
            target: Tensor [batch_size, max_sequence_length] with predictions.

        Returns:
            Dictionary of tensors with logits [batch_size, max_sequence_length, vocab_size].
        """
        hidden = inputs[HIDDEN]
        if len(hidden.size()) != 3:
            raise ValueError(
                f"Decoder inputs rank is {len(hidden.size())}, but should be 3: "
                + "[batch_size x max_sequence_length x hidden_size] in when using a tagger sequential decoder. "
                + "Consider setting reduce_output to None if a sequential encoder / combiner is used."
            )
        if list(hidden.shape[1:]) != [self.max_sequence_length, self.input_size]:
            raise ValueError(
                "Sequence tagger decoder inputs (hidden) should be [batch_size, self.max_sequence_length, "
                + f"input_size], or [batch_size, {self.max_sequence_length}, {self.input_size}]. However, the "
                + f"inputs (hidden) was instead: {list(hidden.size())}. "
                + "The encoder is not length preserving. Please check its configuration."
            )

        if self.use_attention:
            hidden = self.self_attention(hidden)

        logits = self.projection_layer(hidden)
        return {LOGITS: logits}

    def get_prediction_set(self):
        return {LOGITS, PROBABILITIES, PREDICTIONS}

    @staticmethod
    def get_schema_cls():
        return SequenceTaggerDecoderConfig

    @property
    def input_shape(self):
        # Dummy implementation.
        return torch.Size([1])

    @property
    def output_shape(self):
        return torch.Size([self.max_sequence_length, self.vocab_size])
