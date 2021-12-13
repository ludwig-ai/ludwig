import logging
from typing import Dict

import torch

from ludwig.constants import HIDDEN, LOGITS, SEQUENCE, TEXT
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder
from ludwig.modules.attention_modules import MultiHeadSelfAttention
from ludwig.utils.torch_utils import Dense

logger = logging.getLogger(__name__)


@register_decoder("tagger", [SEQUENCE, TEXT])
class SequenceTaggerDecoder(Decoder):
    def __init__(
        self,
        vocab_size,
        max_sequence_length=100,
        use_attention=False,
        use_bias=True,
        attention_embedding_size=256,
        attention_num_heads=8,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        print(f"max_sequence_length: {max_sequence_length}")
        self.use_attention = use_attention
        if use_attention:
            logger.debug("  MultiHeadSelfAttention")
            self.self_attention = MultiHeadSelfAttention(
                hidden_size=attention_embedding_size, num_heads=attention_num_heads
            )
        # self.projection_layer = Dense(input_size=max_sequence_length, output_size=vocab_size, use_bias=use_bias)
        self.projection_layer = Dense(input_size=kwargs["embedding_size"], output_size=vocab_size, use_bias=use_bias)

    def forward(self, inputs: Dict[str, torch.Tensor], target: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Decodes the inputs into a sequence.

        Args:
            inputs: Dictionary of tensors from the outputs of the combiner and other output features.
            target: Tensor [batch_size, max_sequence_length] with predictions.

        Returns:
            Dictionary of tensors with logits [batch_size, max_sequence_length, vocab_size].
        """
        print(f"Called forward with inputs.keys(): {inputs.keys()}")
        # Shape [batch_size, seq_size, state_size]
        hidden = inputs[HIDDEN]
        if len(hidden.shape) != 3:
            raise ValueError(
                "Decoder inputs rank is {}, but should be 3 [batch x sequence x hidden] "
                "when using a tagger sequential decoder. "
                "Consider setting reduce_output to null / None if a sequential encoder / combiner is used.".format(
                    len(hidden.shape)
                )
            )

        print(f"hidden.size(), before attention: {hidden.size()}")
        if self.use_attention:
            print(f"hidden.size(), before attention: {hidden.size()}")
            hidden = self.self_attention(hidden)
            print(f"hidden.size(), before attention: {hidden.size()}")

        logits = self.projection_layer(hidden)
        print(f"logits: {logits.size()}")
        return {LOGITS: logits}

    def get_prediction_set(self):
        return {LOGITS}

    @property
    def input_shape(self):
        # Dummy implementation.
        return torch.Size([1])

    @property
    def output_shape(self):
        return torch.Size([self.max_sequence_length, self.vocab_size])
