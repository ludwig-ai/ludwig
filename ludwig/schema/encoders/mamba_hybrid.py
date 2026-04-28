"""Schemas for Mamba-2 and Jamba encoders (Phase 6.6.2)."""

from __future__ import annotations

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import AUDIO, SEQUENCE, TEXT, TIMESERIES
from ludwig.schema import common_fields
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.sequence_encoders import SequenceEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config


@DeveloperAPI
@register_encoder_config("mamba2", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
class Mamba2EncoderConfig(SequenceEncoderConfig):
    @staticmethod
    def module_name():
        return "Mamba2Encoder"

    type: str = schema_utils.ProtectedString(
        "mamba2",
        description=(
            "Mamba-2 (Dao & Gu, 2024) multi-head selective state space encoder. Linear-time "
            "in sequence length with per-head scalar decay; pure-PyTorch approximation of the "
            "SSD formulation — no mamba_ssm CUDA kernel required."
        ),
    )

    dropout: float = common_fields.DropoutField(default=0.1, description="Dropout rate.")
    max_sequence_length: int = common_fields.MaxSequenceLengthField()
    representation: str = common_fields.RepresentationField()
    vocab: list = common_fields.VocabField()
    embedding_size: int = common_fields.EmbeddingSizeField()
    embeddings_on_cpu: bool = common_fields.EmbeddingsOnCPUField()
    embeddings_trainable: bool = common_fields.EmbeddingsTrainableField()
    pretrained_embeddings: str = common_fields.PretrainedEmbeddingsField()
    reduce_output: str = common_fields.ReduceOutputField(default="mean")
    should_embed: bool = schema_utils.Boolean(
        default=True,
        description="If True the input sequence is expected to be made of integers and will be mapped into embeddings.",
    )

    d_model: int = schema_utils.PositiveInteger(
        default=256,
        description="Hidden width of each Mamba-2 block.",
    )
    n_layers: int = schema_utils.PositiveInteger(
        default=4,
        description="Number of stacked Mamba-2 blocks.",
    )
    num_heads: int = schema_utils.PositiveInteger(
        default=8,
        description="Number of SSD heads. `d_model * expand_factor` must be divisible by `num_heads`.",
    )
    d_conv: int = schema_utils.PositiveInteger(
        default=4,
        description="Width of the depthwise 1D convolution inside each block.",
    )
    expand_factor: int = schema_utils.PositiveInteger(
        default=2,
        description="Inner expansion factor for each block.",
    )
    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Output feature width emitted by the encoder.",
    )


@DeveloperAPI
@register_encoder_config("jamba", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
class JambaEncoderConfig(SequenceEncoderConfig):
    @staticmethod
    def module_name():
        return "JambaEncoder"

    type: str = schema_utils.ProtectedString(
        "jamba",
        description=(
            "Jamba-style hybrid encoder (Lieber et al., 2024) interleaving Mamba-2 SSM blocks "
            "with TransformerEncoderLayer attention blocks. Every `attention_every_k`-th layer "
            "is attention; the rest are SSM."
        ),
    )

    dropout: float = common_fields.DropoutField(default=0.1, description="Dropout rate.")
    max_sequence_length: int = common_fields.MaxSequenceLengthField()
    representation: str = common_fields.RepresentationField()
    vocab: list = common_fields.VocabField()
    embedding_size: int = common_fields.EmbeddingSizeField()
    embeddings_on_cpu: bool = common_fields.EmbeddingsOnCPUField()
    embeddings_trainable: bool = common_fields.EmbeddingsTrainableField()
    pretrained_embeddings: str = common_fields.PretrainedEmbeddingsField()
    reduce_output: str = common_fields.ReduceOutputField(default="mean")
    should_embed: bool = schema_utils.Boolean(
        default=True,
        description="If True the input sequence is expected to be made of integers and will be mapped into embeddings.",
    )

    d_model: int = schema_utils.PositiveInteger(
        default=256,
        description="Hidden width of every block — SSM and attention share the same d_model.",
    )
    n_layers: int = schema_utils.PositiveInteger(
        default=8,
        description="Total number of stacked blocks (SSM + attention combined).",
    )
    attention_every_k: int = schema_utils.PositiveInteger(
        default=4,
        description=(
            "Every `attention_every_k`-th block is attention, the remainder are SSM. "
            "Default 4 gives a 1:3 attention:SSM ratio matching the Jamba paper."
        ),
    )
    num_heads: int = schema_utils.PositiveInteger(
        default=8,
        description="Number of attention heads (and SSD heads, shared).",
    )
    ffn_size: int = schema_utils.PositiveInteger(
        default=1024,
        description="Feed-forward width inside each attention block.",
    )
    d_conv: int = schema_utils.PositiveInteger(
        default=4,
        description="Width of the depthwise 1D convolution inside each SSM block.",
    )
    expand_factor: int = schema_utils.PositiveInteger(
        default=2,
        description="Inner expansion factor inside each SSM block.",
    )
    output_size: int = schema_utils.PositiveInteger(
        default=256,
        description="Output feature width emitted by the encoder.",
    )
