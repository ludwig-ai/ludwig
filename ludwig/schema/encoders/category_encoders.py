from typing import List

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig


@dataclass
class CategoricalEmbedConfig(BaseEncoderConfig):

    type: str = "dense"

    vocab: List[str] = schema_utils.List(
        default=None,
        description="Vocabulary of the encoder",
    )

    embedding_size: int = schema_utils.NonNegativeInteger(
        default=50,
        description="The maximum embedding size, the actual size will be min(vocabulary_size, embedding_size) for "
        "dense representations and exactly vocabulary_size for the sparse encoding, where vocabulary_size "
        "is the number of different strings appearing in the training set in the column the feature is "
        "named after (plus 1 for <UNK>).",
    )

    embeddings_trainable: bool = schema_utils.Boolean(
        default=True,
        description="If true embeddings are trained during the training process, if false embeddings are fixed. It "
        "may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter "
        "has effect only when representation is dense as sparse one-hot encodings are not trainable. ",
    )

    pretrained_embeddings: str = schema_utils.String(
        default=None,
        description="By default dense embeddings are initialized randomly, but this parameter allows to specify a "
        "path to a file containing embeddings in the GloVe format. When the file containing the "
        "embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, "
        "the others are discarded. If the vocabulary contains strings that have no match in the "
        "embeddings file, their embeddings are initialized with the average of all other embedding plus "
        "some random noise to make them different from each other. This parameter has effect only if "
        "representation is dense.",
    )

    embeddings_on_cpu: bool = schema_utils.Boolean(
        default=False,
        description="By default embedding matrices are stored on GPU memory if a GPU is used, as it allows for faster "
        "access, but in some cases the embedding matrix may be too large. This parameter forces the "
        "placement of the embedding matrix in regular memory and the CPU is used for embedding lookup, "
        "slightly slowing down the process as a result of data transfer between CPU and GPU memory.",
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate.",
    )

    embedding_initializer: str = schema_utils.StringOptions(
        [
            "constant",
            "identity",
            "zeros",
            "ones",
            "orthogonal",
            "normal",
            "uniform",
            "truncated_normal",
            "variance_scaling",
            "glorot_normal",
            "glorot_uniform",
            "xavier_normal",
            "xavier_uniform",
            "he_normal",
            "he_uniform",
            "lecun_normal",
            "lecun_uniform",
        ],
        default=None,
        description="Initializer for the embedding matrix.",
    )


@dataclass
class CategoricalSparseConfig(BaseEncoderConfig):

    type: str = "sparse"

    vocab: List[str] = schema_utils.List(
        default=None,
        description="Vocabulary of the encoder",
    )

    embeddings_trainable: bool = schema_utils.Boolean(
        default=False,
        description="If true embeddings are trained during the training process, if false embeddings are fixed. It "
        "may be useful when loading pretrained embeddings for avoiding finetuning them. This parameter "
        "has effect only when representation is dense as sparse one-hot encodings are not trainable. ",
    )

    pretrained_embeddings: str = schema_utils.String(
        default=None,
        description="By default dense embeddings are initialized randomly, but this parameter allows to specify a "
        "path to a file containing embeddings in the GloVe format. When the file containing the "
        "embeddings is loaded, only the embeddings with labels present in the vocabulary are kept, "
        "the others are discarded. If the vocabulary contains strings that have no match in the "
        "embeddings file, their embeddings are initialized with the average of all other embedding plus "
        "some random noise to make them different from each other. This parameter has effect only if "
        "representation is dense.",
    )

    embeddings_on_cpu: bool = schema_utils.Boolean(
        default=False,
        description="By default embedding matrices are stored on GPU memory if a GPU is used, as it allows for faster "
        "access, but in some cases the embedding matrix may be too large. This parameter forces the "
        "placement of the embedding matrix in regular memory and the CPU is used for embedding lookup, "
        "slightly slowing down the process as a result of data transfer between CPU and GPU memory.",
    )

    dropout: float = schema_utils.FloatRange(
        default=0.0,
        min=0,
        max=1,
        description="Dropout rate.",
    )

    embedding_initializer: str = schema_utils.StringOptions(
        [
            "constant",
            "identity",
            "zeros",
            "ones",
            "orthogonal",
            "normal",
            "uniform",
            "truncated_normal",
            "variance_scaling",
            "glorot_normal",
            "glorot_uniform",
            "xavier_normal",
            "xavier_uniform",
            "he_normal",
            "he_uniform",
            "lecun_normal",
            "lecun_uniform",
        ],
        default=None,
        description="Initializer for the embedding matrix.",
    )
