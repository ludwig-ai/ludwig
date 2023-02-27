from typing import List

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import CATEGORY, MODEL_ECD, MODEL_GBM
from ludwig.schema import common_fields
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config
from ludwig.schema.metadata import ENCODER_METADATA
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@register_encoder_config("passthrough", CATEGORY, model_types=[MODEL_ECD, MODEL_GBM])
@ludwig_dataclass
class CategoricalPassthroughEncoderConfig(BaseEncoderConfig):
    """CategoricalPassthroughEncoderConfig is a dataclass that configures the parameters used for a categorical
    passthrough encoder."""

    @staticmethod
    def module_name():
        return "CategoricalPassthroughEncoder"

    type: str = schema_utils.ProtectedString(
        "passthrough",
        description=ENCODER_METADATA["PassthroughEncoder"]["type"].long_description,
    )


@DeveloperAPI
@register_encoder_config("dense", CATEGORY)
@ludwig_dataclass
class CategoricalEmbedConfig(BaseEncoderConfig):
    @staticmethod
    def module_name():
        return "CategoricalEmbed"

    type: str = schema_utils.ProtectedString(
        "dense",
        description=ENCODER_METADATA["CategoricalEmbed"]["type"].long_description,
    )

    dropout: float = common_fields.DropoutField()

    vocab: List[str] = common_fields.VocabField()

    embedding_initializer: str = common_fields.EmbeddingInitializerField()

    embedding_size: int = common_fields.EmbeddingSizeField(
        default=50,
        description=(
            "The maximum embedding size, the actual size will be min(vocabulary_size, embedding_size) for "
            "dense representations and exactly vocabulary_size for the sparse encoding, where vocabulary_size "
            "is the number of different strings appearing in the training set in the column the feature is "
            "named after (plus 1 for <UNK>)."
        ),
    )

    embeddings_on_cpu: bool = common_fields.EmbeddingsOnCPUField()

    embeddings_trainable: bool = common_fields.EmbeddingsTrainableField()

    pretrained_embeddings: str = common_fields.PretrainedEmbeddingsField()


@DeveloperAPI
@register_encoder_config("sparse", CATEGORY)
@ludwig_dataclass
class CategoricalSparseConfig(BaseEncoderConfig):
    @staticmethod
    def module_name():
        return "CategorySparse"

    type: str = schema_utils.ProtectedString(
        "sparse",
        description=ENCODER_METADATA["CategoricalSparse"]["type"].long_description,
    )

    dropout: float = common_fields.DropoutField()

    vocab: List[str] = common_fields.VocabField()

    embedding_initializer: str = common_fields.EmbeddingInitializerField()

    embeddings_on_cpu: bool = common_fields.EmbeddingsOnCPUField()

    # TODO(travis): seems like this is not really a valid user option. We should probably just remove these
    # params entirely and update the encoder implementation.
    embeddings_trainable: bool = common_fields.EmbeddingsTrainableField(default=False)

    pretrained_embeddings: str = common_fields.PretrainedEmbeddingsField()
