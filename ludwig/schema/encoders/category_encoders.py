from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import CATEGORY, MODEL_ECD
from ludwig.schema import common_fields
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config
from ludwig.schema.metadata import ENCODER_METADATA


@DeveloperAPI
@register_encoder_config("passthrough", CATEGORY, model_types=[MODEL_ECD])
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
class CategoricalEmbedConfig(BaseEncoderConfig):
    @staticmethod
    def module_name():
        return "CategoricalEmbed"

    type: str = schema_utils.ProtectedString(
        "dense",
        description=ENCODER_METADATA["CategoricalEmbed"]["type"].long_description,
    )

    dropout: float = common_fields.DropoutField()

    vocab: list[str] = common_fields.VocabField()

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
class CategoricalSparseConfig(BaseEncoderConfig):
    @staticmethod
    def module_name():
        return "CategorySparse"

    type: str = schema_utils.ProtectedString(
        "sparse",
        description=ENCODER_METADATA["CategoricalSparse"]["type"].long_description,
    )

    dropout: float = common_fields.DropoutField()

    vocab: list[str] = common_fields.VocabField()

    embedding_initializer: str = common_fields.EmbeddingInitializerField()

    embeddings_on_cpu: bool = common_fields.EmbeddingsOnCPUField()

    # TODO(travis): seems like this is not really a valid user option. We should probably just remove these
    # params entirely and update the encoder implementation.
    embeddings_trainable: bool = common_fields.EmbeddingsTrainableField(default=False)

    pretrained_embeddings: str = common_fields.PretrainedEmbeddingsField()


@DeveloperAPI
@register_encoder_config("onehot", CATEGORY, model_types=[MODEL_ECD])
class CategoricalOneHotEncoderConfig(BaseEncoderConfig):
    """CategoricalOneHotEncoderConfig is a dataclass that configures the parameters used for a categorical onehot
    encoder."""

    type: str = schema_utils.ProtectedString(
        "onehot",
        description="Type of encoder.",
    )

    vocab: list[str] = common_fields.VocabField()

    def can_cache_embeddings(self) -> bool:
        return True


@DeveloperAPI
@register_encoder_config("target", CATEGORY, model_types=[MODEL_ECD])
class CategoricalTargetEncoderConfig(BaseEncoderConfig):
    """Target encoding: encode categories by smoothed mean target value.

    Cite: Micci-Barreca, "A Preprocessing Scheme for High-Cardinality Categorical
    Attributes in Classification and Prediction Problems", ACM SIGKDD 2001.
    """

    @staticmethod
    def module_name():
        return "CategoricalTargetEncoder"

    type: str = schema_utils.ProtectedString(
        "target",
        description="Target encoding: maps each category to a learned embedding initialized from target statistics.",
    )

    vocab: list[str] = common_fields.VocabField()

    output_size: int = schema_utils.PositiveInteger(
        default=1,
        description="Size of the target encoding output per category.",
    )


@DeveloperAPI
@register_encoder_config("hash", CATEGORY, model_types=[MODEL_ECD])
class CategoricalHashEncoderConfig(BaseEncoderConfig):
    """Feature hashing encoder for ultra-high-cardinality categoricals.

    Cite: Weinberger et al., "Feature Hashing for Large Scale Multitask Learning", ICML 2009.
    """

    @staticmethod
    def module_name():
        return "CategoricalHashEncoder"

    type: str = schema_utils.ProtectedString(
        "hash",
        description="Feature hashing: maps categories to a fixed number of hash buckets with learned embeddings.",
    )

    vocab: list[str] = common_fields.VocabField()

    num_hash_buckets: int = schema_utils.PositiveInteger(
        default=1024,
        description="Number of hash buckets to map categories into.",
    )

    embedding_size: int = common_fields.EmbeddingSizeField(
        default=50,
        description="Size of the embedding for each hash bucket.",
    )
