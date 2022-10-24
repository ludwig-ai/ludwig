from dataclasses import dataclass, field
from typing import List, Union

from ludwig.constants import (
    ACCURACY,
    BINARY,
    CATEGORY,
    JACCARD,
    LOSS,
    MEAN_SQUARED_ERROR,
    NUMBER,
    ROC_AUC,
    SEQUENCE,
    SET,
    TEXT,
    VECTOR,
)
from ludwig.schema.config.utils import internal_output_config_registry
from ludwig.schema.features.base import BaseFeatureConfig


@dataclass
class InternalEncoderMetadata:
    """Class for internal encoder parameters."""

    vocab: List[str] = None

    vocab_size: int = None

    should_embed: bool = True


@dataclass
class InternalDecoderMetadata:
    """Class for internal decoder parameters."""

    vocab_size: int = None


@dataclass
class InternalPreprocessingMetadata:
    """Class for internal feature preprocessing parameters."""

    computed_fill_value: Union[str, bool, int, float] = None


@dataclass
class InternalOptimizerMetadata:
    """Class for internal optimizer parameters."""

    lr: float = 1e-03


@dataclass
class InternalTrainerMetadata:
    """Class for internal trainer parameters."""

    optimizer: InternalOptimizerMetadata = field(default_factory=InternalOptimizerMetadata)


@dataclass
class InternalInputFeatureMetadata(BaseFeatureConfig):
    """Base class for input feature metadata."""

    column: str = None

    proc_column: str = None

    encoder: InternalEncoderMetadata = field(default_factory=InternalEncoderMetadata)

    preprocessing: InternalPreprocessingMetadata = field(default_factory=InternalPreprocessingMetadata)


@dataclass
class InternalOutputFeatureMetadata(BaseFeatureConfig):
    """Base class for output feature metadata."""

    column: str = None

    proc_column: str = None

    default_validation_metric: str = None

    input_size: int = None

    num_classes: int = None

    decoder: InternalDecoderMetadata = field(default_factory=InternalDecoderMetadata)

    preprocessing: InternalPreprocessingMetadata = field(default_factory=InternalPreprocessingMetadata)


@internal_output_config_registry.register(BINARY)
@dataclass
class BinaryOutputFeatureMetadata(InternalOutputFeatureMetadata):
    """Base class for binary output feature metadata."""

    default_validation_metric: str = ROC_AUC


@internal_output_config_registry.register(CATEGORY)
@dataclass
class CategoryOutputFeatureMetadata(InternalOutputFeatureMetadata):
    """Base class for category output feature metadata."""

    default_validation_metric: str = ACCURACY


@internal_output_config_registry.register(NUMBER)
@dataclass
class NumberOutputFeatureMetadata(InternalOutputFeatureMetadata):
    """Base class for number output feature metadata."""

    default_validation_metric: str = MEAN_SQUARED_ERROR


@internal_output_config_registry.register(SEQUENCE)
@dataclass
class SequenceOutputFeatureMetadata(InternalOutputFeatureMetadata):
    """Base class for sequence output feature metadata."""

    default_validation_metric: str = LOSS


@internal_output_config_registry.register(SET)
@dataclass
class SetOutputFeatureMetadata(InternalOutputFeatureMetadata):
    """Base class for set output feature metadata."""

    default_validation_metric: str = JACCARD


@internal_output_config_registry.register(TEXT)
@dataclass
class TextOutputFeatureMetadata(InternalOutputFeatureMetadata):
    """Base class for text output feature metadata."""

    default_validation_metric: str = LOSS


@internal_output_config_registry.register(VECTOR)
@dataclass
class VectorOutputFeatureMetadata(InternalOutputFeatureMetadata):
    """Base class for vector output feature metadata."""

    default_validation_metric: str = MEAN_SQUARED_ERROR
