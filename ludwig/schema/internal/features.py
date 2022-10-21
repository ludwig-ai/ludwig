from dataclasses import dataclass

from ludwig.constants import (
    ROC_AUC,
    ACCURACY,
    MEAN_SQUARED_ERROR,
    LOSS,
    JACCARD,
)
from ludwig.schema.features.base import BaseFeatureConfig


@dataclass
class InternalInputFeatureConfig(BaseFeatureConfig):
    """Base class for feature metadata."""

    column: str = None

    proc_column: str = None


@dataclass
class InternalOutputFeatureConfig(BaseFeatureConfig):
    """Base class for feature metadata."""

    column: str = None

    proc_column: str = None

    default_validation_metric: str = None

    input_size: int = None

    num_classes: int = None


@dataclass
class BinaryOutputFeatureConfig(InternalOutputFeatureConfig):

    default_validation_metric: str = ROC_AUC


@dataclass
class CategoryOutputFeatureConfig(InternalOutputFeatureConfig):

    default_validation_metric: str = ACCURACY


@dataclass
class NumberOutputFeatureConfig(InternalOutputFeatureConfig):

    default_validation_metric: str = MEAN_SQUARED_ERROR


@dataclass
class SequenceOutputFeatureConfig(InternalOutputFeatureConfig):

    default_validation_metric: str = LOSS


@dataclass
class SetOutputFeatureConfig(InternalOutputFeatureConfig):

    default_validation_metric: str = JACCARD


@dataclass
class TextOutputFeatureConfig(InternalOutputFeatureConfig):

    default_validation_metric: str = LOSS


@dataclass
class VectorOutputFeatureConfig(InternalOutputFeatureConfig):

    default_validation_metric: str = MEAN_SQUARED_ERROR
