import logging
from dataclasses import Field, field
from typing import Any, Dict, Generic, Iterable, List, Optional, Tuple, TypeVar

from marshmallow import fields, validate
from rich.console import Console

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import (
    AUDIO,
    BAG,
    BINARY,
    CATEGORY,
    DATE,
    H3,
    IMAGE,
    MODEL_ECD,
    MODEL_GBM,
    NUMBER,
    SEQUENCE,
    SET,
    TEXT,
    TIMESERIES,
    VECTOR,
)
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.utils import (
    ecd_input_config_registry,
    gbm_input_config_registry,
    get_input_feature_jsonschema,
    get_output_feature_jsonschema,
    output_config_registry,
)
from ludwig.schema.metadata.parameter_metadata import INTERNAL_ONLY, ParameterMetadata
from ludwig.schema.utils import ludwig_dataclass

logger = logging.getLogger(__name__)
_error_console = Console(stderr=True, style="bold red")
_info_console = Console(stderr=True, style="bold green")


@DeveloperAPI
@ludwig_dataclass
class BaseFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """Base class for feature configs."""

    active: bool = True

    name: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of the feature.",
    )

    type: str = schema_utils.StringOptions(
        default=None,
        allow_none=True,
        options=[AUDIO, BAG, BINARY, CATEGORY, DATE, H3, IMAGE, NUMBER, SEQUENCE, SET, TEXT, TIMESERIES, VECTOR],
        description="Type of the feature.",
    )

    column: str = schema_utils.String(
        allow_none=True,
        default=None,
        description="The column name of this feature. Defaults to name if not specified.",
    )

    proc_column: str = schema_utils.String(
        allow_none=True,
        default=None,
        description="The name of the preprocessed column name of this feature. Internal only.",
        parameter_metadata=ParameterMetadata(internal_only=True),
    )

    def enable(self):
        """This function allows the user to specify which features from a dataset should be included during model
        training. This is the equivalent to toggling on a feature in the model creation UI.

        Returns:
            None
        """
        if self.active:
            _error_console.print("This feature is already enabled!")
        else:
            self.active = True
            _info_console.print(f"{self.name} feature enabled!\n")
            logger.info(self.__repr__())

    def disable(self):
        """This function allows the user to specify which features from a dataset should not be included during
        model training. This is the equivalent to toggling off a feature in the model creation UI.

        Returns:
            None
        """
        if not self.active:
            _error_console.print("This feature is already disabled!")
        else:
            self.active = False
            _info_console.print(f"{self.name} feature disabled!\n")
            logger.info(self.__repr__())


@DeveloperAPI
@ludwig_dataclass
class BaseInputFeatureConfig(BaseFeatureConfig):
    """Base input feature config class."""

    tied: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
        "feature of the same type and with the same encoder parameters.",
    )

    def has_augmentation(self) -> bool:
        return False


@DeveloperAPI
@ludwig_dataclass
class ECDInputFeatureConfig(BaseFeatureConfig):
    pass


@DeveloperAPI
@ludwig_dataclass
class GBMInputFeatureConfig(BaseFeatureConfig):
    pass


@DeveloperAPI
@ludwig_dataclass
class BaseOutputFeatureConfig(BaseFeatureConfig):
    """Base output feature config class."""

    reduce_input: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
    )

    default_validation_metric: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="Internal only use parameter: default validation metric for output feature.",
        parameter_metadata=INTERNAL_ONLY,
    )

    dependencies: List[str] = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
    )

    reduce_dependencies: str = schema_utils.ReductionOptions(
        default="sum",
        description="How to reduce the dependencies of the output feature.",
    )

    input_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Size of the input to the decoder.",
        parameter_metadata=ParameterMetadata(internal_only=True),
    )

    num_classes: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="Size of the input to the decoder.",
        parameter_metadata=ParameterMetadata(internal_only=True),
    )


T = TypeVar("T", bound=BaseFeatureConfig)


class FeatureCollection(Generic[T], schema_utils.ListSerializable):
    def __init__(self, features: List[T]):
        self._features = features
        self._name_to_feature = {f.name: f for f in features}
        for k, v in self._name_to_feature.items():
            setattr(self, k, v)

    def to_list(self) -> List[Dict[str, Any]]:
        out_list = []
        for feature in self._features:
            out_list.append(feature.to_dict())
        return out_list

    def items(self) -> Iterable[Tuple[str, T]]:
        return self._name_to_feature.items()

    def __iter__(self):
        return iter(self._features)

    def __len__(self):
        return len(self._features)

    def __getitem__(self, i) -> T:
        if isinstance(i, str):
            return self._name_to_features[i]
        else:
            return self._features[i]


class FeatureList(fields.List):
    def _serialize(self, value, attr, obj, **kwargs) -> Optional[List[Any]]:
        if value is None:
            return None

        value_list = value.to_list()
        return super()._serialize(value_list, attr, obj, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs) -> FeatureCollection:
        feature_list = super()._deserialize(value, attr, data, **kwargs)
        return FeatureCollection(feature_list)


class FeaturesTypeSelection(schema_utils.TypeSelection):
    def __init__(self, *args, min_length: Optional[int] = 1, max_length: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_length = min_length
        self.max_length = max_length

    def get_list_field(self) -> Field:
        min_length = self.min_length
        max_length = self.max_length
        equal = None
        if min_length == max_length:
            min_length = None
            max_length = None
            equal = self.max_length

        return field(
            metadata={
                "marshmallow_field": FeatureList(
                    self,
                    validate=validate.Length(
                        min=min_length,
                        max=max_length,
                        equal=equal,
                    ),
                )
            },
        )


class ECDInputFeatureSelection(FeaturesTypeSelection):
    def __init__(self):
        super().__init__(registry=ecd_input_config_registry, description="Type of the input feature")

    @staticmethod
    def _jsonschema_type_mapping():
        return get_input_feature_jsonschema(MODEL_ECD)


class GBMInputFeatureSelection(FeaturesTypeSelection):
    def __init__(self):
        super().__init__(registry=gbm_input_config_registry, description="Type of the input feature")

    @staticmethod
    def _jsonschema_type_mapping():
        return get_input_feature_jsonschema(MODEL_GBM)


class ECDOutputFeatureSelection(FeaturesTypeSelection):
    def __init__(self):
        super().__init__(registry=output_config_registry, description="Type of the output feature")

    @staticmethod
    def _jsonschema_type_mapping():
        return get_output_feature_jsonschema(MODEL_ECD)


class GBMOutputFeatureSelection(FeaturesTypeSelection):
    def __init__(self):
        super().__init__(max_length=1, registry=output_config_registry, description="Type of the output feature")

    @staticmethod
    def _jsonschema_type_mapping():
        return get_output_feature_jsonschema(MODEL_GBM)
