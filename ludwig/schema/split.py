from dataclasses import Field
from typing import Type

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import SPLIT, TYPE
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata import PREPROCESSING_METADATA
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils.registry import Registry

split_config_registry = Registry()
DEFAULT_PROBABILITIES = [0.7, 0.1, 0.2]


@DeveloperAPI
def get_split_cls(name: str):
    return split_config_registry[name]


@DeveloperAPI
@ludwig_dataclass
class BaseSplitConfig(schema_utils.BaseMarshmallowConfig):
    """This Dataclass is a base schema for the nested split config under preprocessing."""

    type: str
    "Name corresponding to the splitting type."


@DeveloperAPI
@split_config_registry.register("random")
@ludwig_dataclass
class RandomSplitConfig(BaseSplitConfig):
    """This Dataclass generates a schema for the random splitting config."""

    type: str = schema_utils.ProtectedString(
        "random",
        description="Type of splitting to use during preprocessing.",
    )

    probabilities: list = schema_utils.List(
        list_type=float,
        default=DEFAULT_PROBABILITIES,
        description="Probabilities for splitting data into train, validation, and test sets.",
        parameter_metadata=PREPROCESSING_METADATA["split_probabilities"],
    )


@DeveloperAPI
@split_config_registry.register("fixed")
@ludwig_dataclass
class FixedSplitConfig(BaseSplitConfig):
    """This Dataclass generates a schema for the fixed splitting config."""

    type: str = schema_utils.ProtectedString(
        "fixed",
        description="Type of splitting to use during preprocessing.",
    )

    column: str = schema_utils.String(
        default=SPLIT,
        allow_none=False,
        description="The column name to use for fixed splitting.",
    )


@DeveloperAPI
@split_config_registry.register("stratify")
@ludwig_dataclass
class StratifySplitConfig(BaseSplitConfig):
    """This Dataclass generates a schema for the fixed splitting config."""

    type: str = schema_utils.ProtectedString(
        "stratify",
        description="Type of splitting to use during preprocessing.",
    )

    column: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="The column name to base the stratified splitting on.",
    )

    probabilities: list = schema_utils.List(
        list_type=float,
        default=DEFAULT_PROBABILITIES,
        description="Probabilities for splitting data into train, validation, and test sets.",
        parameter_metadata=PREPROCESSING_METADATA["split_probabilities"],
    )


@DeveloperAPI
@split_config_registry.register("datetime")
@ludwig_dataclass
class DateTimeSplitConfig(BaseSplitConfig):
    """This Dataclass generates a schema for the fixed splitting config."""

    type: str = schema_utils.ProtectedString(
        "datetime",
        description="Type of splitting to use during preprocessing.",
    )

    column: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="The column name to perform datetime splitting on.",
    )

    probabilities: list = schema_utils.List(
        list_type=float,
        default=DEFAULT_PROBABILITIES,
        description="Proportion of data to split into train, validation, and test sets.",
        parameter_metadata=PREPROCESSING_METADATA["split_probabilities"],
    )


@DeveloperAPI
@split_config_registry.register("hash")
@ludwig_dataclass
class HashSplitConfig(BaseSplitConfig):
    """This Dataclass generates a schema for the hash splitting config.

    This is useful for deterministically splitting on a unique ID. Even when additional rows are added to the dataset
    in the future, each ID will retain its original split assignment.

    This approach does not guarantee that the split proportions will be assigned exactly, but the larger the dataset,
    the more closely the assignment should match the given proportions.

    This approach can be used on a column with duplicates, but it will further skew the assignments of rows to splits.
    """

    type: str = schema_utils.ProtectedString(
        "hash",
        description="Type of splitting to use during preprocessing.",
    )

    column: str = schema_utils.String(
        default=None,
        allow_none=True,
        description="The column name to perform hash splitting on.",
    )

    probabilities: list = schema_utils.List(
        list_type=float,
        default=DEFAULT_PROBABILITIES,
        description="Proportion of data to split into train, validation, and test sets.",
        parameter_metadata=PREPROCESSING_METADATA["split_probabilities"],
    )


@DeveloperAPI
def get_split_conds():
    """Returns a JSON schema of conditionals to validate against optimizer types defined in
    `ludwig.modules.optimization_modules.optimizer_registry`."""
    conds = []
    for splitter in split_config_registry.data:
        splitter_cls = split_config_registry.data[splitter]
        other_props = schema_utils.unload_jsonschema_from_marshmallow_class(splitter_cls)["properties"]
        schema_utils.remove_duplicate_fields(other_props, [TYPE])
        splitter_cond = schema_utils.create_cond(
            {"type": splitter},
            other_props,
        )
        conds.append(splitter_cond)
    return conds


@DeveloperAPI
def SplitDataclassField(default: str) -> Field:
    """Custom dataclass field that when used inside a dataclass will allow the user to specify a nested split
    config.

    Returns: Initialized dataclass field that converts an untyped dict with params to a split config.
    """

    class SplitSelection(schema_utils.TypeSelection):
        def __init__(self):
            super().__init__(registry=split_config_registry.data, default_value=default)

        def get_schema_from_registry(self, key: str) -> Type[schema_utils.BaseMarshmallowConfig]:
            return split_config_registry.data[key]

        @staticmethod
        def _jsonschema_type_mapping():
            return {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Type of splitting to use during preprocessing.",
                        "enum": list(split_config_registry.data.keys()),
                        "default": default,
                    },
                },
                "title": "split_options",
                "allOf": get_split_conds(),
            }

    return SplitSelection().get_default_field()
