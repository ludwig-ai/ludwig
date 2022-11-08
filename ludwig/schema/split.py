from dataclasses import field

from marshmallow import fields, ValidationError
from marshmallow_dataclass import dataclass

from ludwig.constants import SPLIT, TYPE
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata.preprocessing_metadata import PREPROCESSING_METADATA
from ludwig.utils.registry import Registry

split_config_registry = Registry()
DEFAULT_PROBABILITIES = [0.7, 0.1, 0.2]


def get_split_cls(name: str):
    return split_config_registry[name]


@dataclass(repr=False, order=True)
class BaseSplitConfig(schema_utils.BaseMarshmallowConfig):
    """This Dataclass is a base schema for the nested split config under preprocessing."""

    type: str
    "Name corresponding to the splitting type."


@split_config_registry.register("random")
@dataclass(repr=False, order=True)
class RandomSplitConfig(BaseSplitConfig):
    """This Dataclass generates a schema for the random splitting config."""

    type: str = schema_utils.StringOptions(
        ["random"],
        default="random",
        allow_none=False,
        description="Type of splitting to use during preprocessing.",
    )

    probabilities: list = schema_utils.List(
        list_type=float,
        default=DEFAULT_PROBABILITIES,
        description="Probabilities for splitting data into train, validation, and test sets.",
        parameter_metadata=PREPROCESSING_METADATA["split_probabilities"],
    )


@split_config_registry.register("fixed")
@dataclass(repr=False, order=True)
class FixedSplitConfig(BaseSplitConfig):
    """This Dataclass generates a schema for the fixed splitting config."""

    type: str = schema_utils.StringOptions(
        ["fixed"],
        default="fixed",
        allow_none=False,
        description="Type of splitting to use during preprocessing.",
    )

    column: str = schema_utils.String(
        default=SPLIT,
        description="The column name to use for fixed splitting.",
    )


@split_config_registry.register("stratify")
@dataclass(repr=False, order=True)
class StratifySplitConfig(BaseSplitConfig):
    """This Dataclass generates a schema for the fixed splitting config."""

    type: str = schema_utils.StringOptions(
        ["stratify"],
        default="stratify",
        allow_none=False,
        description="Type of splitting to use during preprocessing.",
    )

    column: str = schema_utils.String(
        description="The column name to base the stratified splitting on.",
    )

    probabilities: list = schema_utils.List(
        list_type=float,
        default=DEFAULT_PROBABILITIES,
        description="Probabilities for splitting data into train, validation, and test sets.",
        parameter_metadata=PREPROCESSING_METADATA["split_probabilities"],
    )


@split_config_registry.register("datetime")
@dataclass(repr=False, order=True)
class DateTimeSplitConfig(BaseSplitConfig):
    """This Dataclass generates a schema for the fixed splitting config."""

    type: str = schema_utils.StringOptions(
        ["datetime"],
        default="datetime",
        allow_none=False,
        description="Type of splitting to use during preprocessing.",
    )

    column: str = schema_utils.String(
        description="The column name to perform datetime splitting on.",
    )

    probabilities: list = schema_utils.List(
        list_type=float,
        default=DEFAULT_PROBABILITIES,
        description="Proportion of data to split into train, validation, and test sets.",
        parameter_metadata=PREPROCESSING_METADATA["split_probabilities"],
    )


@split_config_registry.register("hash")
@dataclass(order=True)
class HashSplitConfig(BaseSplitConfig):
    """This Dataclass generates a schema for the hash splitting config.

    This is useful for deterministically splitting on a unique ID. Even when additional rows are added to the dataset
    in the future, each ID will retain its original split assignment.

    This approach does not guarantee that the split proportions will be assigned exactly, but the larger the dataset,
    the more closely the assignment should match the given proportions.

    This approach can be used on a column with duplicates, but it will further skew the assignments of rows to splits.
    """

    type: str = schema_utils.StringOptions(
        ["hash"],
        default="hash",
        allow_none=False,
        description="Type of splitting to use during preprocessing.",
    )

    column: str = schema_utils.String(
        description="The column name to perform hash splitting on.",
    )

    probabilities: list = schema_utils.List(
        list_type=float,
        default=DEFAULT_PROBABILITIES,
        description="Proportion of data to split into train, validation, and test sets.",
        parameter_metadata=PREPROCESSING_METADATA["split_probabilities"],
    )


def get_split_conds():
    """Returns a JSON schema of conditionals to validate against optimizer types defined in
    `ludwig.modules.optimization_modules.optimizer_registry`."""
    conds = []
    for splitter in split_config_registry.data:
        splitter_cls = split_config_registry.data[splitter]
        other_props = schema_utils.unload_jsonschema_from_marshmallow_class(splitter_cls)["properties"]
        schema_utils.remove_duplicate_fields(other_props)
        splitter_cond = schema_utils.create_cond(
            {"type": splitter},
            other_props,
        )
        conds.append(splitter_cond)
    return conds


def SplitDataclassField(default: str):
    """Custom dataclass field that when used inside a dataclass will allow the user to specify a nested split
    config.

    Returns: Initialized dataclass field that converts an untyped dict with params to a split config.
    """

    class SplitMarshmallowField(fields.Field):
        """Custom marshmallow field that deserializes a dict for a valid split config from the split_registry and
        creates a corresponding JSON schema for external usage."""

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return None
            if isinstance(value, dict):
                if TYPE in value and value[TYPE] in split_config_registry.data:
                    split_class = split_config_registry.data[value[TYPE]]
                    try:
                        return split_class.get_schema_cls().Schema().load(value)
                    except (TypeError, ValidationError) as error:
                        raise ValidationError(
                            f"Invalid split params: {value}, see `{split_class}` definition. Error: {error}"
                        )
                raise ValidationError(
                    f"Invalid params for splitter: {value}, expected dict with at least a valid `type` attribute."
                )
            raise ValidationError("Field should be None or dict")

        @staticmethod
        def _jsonschema_type_mapping():
            return {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(split_config_registry.data.keys()), "default": default},
                },
                "title": "split_options",
                "allOf": get_split_conds(),
            }

    try:
        splitter = split_config_registry.data[default]
        load_default = splitter.Schema().load({"type": default})
        dump_default = splitter.Schema().dump({"type": default})

        return field(
            metadata={
                "marshmallow_field": SplitMarshmallowField(
                    allow_none=False,
                    dump_default=dump_default,
                    load_default=load_default,
                )
            },
            default_factory=lambda: load_default,
        )
    except Exception as e:
        raise ValidationError(f"Unsupported splitter type: {default}. See split_registry. " f"Details: {e}")
