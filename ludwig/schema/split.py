from typing import Tuple, Optional
from marshmallow import fields, ValidationError
from marshmallow_dataclass import dataclass
from dataclasses import field

from ludwig.constants import TYPE
from ludwig.schema import utils as schema_utils
from ludwig.data.split import DEFAULT_PROBABILITIES, split_registry


@dataclass
class BaseSplitConfig(schema_utils.BaseMarshmallowConfig):
    """
    This Dataclass is a base schema for the nested split config under preprocessing
    """

    type: str
    "Name corresponding to the splitting type."


@dataclass
class RandomSplitConfig(BaseSplitConfig):
    """
    This Dataclass generates a schema for the random splitting config
    """

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
    )


@dataclass
class FixedSplitConfig(BaseSplitConfig):
    """
    This Dataclass generates a schema for the fixed splitting config
    """

    type: str = schema_utils.StringOptions(
        ["fixed"],
        default="fixed",
        allow_none=False,
        description="Type of splitting to use during preprocessing.",
    )

    column: str = schema_utils.String(
        allow_none=False,
        description="The column name to use for fixed splitting.",
    )


@dataclass
class StratifySplitConfig(BaseSplitConfig):
    """
    This Dataclass generates a schema for the fixed splitting config
    """

    type: str = schema_utils.StringOptions(
        ["stratify"],
        default="stratify",
        allow_none=False,
        description="Type of splitting to use during preprocessing.",
    )

    column: str = schema_utils.String(
        allow_none=False,
        description="The column name to base the stratified splitting on.",
    )

    probabilities: list = schema_utils.List(
        list_type=float,
        default=DEFAULT_PROBABILITIES,
        description="Probabilities for splitting data into train, validation, and test sets.",
    )


@dataclass
class DateTimeSplitConfig(BaseSplitConfig):
    """
    This Dataclass generates a schema for the fixed splitting config
    """

    type: str = schema_utils.StringOptions(
        ["datetime"],
        default="datetime",
        allow_none=False,
        description="Type of splitting to use during preprocessing.",
    )

    column: str = schema_utils.String(
        allow_none=False,
        description="The column name to perform datetime splitting on.",
    )


def get_split_conds():
    """Returns a JSON schema of conditionals to validate against optimizer types defined in
    `ludwig.modules.optimization_modules.optimizer_registry`."""
    conds = []
    for splitter in split_registry:
        splitter_cls = split_registry[splitter].get_schema_cls()
        other_props = schema_utils.unload_jsonschema_from_marshmallow_class(splitter_cls)["properties"]
        other_props.pop("type")
        splitter_cond = schema_utils.create_cond(
            {"type": splitter},
            other_props,
        )
        conds.append(splitter_cond)
    return conds


def SplitDataclassField(default: str):
    """Custom dataclass field that when used inside a dataclass will allow the user to specify a nested split config.

    Returns: Initialized dataclass field that converts an untyped dict with params to a split config.
    """

    class SplitMarshmallowField(fields.Field):
        """Custom marshmallow field that deserializes a dict for a valid split config from the
        split_registry and creates a corresponding JSON schema for external usage."""

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return None
            if isinstance(value, dict):
                if TYPE in value and value[TYPE] in split_registry:
                    split_class = split_registry[value[TYPE]]
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
                    "type": {"type": "string", "enum": list(split_registry.keys()), "default": default},
                },
                "title": "split_options",
                "allOf": get_split_conds()
            }

    try:
        splitter = split_registry[default].get_schema_cls()
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