from abc import ABC
from dataclasses import field
from typing import ClassVar, Optional, Union

from marshmallow import fields, ValidationError
from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils


@dataclass
class BasePreprocessingConfig(schema_utils.BaseMarshmallowConfig, ABC):
    """Base class for input feature preprocessing. Not meant to be used directly.

    The dataclass format prevents arbitrary properties from being set. Consequently, in child classes, all properties
    from the corresponding input feature class are copied over: check each class to check which attributes are
    different from the preprocessing of each feature.
    """

    feature_type: ClassVar[Optional[str]] = None
    "Class variable pointing to the corresponding preprocessor."
    type: str


@dataclass
class BinaryPreprocessingConfig(schema_utils.BaseMarshmallowConfig):
    """BinaryPreprocessingConfig is a dataclass that configures the parameters used for a binary input feature."""

    missing_value_strategy: Optional[str] = schema_utils.StringOptions(
        ["fill_with_false", "fill_with_const", "fill_with_mode", "fill_with_mean", "backfill"],
        default="fill_with_false",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a binary column",
    )

    fill_value: Union[int, float] = schema_utils.NumericOrStringOptionsField(
        ["yes", "YES", "Yes", "y", "Y", "true", "True", "TRUE", "t", "T", "1", "1.0", "no", "NO", "No", "n", "N",
         "false", "False", "FALSE", "f", "F", "0", "0.0"],
        allow_none=False,
        default=None,
        default_numeric=0,
        min=0,
        max=1,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    fallback_true_label: Optional[str] = schema_utils.NumericOrStringOptionsField(
        ["True", "False"],
        allow_none=True,
        default=None,
        default_numeric=1,
        default_option=None,
        min=0,
        max=1,
        description="The label to interpret as 1 (True) when the binary feature doesn't have a "
                    "conventional boolean value"
    )


@dataclass
class CategoryPreprocessingConfig(schema_utils.BaseMarshmallowConfig):
    """CategoryPreprocessingConfig is a dataclass that configures the parameters used for a category input feature."""

    missing_value_strategy: Optional[str] = schema_utils.StringOptions(
        ["fill_with_const", "fill_with_mode", "fill_with_mean", "backfill"],
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a category column",
    )

    fill_value: Optional[str] = schema_utils.String(
        default="<UNK>",
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    lowercase: Optional[bool] = schema_utils.Boolean(
        default=False,
        description="Whether the string has to be lowercased before being handled by the tokenizer.",
    )

    most_common_label: Optional[int] = schema_utils.PositiveInteger(
        default=10000,
        allow_none=True,
        description="The maximum number of most common tokens to be considered. if the data contains more than this "
                    "amount, the most infrequent tokens will be treated as unknown.",
    )


@dataclass
class NumberPreprocessingConfig(schema_utils.BaseMarshmallowConfig):
    """NumberPreprocessingConfig is a dataclass that configures the parameters used for a number input feature."""

    missing_value_strategy: Optional[str] = schema_utils.StringOptions(
        ["fill_with_const", "fill_with_mode", "fill_with_mean", "backfill"],
        default="fill_with_const",
        allow_none=False,
        description="What strategy to follow when there's a missing value in a number column",
    )

    fill_value: Optional[float] = schema_utils.NonNegativeFloat(
        default=0.0,
        allow_none=False,
        description="The value to replace missing values with in case the missing_value_strategy is fill_with_const",
    )

    normalization: Optional[str] = schema_utils.StringOptions(
        ["zscore", "minmax", "log1p"],
        default=None,
        allow_none=True,
        description="Normalization strategy to use for this number feature.",
    )


def PreprocessingDataclassField(description="TODO"):
    """Custom dataclass field that when used inside of a dataclass will allow any optimizer in
    `ludwig.modules.optimization_modules.optimizer_registry`.

    Sets default optimizer to 'adam'.

    :param default: Dict specifying an optimizer with a `type` field and its associated parameters. Will attempt to use
           `type` to load optimizer from registry with given params. (default: {"type": "adam"}).
    :return: Initialized dataclass field that converts untyped dicts with params to optimizer dataclass instances.
    """

    class PreprocesingMarshmallowField(fields.Field):
        """Custom marshmallow field that deserializes a dict to a valid optimizer from
        `ludwig.modules.optimization_modules.optimizer_registry` and creates a corresponding `oneOf` JSON schema
        for external usage."""

        def _deserialize(self, value, attr, data, **kwargs):
            if value is None:
                return None
            if isinstance(value, dict):
                if "type" in value and value["type"] in optimizer_registry:
                    opt = optimizer_registry[value["type"].lower()][1]
                    try:
                        return opt.Schema().load(value)
                    except (TypeError, ValidationError) as e:
                        raise ValidationError(
                            f"Invalid params for optimizer: {value}, see `{opt}` definition. Error: {e}"
                        )
                raise ValidationError(
                    f"Invalid params for optimizer: {value}, expect dict with at least a valid `type` attribute."
                )
            raise ValidationError("Field should be None or dict")

        def _jsonschema_type_mapping(self):
            # Note that this uses the same conditional pattern as combiners:
            return {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(optimizer_registry.keys()), "default": default["type"]},
                },
                "title": "optimizer_options",
                "allOf": get_optimizer_conds(),
                "required": ["type"],
                "description": description,
            }

    if not isinstance(default, dict) or "type" not in default or default["type"] not in optimizer_registry:
        raise ValidationError(f"Invalid default: `{default}`")
    try:
        opt = optimizer_registry[default["type"].lower()][1]
        load_default = opt.Schema().load(default)
        dump_default = opt.Schema().dump(default)

        return field(
            metadata={
                "marshmallow_field": OptimizerMarshmallowField(
                    allow_none=False,
                    dump_default=dump_default,
                    load_default=load_default,
                    metadata={"description": description},
                )
            },
            default_factory=lambda: load_default,
        )
    except Exception as e:
        raise ValidationError(f"Unsupported optimizer type: {default['type']}. See optimizer_registry. Details: {e}")