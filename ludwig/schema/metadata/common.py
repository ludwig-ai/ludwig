from dataclasses import field
from enum import Enum
from typing import Any, List, Union

from marshmallow import fields, ValidationError
from marshmallow_dataclass import dataclass

from ludwig.schema.utils import BaseMarshmallowConfig


class ExpectedImpact(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


def TypeAnnotation(default: Union[str, List[str], None] = None):
    class TypeAnnotationField(fields.Field):
        def _deserialize(self, value, attr, data, **kwargs):
            if isinstance(value, str):
                return value
            if isinstance(value, list) and all([isinstance(elt, str) for elt in value]):
                return value
            raise ValidationError("Field should be either a string or list of strings")

    return field(
        metadata={
            "marshmallow_field": TypeAnnotationField(
                allow_none=True,
                load_default=default,
                dump_default=default,
            )
        },
        default_factory=lambda: default,
    )


def RelatedParameters(default: Union[List[str], None] = None):
    return field(
        metadata={
            "marshmallow_field": fields.List(
                fields.String(),
                allow_none=True,
                load_default=default,
                dump_default=default,
            )
        },
        default_factory=lambda: default,
    )


# Note that the purpose of this schema for now is just runtime validation - the 'schema' here is currenlty not
# able to be represented in JSON and is not intended to be included as part of the Ludwig schema. In
# real-world scenarios, a Ludwig dev writes or adjusts the metadata for a particular parameter on some other
# parameter (that actually is fully represented with a JSON schema), and that new edit should be validated but all of
# these parameters are pulled out of this config and added to the `metadata` property of that parameter's
# JSON schema.
#
# Subject to change?
@dataclass
class BaseMetadataConfig(BaseMarshmallowConfig):
    """BaseMetadataConfig is a dataclass that configures the default metadata values for any Ludwig parameter."""

    allow_none: bool = False
    default_value: Any = None
    ui_display_name: str = ""
    type_annotation: Union[str, List[str], None] = TypeAnnotation()
    description: Union[str, None] = None
    default_value_reasoning: Union[str, None] = None
    example_value: Any = None
    related_parameters: Union[List[str], None] = RelatedParameters()
    other_information: Union[str, None] = None
    description_implications: Union[str, None] = None
    suggested_values: Any = None
    suggested_values_reasoning: Union[str, None] = None
    commonly_used: bool = False
    expected_impact: ExpectedImpact = ExpectedImpact.LOW
    literature_references: Union[str, None] = None
