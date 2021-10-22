from marshmallow import Schema, fields, validate, INCLUDE, EXCLUDE, ValidationError, post_load
from marshmallow_jsonschema import JSONSchema
from dataclasses import dataclass, field
import marshmallow_dataclass
from typing import List, Dict, Optional, Type, Union


def init_with_kwargs_schema(cls, kwargs):
    schema = cls()
    fields = schema.fields.keys()
    return schema.dump(
        {
            k: v for k, v in kwargs.items()
            if k in fields
        }
    )

# Simple field types (for use with marshmallow-dataclass):

# Note: see https://github.com/marshmallow-code/marshmallow/issues/902
def create_field_StrictBoolean(**kwargs) -> Type[field]:
    return field(metadata={
        **dict(
            truthy=[True],
            falsy=[False],
            validate=validate.Range(min=0, min_inclusive=True),
        ),
        **kwargs
    })

def create_field_NonNegativeInt(**kwargs) -> Type[field]:
    return field(metadata={
        **dict(
            strict=True,
            validate=validate.Range(min=0, min_inclusive=True),
        ),
        **kwargs
    })

def create_field_PositiveInt(**kwargs) -> Type[field]:
    return field(metadata={
        **dict(
            strict=True,
            validate=validate.Range(min=1, min_inclusive=True),
        ),
        **kwargs
    })

def create_field_NormalizedFloat(**kwargs) -> Type[field]:
    return field(metadata={
        **dict(
            # strict=True, # doesn't actually do anything: https://github.com/marshmallow-code/marshmallow/issues/1601
            validate=validate.Range(min=0.0, max=1.0, min_inclusive=True, max_inclusive=True),
        ),
        **kwargs
    })

# Note: since there is no strict option for floats, these two fields (for now) have the same field structure as their
# int counterparts above. The "coaxing" haapens by the type on the dataclass member via marshmallow-dataclass.
def create_field_NonNegativeFloat(**kwargs) -> Type[field]:
    return create_field_NonNegativeInt(kwargs=kwargs)
def create_field_PositiveFloat(**kwargs) -> Type[field]:
    return create_field_PositiveInt(kwargs=kwargs)

# Complex field types
def create_field_EnumType(name, nullable=False, enum_registry=[], **kwargs) -> Type[marshmallow_dataclass.NewType]:
    if nullable:
        # Ensure None is in the registry:
        enum_registry = list(set(enum_registry) | set([None]))

        # Create the enum schema:
        class InnerEnumField(fields.Field):  
            def _jsonschema_type_mapping(self):
                return {
                    "oneOf": [
                        {
                            "type": "null",
                        }, 
                        {
                            'type': 'string',
                            'enum': enum_registry
                        }
                    ]
                }
        
        # Register the dataclass type:
        return marshmallow_dataclass.NewType(
            name=name,
            typ=Union[None, str],
            # dump_default=default,
            # load_default=default,
            **kwargs
        )
        
    return field(metadata=dict(
        validate=validate.OneOf(enum_registry),
        # dump_default=default,
        # load_default=default,
        **kwargs
    ))