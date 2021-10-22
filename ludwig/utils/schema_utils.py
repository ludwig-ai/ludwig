from marshmallow import Schema, fields, validate, INCLUDE, EXCLUDE, ValidationError, post_load
from marshmallow_jsonschema import JSONSchema
from dataclasses import dataclass, field
import marshmallow_dataclass
from typing import List, Dict, Optional, Type, Union
from ludwig.modules.initializer_modules import initializers_registry


def init_with_kwargs_schema(cls, kwargs):
    schema = cls()
    fields = schema.fields.keys()
    return schema.dump(
        {
            k: v for k, v in kwargs.items()
            if k in fields
        }
    )

# Declare/shortcut to parameter registries:
preset_weights_initializer_registry = list(initializers_registry.keys())
preset_bias_initializer_registry = list(initializers_registry.keys())
weights_regularizer_registry = ['l1', 'l2', 'l1_l2']
bias_regularizer_registry = ['l1', 'l2', 'l1_l2']
activity_regularizer_registry = ['l1', 'l2', 'l1_l2']
norm_registry = ['batch', 'layer']
activation_registry = ['relu']
reduce_output_registry = ['sum', 'mean', 'sqrt', 'concat', None]


# Simple field types (for use with marshmallow-dataclass):

# Note: see https://github.com/marshmallow-code/marshmallow/issues/902
def create_field_StrictBoolean(f_default: Optional[bool]=None,**kwargs) -> Type[field]:
    if f_default is not None:
        kwargs['dump_default'] = f_default
        kwargs['load_default'] = f_default
    return field(metadata={
        **dict(
            truthy=[True],
            falsy=[False],
            validate=validate.Range(min=0, min_inclusive=True),
        ),
        **kwargs
    })

# def create_type_StrictBoolean(f_default: Optional[bool]=None, **kwargs) -> Type[marshmallow_dataclass.NewType]:
#     return marshmallow_dataclass.NewType(
#         name='StrictBoolean',
#         typ=bool,
#         required=False,
#         field=create_field_StrictBoolean(f_default, **kwargs)
#     )

def create_field_NonNegativeInt(f_default: Optional[int]=None, **kwargs) -> Type[field]:
    if f_default is not None:
        kwargs['dump_default'] = f_default
        kwargs['load_default'] = f_default
    return field(metadata={
        **dict(
            strict=True,
            validate=validate.Range(min=0, min_inclusive=True),
        ),
        **kwargs
    })

def create_field_PositiveInt(f_default: Optional[int], **kwargs) -> Type[field]:
    if f_default is not None:
        kwargs['dump_default'] = f_default
        kwargs['load_default'] = f_default
    return field(metadata={
        **dict(
            strict=True,
            validate=validate.Range(min=1, min_inclusive=True),
        ),
        **kwargs
    })

def create_field_NormalizedFloat(f_default: Optional[float], **kwargs) -> Type[field]:
    if f_default is not None:
        kwargs['dump_default'] = f_default
        kwargs['load_default'] = f_default
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

# Thinking about what behavior was before this PR, consider simple scenario
# fc_size=1 as a default param to some combiner.
# What does this mean in the context of JSON? If there is *any* default value at all, then we must both serialize and deserialize
# to that default value in the JSON context - i.e. the field is always "Optional" - the only difference is that in some cases
# we do not want to allow 'None' as an option that can be specified.
def create_type_EnumType(name, enum_registry, required=False, nullable=False, **kwargs) -> Type[marshmallow_dataclass.NewType]:
    if 'load_default' in kwargs or 'dump_default' in kwargs:
        raise ValidationError("Use 'default_enum_value' instead with this method.")
    if required == ('default_enum_value' in kwargs):
        raise ValidationError("Invalid specification. If a field is not required, you must provide a "\
            "'default_enum_value' argument as an extra option. If a field is required, no default is "\
            "allowed.")

    allowed_enums = list(set(enum_registry) | set([None])) if nullable else enum_registry
    if not required and kwargs['default_enum_value'] not in allowed_enums:
        raise ValidationError(f"Provided default '{kwargs['default_enum_value']}' is not one of acceptable: {allowed_enums}")

    type_params = {}
    if not required:
        type_params['dump_default'] = kwargs['default_enum_value']
        type_params['load_default'] = kwargs['default_enum_value']

    # Create the enum schema:
    # TODO: explore better way for validation rather than in both (de)serialize methods?
    class InnerEnumField(fields.Field):
        # WARNING: I shouldn't need to override __init__, but I cannot figure out why load_default, allow_none,
        # and required do not get set through the normal flow. However, behavior seems good enough for our
        # purposes.
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if not required:
                self.dump_default = type_params['dump_default']
                self.load_default = type_params['load_default']
            self.allow_none = nullable
            self.required = required

        def _deserialize(self, value, attr, data, **kwargs):
            if not value in allowed_enums:
                raise ValidationError(f"Value '{value}' is not one of acceptable: {allowed_enums}")
            return value

        def _serialize(self, value, attr, obj, **kwargs):
            if not value in allowed_enums:
                raise ValidationError(f"Value '{value}' is not one of acceptable: {allowed_enums}")
            return None if value is None else value

        def _jsonschema_type_mapping(self):
            json_mapping = {'description': name}
            if not required:
                json_mapping['default'] = kwargs['default_enum_value']

            if nullable:
                json_mapping["oneOf"] = [
                    {
                        "type": "null", # Note that here it actually has to "null" not None
                    }, 
                    {
                        'type': 'string',
                        'enum': enum_registry
                    }
                ]
            else:
                json_mapping = {
                    **json_mapping,
                    **{
                        "type": "string",
                        "enum": enum_registry,
                        "enumNames": []
                    }
                }
            return json_mapping
        
    # Register the dataclass type:
    allowed_types = Union[None, str] if nullable else str
    t = marshmallow_dataclass.NewType(
        name=name,
        typ=allowed_types,
        field=InnerEnumField,
        **kwargs
    )

    # Unnecessary if self.required is set via __init__:
    # if not required:
    #     # Necessary for correct construction with dataclass:
    #     return Optional[t]
    return t


def create_type_InitializerType(name, enum_registry, required=False, nullable=False, **kwargs) -> Type[marshmallow_dataclass.NewType]:
    if 'load_default' in kwargs or 'dump_default' in kwargs:
        raise ValidationError("Use 'default_enum_value' instead with this method.")
    if required == ('default_enum_value' in kwargs):
        raise ValidationError("Invalid specification. If a field is not required, you must provide a "\
            "'default_enum_value' argument as an extra option. If a field is required, no default is "\
            "allowed.")

    allowed_enums = list(set(enum_registry) | set([None])) if nullable else enum_registry
    if not required and kwargs['default_enum_value'] not in allowed_enums:
        raise ValidationError(f"Provided default '{kwargs['default_enum_value']}' is not one of acceptable: {allowed_enums}")

    type_params = {}
    if not required:
        type_params['dump_default'] = kwargs['default_enum_value']
        type_params['load_default'] = kwargs['default_enum_value']

    # Create the enum schema:
    # TODO: explore better way for validation rather than in both (de)serialize methods?
    class InnerEnumField(fields.Field):
        # WARNING: I shouldn't need to override __init__, but I cannot figure out why load_default, allow_none,
        # and required do not get set through the normal flow. However, behavior seems good enough for our
        # purposes.
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if not required:
                self.dump_default = type_params['dump_default']
                self.load_default = type_params['load_default']
            self.allow_none = nullable
            self.required = required

        def _deserialize(self, value, attr, data, **kwargs):
            if not value in allowed_enums:
                raise ValidationError(f"Value '{value}' is not one of acceptable: {allowed_enums}")
            return value

        def _serialize(self, value, attr, obj, **kwargs):
            if not value in allowed_enums:
                raise ValidationError(f"Value '{value}' is not one of acceptable: {allowed_enums}")
            return None if value is None else value

        def _jsonschema_type_mapping(self):
            json_mapping = {'description': name}
            if not required:
                json_mapping['default'] = kwargs['default_enum_value']
            json_mapping['oneOf'] = [
                {
                    "type": "object",
                }, 
                {
                    'type': 'string',
                    'enum': enum_registry,
                    "enumNames": []

                }
            ]
            if nullable:
                json_mapping["oneOf"] = json_mapping["oneOf"] + { "type": "null" }

            return json_mapping
        
    # Register the dataclass type:
    allowed_types = Union[None, str] if nullable else str
    t = marshmallow_dataclass.NewType(
        name=name,
        typ=allowed_types,
        field=InnerEnumField,
        **kwargs
    )

    return t


# Initializers accept presets or customized dicts (not JSON-validated):
class WeightsInitializerField(fields.Field):
    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, str) or isinstance(value, dict):
            return value
        else:
            raise ValidationError('Field should be str or dict')
    
    def _jsonschema_type_mapping(self):
        return {
            "anyOf": [
                {
                    "type": "object",
                }, 
                {
                    'type': 'string',
                    'enum': preset_weights_initializer_registry
                }
            ]
        }
WeightsInitializerType = marshmallow_dataclass.NewType(
        name='WeightsInitializerType',
        typ=Union[str, dict],
        required=False,
        dump_default='glorot_uniform',
        load_default='glorot_uniform'
)

class BiasInitializerField(fields.Field):
    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, str) or isinstance(value, dict):
            return value
        else:
            raise ValidationError('Field should be str or dict')
    
    def _jsonschema_type_mapping(self):

        return {
            "anyOf": [
                {
                    "type": "object",
                }, 
                {
                    'type': 'string',
                    'enum': preset_bias_initializer_registry
                }
            ]
        }
BiasInitializerType = marshmallow_dataclass.NewType(
    name='BiasInitializerType',
    typ=Union[str, dict],
    required=False,
    dump_default='zeros',
    load_default='zeros'
)