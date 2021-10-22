import enum
from types import MethodWrapperType, SimpleNamespace
from typing import List, Dict, Optional, Type, Union
from marshmallow_jsonschema.base import ALLOW_UNIONS
from marshmallow import Schema, fields, validate, INCLUDE, EXCLUDE, ValidationError, post_load
from marshmallow_jsonschema import JSONSchema
from dataclasses import dataclass, field
import marshmallow_dataclass
from ludwig.modules.initializer_modules import initializers_registry
import json

jsonGenerator = JSONSchema()

# Declare/shortcut to parameter registries:
# TODO: initializers_ actually has None as an option lol
preset_weights_initializer_registry = list(initializers_registry.keys())
preset_bias_initializer_registry = list(initializers_registry.keys())
weights_regularizer_registry = ['l1', 'l2', 'l1_l2']
bias_regularizer_registry = ['l1', 'l2', 'l1_l2']
activity_regularizer_registry = ['l1', 'l2', 'l1_l2']
norm_registry = ['batch', 'layer']
activation_registry = ['relu']
reduce_output_registry = ['sum', 'mean', 'sqrt', 'concat', None]


# Initializers accept presets or customized dicts (not JSON-validated):
class WeightsInitializerField(fields.Field):
    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, str) or isinstance(value, dict):
            return value
        else:
            raise ValidationError('Field should be str or dict')
    
    def _jsonschema_type_mapping(self):
        return {
            "oneOf": [
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
        field=WeightsInitializerField
        # required=False,
        # dump_default='glorot_uniform',
        # load_default='glorot_uniform'
)

# Thinking about what behavior was before this PR at all, consider simple scenario
# fc_size=1 as a default param to some combiner.
# What does this mean in the context of JSON? If there is *any* default value at all, then we must both serialize and deserialize
# to that default value in the JSON context - i.e. the field is always "Optional" - the only difference is that in some cases
# we do not want to allow 'None' as an option that can be specified.

def create_field_EnumType(name, enum_registry, required=False, nullable=False, **kwargs) -> Type[marshmallow_dataclass.NewType]:
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
        print(type_params)

    # Create the enum schema:
    # TODO: Still need to confirm 
    class InnerEnumField(fields.Field):
        # WARNING: I shouldn't need to override __init__, but I cannot figure out why load_default is not set through
        # the normal flow. This hack only modifies that field, so it should work for now.
        def __init__(self, **kwargs):
            if not required:
                kwargs['load_default'] = type_params['load_default'] 
            super().__init__(**kwargs)

        def _deserialize(self, value, attr, data, **kwargs):
            print('deserialize')
            if not value in allowed_enums:
                raise ValidationError(f"Value '{value}' is not one of acceptable: {allowed_enums}")
            return value

        def _serialize(self, value, attr, obj, **kwargs):
            print('serialize')
            if not value in allowed_enums:
                raise ValidationError(f"Value '{value}' is not one of acceptable: {allowed_enums}")
            return None if value is None else value
            # if value is None:
            #     return ""
            # return "".join(str(d) for d in value)


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
        #allow_none=True,
        # _deserialize=InnerEnumField._deserialize
        # required=required,
        **type_params,
        #load_default=kwargs['default_enum_value'],
        #validate=validate.OneOf(enum_registry),
        **kwargs
    )

    if not required:
        # Necessary for correct construction with dataclass:
        return Optional[t]
    return t

    # return marshmallow_dataclass.NewType(
    #     name=name,
    #     typ=str,
    #     validate=validate.OneOf(enum_registry),
    #     required=required,
    #     # dump_default=default,
    #     # load_default=default,
    #     **kwargs
    # )


    # return field(metadata=dict(
    #     validate=validate.OneOf(enum_registry),
    #     **kwargs
    # ))

@dataclass
class ConcatCombinerData:
    # fc_layers: Optional[List[Dict]] = field(metadata=dict(
    #     dump_default=None,
    #     load_default=None
    # ))
    # num_fc_layers: Optional[int] = field(metadata=dict(
    #     strict=True,
    #     validate=validate.Range(min=0, min_inclusive=True),
    #     dump_default=None,
    # ))
    # fc_size: int = field(metadata=dict(
    #     strict=True,
    #     validate=validate.Range(min=1, min_inclusive=True),
    #     dump_default=256,
    #     load_default=256,
    # ))
    # use_bias: bool = field(metadata=dict(
    #     dump_default=True,
    #     load_default=True,
    # ))

   # weights_initializer: WeightsInitializerType
    # bias_initializer: BiasInitializerType
    weights_regularizer:create_field_EnumType(
        'WeightsRegularizerType',
        weights_regularizer_registry,
        required=False,
        nullable=True,
        default_enum_value='l1'
    )
    # bias_regularizer: Optional[str] = field(metadata=dict(
    #     validate=validate.OneOf(bias_regularizer_registry),
    #     allow_none=True,
    #     dump_default=None,
    #     load_default=None
    # ))
    # test: Union[str, dict] = field()
    # activity_regularizer: Optional[str] = field(metadata=dict(
    #     validate=validate.OneOf(activity_regularizer_registry),
    #     allow_none=True,
    #     dump_default=None,
    #     load_default=None
    # ))
    # norm: Optional[str] = field(metadata=dict(
    #     validate=validate.OneOf(norm_registry),
    #     allow_none=True,
    #     dump_default=None,
    #     load_default=None
    # ))
    # norm_params: Optional[dict] = field(metadata=dict(
    #     allow_none=True,
    #     dump_default=None,
    #     load_default=None
    # ))
    # activation: str = field(metadata=dict(
    #     validate=validate.OneOf(activation_registry),
    #     dump_default='relu',
    #     load_default='relu'
    # ))
    # dropout: float = field(metadata=dict(
    #     validate=validate.Range(min=0, max=1, min_inclusive=True, max_inclusive=True),
    #     dump_default=0.0,
    #     load_default=0.0
    # ))
    # flatten_inputs: bool = field(metadata=dict(
    #     dump_default=False,
    #     load_default=False
    # ))
    # residual: bool = field(metadata=dict(
    #     dump_default=False,
    #     load_default=False
    # ))

    # class Meta:
    #     unknown = INCLUDE

ConcatCombinerSchema = marshmallow_dataclass.class_schema(ConcatCombinerData)()

print(jsonGenerator.dump(ConcatCombinerSchema))
print(json.dumps(jsonGenerator.dump(ConcatCombinerSchema)))
print(ConcatCombinerSchema.dump({'weights_regularizer': 'l1'}))
print(ConcatCombinerSchema.load({'weights_regularizer': 'None'}))
