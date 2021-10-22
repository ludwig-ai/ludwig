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
import ludwig.utils.schema_utils as lus

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

def create_field_NonNegativeInt(f_default: Optional[int]=None, **kwargs) -> Type[field]:
    if f_default is not None:
        kwargs = {
            **kwargs,
            **dict(
                load_default=f_default,
                dump_default=f_default
            )
        }
        # kwargs['dump_default'] = f_default
        # kwargs['load_default'] = f_default
    metadata={
        **dict(
            strict=True,
            validate=validate.Range(min=0, min_inclusive=True),
        ),
        **kwargs
    }
    print(metadata)
    return field(metadata={
        **dict(
            strict=True,
            validate=validate.Range(min=0, min_inclusive=True),
        ),
        **kwargs
    })

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
    activation: create_field_EnumType('ActivationType', activation_registry, default_enum_value='relu')

   # weights_initializer: WeightsInitializerType
    # bias_initializer: BiasInitializerType
    weights_regularizer_:create_field_EnumType(
        'WeightsRegularizerType',
        weights_regularizer_registry,
        required=False,
        nullable=False,
        default_enum_value='l2'
    )
    weights_regularizer_r:create_field_EnumType(
        'WeightsRegularizerType',
        weights_regularizer_registry,
        required=True,
        nullable=False,
        #default_enum_value='l1'
    )
    weights_regularizer_n:create_field_EnumType(
        'WeightsRegularizerType',
        weights_regularizer_registry,
        required=False,
        nullable=True,
        default_enum_value='l1_l2'
    )
    weights_regularizer_rn:create_field_EnumType(
        'WeightsRegularizerType',
        weights_regularizer_registry,
        required=True,
        nullable=True,
        # default_enum_value=None
    )

    fc_size: Optional[int] = create_field_NonNegativeInt(f_default=256)

    fc_layers: List[dict] = field(default_factory=list,
        metadata=dict(
            dump_default=None,
            load_default=None
        )
    )

    class Meta:
        unknown = INCLUDE

ConcatCombinerSchema = marshmallow_dataclass.class_schema(ConcatCombinerData)()
# print()
# print(jsonGenerator.dump(ConcatCombinerSchema))
# print(json.dumps(jsonGenerator.dump(ConcatCombinerSchema)))
# print(ConcatCombinerSchema.dump({'weights_regularizer_rn': None, 'weights_regularizer_r': 'l1_l2'}))
# print(ConcatCombinerSchema.load({'weights_regularizer_rn': None, 'weights_regularizer_r': 'l1'}))


@dataclass
class TestData:
    # weights_initializer: lus.WeightsInitializerType
    # bias_initializer: lus.BiasInitializerType
    weights_regularizer: lus.create_type_EnumType(
        'WeightsRegularizerType',
        lus.weights_regularizer_registry,
        nullable=True,
        default_enum_value=None
    )
    bias_regularizer: lus.create_type_EnumType(
        'BiasRegularizerType',
        lus.bias_regularizer_registry,
        nullable=True,
        default_enum_value=None
    )
    activity_regularizer: lus.create_type_EnumType(
        'ActivityRegularizerType',
        lus.activity_regularizer_registry,
        nullable=True,
        default_enum_value=None
    )
    norm: lus.create_type_EnumType(
        'NormType',
        lus.norm_registry,
        nullable=True,
        default_enum_value=None
    )
    activation: lus.create_type_EnumType(
        'ActivationType', 
        lus.activation_registry, 
        default_enum_value='relu'
    )
    # use_bias: lus.create_type_StrictBoolean(False)

    fc_size: Optional[int] = lus.create_field_PositiveInt(256)
    use_bias: Optional[bool] = lus.create_field_StrictBoolean(True)
    dropout: Optional[float] = lus.create_field_NormalizedFloat(0.0)
    flatten_inputs: Optional[bool] = lus.create_field_StrictBoolean(False)
    residual: Optional[bool] = lus.create_field_StrictBoolean(False)

    num_fc_layers: Optional[int] = field(metadata=dict(
        strict=True,
        validate=validate.Range(min=0, min_inclusive=True),
        dump_default=None,
        load_default=None
    ))

    norm_params: Optional[dict] = field(metadata=dict(
        allow_none=True,
        dump_default=None,
        load_default=None
    ))
    fc_layers: Optional[List[dict]] = field(default_factory=list,
        metadata=dict(
            dump_default=None,
            load_default=None
        )
    )
    class Meta:
        unknown = INCLUDE

TestSchema = marshmallow_dataclass.class_schema(TestData)()
print(TestSchema.dump({}))
print(TestSchema.load({}))