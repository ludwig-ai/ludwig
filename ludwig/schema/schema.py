"""Module description: ludwig.schema               <-- Meant to contain all schemas, utilities, helpers related to
describing and validating.

|                               Ludwig configs. ├── utils.py                <-- An extensive set of marshmallow-related
fields, methods, and schemas that are used |                               elsewhere in Ludwig. ├── schema.py
<-- Contains the fully assembled Ludwig schema and validate() function that is used for |
user-input YAML validation. Users should generally only need to look at this. └── __init__.py
"""
from jsonschema import Draft7Validator, validate
from jsonschema.validators import extend

from ludwig.combiners.combiners import get_combiner_jsonschema
from ludwig.constants import COMBINER, HYPEROPT, PREPROCESSING, TRAINER
from ludwig.decoders.registry import get_decoder_classes
from ludwig.encoders.registry import get_encoder_classes
from ludwig.features.feature_registries import input_type_registry, output_type_registry
from ludwig.models.trainer import get_trainer_jsonschema
from ludwig.schema.utils import create_cond


def get_schema():
    input_feature_types = sorted(list(input_type_registry.keys()))
    output_feature_types = sorted(list(output_type_registry.keys()))

    schema = {
        "type": "object",
        "properties": {
            "input_features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": input_feature_types},
                        "column": {"type": "string"},
                        "encoder": {"type": "string"},
                    },
                    "allOf": get_input_encoder_conds(input_feature_types)
                    + get_input_preproc_conds(input_feature_types),
                    "required": ["name", "type"],
                },
            },
            "output_features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": output_feature_types},
                        "column": {"type": "string"},
                        "decoder": {"type": "string"},
                    },
                    "allOf": get_output_decoder_conds(output_feature_types)
                    + get_output_preproc_conds(output_feature_types),
                    "required": ["name", "type"],
                },
            },
            COMBINER: get_combiner_jsonschema(),
            TRAINER: get_trainer_jsonschema(),
            PREPROCESSING: {},
            HYPEROPT: {},
        },
        "definitions": get_custom_definitions(),
        "required": ["input_features", "output_features"],
    }
    return schema


def get_input_encoder_conds(input_feature_types):
    conds = []
    for feature_type in input_feature_types:
        encoder_names = list(get_encoder_classes(feature_type).keys())
        encoder_cond = create_cond(
            {"type": feature_type},
            {"encoder": {"enum": encoder_names}},
        )
        conds.append(encoder_cond)
    return conds


def get_input_preproc_conds(input_feature_types):
    conds = []
    for feature_type in input_feature_types:
        feature_cls = input_type_registry[feature_type]
        preproc_spec = {
            "type": "object",
            "properties": feature_cls.preprocessing_schema(),
            "additionalProperties": False,
        }
        preproc_cond = create_cond(
            {"type": feature_type},
            {"preprocessing": preproc_spec},
        )
        conds.append(preproc_cond)
    return conds


def get_output_decoder_conds(output_feature_types):
    conds = []
    for feature_type in output_feature_types:
        decoder_names = list(get_decoder_classes(feature_type).keys())
        decoder_cond = create_cond(
            {"type": feature_type},
            {"decoder": {"enum": decoder_names}},
        )
        conds.append(decoder_cond)
    return conds


def get_output_preproc_conds(output_feature_types):
    conds = []
    for feature_type in output_feature_types:
        feature_cls = output_type_registry[feature_type]
        preproc_spec = {
            "type": "object",
            "properties": feature_cls.preprocessing_schema(),
            "additionalProperties": False,
        }
        preproc_cond = create_cond(
            {"type": feature_type},
            {"preprocessing": preproc_spec},
        )
        conds.append(preproc_cond)
    return conds


def get_custom_definitions():
    return {}


def validate_config(config):
    # Manually add support for tuples (pending upstream changes: https://github.com/Julian/jsonschema/issues/148):
    def custom_is_array(checker, instance):
        return isinstance(instance, list) or isinstance(instance, tuple)

    # TODO(#1783): Change to Draft7Validator to _LATEST_VERSION or Draft202012Validator when py3.6 deprecated:
    type_checker = Draft7Validator.TYPE_CHECKER.redefine("array", custom_is_array)
    CustomValidator = extend(Draft7Validator, type_checker=type_checker)
    validate(instance=config, schema=get_schema(), cls=CustomValidator)
