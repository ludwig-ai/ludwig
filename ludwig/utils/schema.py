#! /usr/bin/env python
# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import marshmallow_dataclass
from jsonschema import validate
from marshmallow_jsonschema import JSONSchema

from ludwig.combiners.combiners import combiner_registry
from ludwig.decoders.registry import get_decoder_classes
from ludwig.encoders.registry import get_encoder_classes
from ludwig.features.feature_registries import input_type_registry, output_type_registry
from ludwig.models.trainer import TrainerConfig


def get_schema():
    input_feature_types = sorted(list(input_type_registry.keys()))
    output_feature_types = sorted(list(output_type_registry.keys()))
    combiner_types = sorted(list(combiner_registry.keys()))

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
            "combiner": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": combiner_types},
                },
                "allOf": get_combiner_conds(combiner_types),
                "required": ["type"],
            },
            "training": JSONSchema().dump(TrainerConfig.Schema())["definitions"][TrainerConfig.__name__],
            "preprocessing": {},
            "hyperopt": {},
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


def get_combiner_conds(combiner_types):
    conds = []
    for combiner_type in combiner_types:
        combiner_cls = combiner_registry[combiner_type]
        schema_cls = combiner_cls.get_schema_cls()
        schema = marshmallow_dataclass.class_schema(schema_cls)()
        schema_json = JSONSchema().dump(schema)
        combiner_json = schema_json["definitions"][schema_cls.__name__]["properties"]
        combiner_cond = create_cond({"type": combiner_type}, combiner_json)
        conds.append(combiner_cond)
    return conds


def get_custom_definitions():
    return {}


def create_cond(if_pred, then_pred):
    return {
        "if": {"properties": {k: {"const": v} for k, v in if_pred.items()}},
        "then": {"properties": {k: v for k, v in then_pred.items()}},
    }


def validate_config(config):
    validate(instance=config, schema=get_schema())
