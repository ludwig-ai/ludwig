#! /usr/bin/env python
# coding=utf-8
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
import json

import marshmallow_dataclass
from jsonschema import validate
from marshmallow_jsonschema import JSONSchema

from ludwig.combiners.combiners import combiner_registry
from ludwig.features.feature_registries import input_type_registry, output_type_registry

INPUT_FEATURE_TYPES = sorted(list(input_type_registry.keys()))
OUTPUT_FEATURE_TYPES = sorted(list(output_type_registry.keys()))
COMBINER_TYPES = sorted(list(combiner_registry.keys()))

def get_schema():
    schema = {
        'type': 'object',
        'properties': {
            'input_features': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'name': {'type': 'string'},
                        'type': {'type': 'string', 'enum': INPUT_FEATURE_TYPES},
                        'column': {'type': 'string'},
                        'encoder': {'type': 'string'}
                    },
                    'allOf': get_input_encoder_conds() + get_input_preproc_conds(),
                    'required': ['name', 'type'],
                }
            },
            'output_features': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'name': {'type': 'string'},
                        'type': {'type': 'string', 'enum': OUTPUT_FEATURE_TYPES},
                        'column': {'type': 'string'},
                        'decoder': {'type': 'string'}
                    },
                    'allOf': get_output_decoder_conds() + get_output_preproc_conds(),
                    'required': ['name', 'type'],
                }
            },
            'combiner': {
                'type': 'object',
                'properties': {
                    'type': {'type': 'string', 'enum': COMBINER_TYPES},
                },
                'allOf': get_combiner_conds(),
                'required': ['type'],
            },
            'training': {},
            'preprocessing': {},
            'hyperopt': {},
        },
        'definitions': get_custom_definitions(),
        'required': ['input_features', 'output_features']
    }
    return schema


def get_input_encoder_conds():
    conds = []
    for feature_type in INPUT_FEATURE_TYPES:
        feature_cls = input_type_registry[feature_type]
        encoder_names = list(feature_cls.encoder_registry.keys())
        encoder_cond = create_cond(
            {'type': feature_type},
            {'encoder': {'enum': encoder_names}},
        )
        conds.append(encoder_cond)
    return conds


def get_input_preproc_conds():
    conds = []
    for feature_type in INPUT_FEATURE_TYPES:
        feature_cls = input_type_registry[feature_type]
        preproc_spec = {
            'type': 'object',
            'properties': feature_cls.preprocessing_schema,
            'additionalProperties': False,
        }
        preproc_cond = create_cond(
            {'type': feature_type},
            {'preprocessing': preproc_spec},
        )
        conds.append(preproc_cond)
    return conds


def get_output_decoder_conds():
    conds = []
    for feature_type in OUTPUT_FEATURE_TYPES:
        feature_cls = output_type_registry[feature_type]
        decoder_names = list(feature_cls.decoder_registry.keys())
        decoder_cond = create_cond(
            {'type': feature_type},
            {'decoder': {'enum': decoder_names}},
        )
        conds.append(decoder_cond)
    return conds


def get_output_preproc_conds():
    conds = []
    for feature_type in OUTPUT_FEATURE_TYPES:
        feature_cls = output_type_registry[feature_type]
        preproc_spec = {
            'type': 'object',
            'properties': feature_cls.preprocessing_schema,
            'additionalProperties': False,
        }
        preproc_cond = create_cond(
            {'type': feature_type},
            {'preprocessing': preproc_spec},
        )
        conds.append(preproc_cond)
    return conds


def get_combiner_conds():
    conds = []
    for combiner_type in COMBINER_TYPES:
        combiner_cls = combiner_registry[combiner_type]
        schema_cls = combiner_cls.get_schema_cls()
        schema = marshmallow_dataclass.class_schema(schema_cls)()
        schema_json = JSONSchema().dump(schema)
        combiner_json = schema_json['definitions'][schema_cls.__name__]['properties']
        combiner_cond = create_cond(
            {'type': combiner_type},
            combiner_json
        )
        conds.append(combiner_cond)
    return conds

def get_custom_definitions():
    return {}

def create_cond(if_pred, then_pred):
    return {
        'if': {
            'properties': {k: {'const': v} for k, v in if_pred.items()}
        },
        'then': {
            'properties': {k: v for k, v in then_pred.items()}
        }
    }


def validate_config(config):
    validate(instance=config, schema=get_schema())