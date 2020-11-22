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

import inspect

from jsonschema import validate

from ludwig.features.feature_registries import input_type_registry, output_type_registry

INPUT_FEATURE_TYPES = sorted(list(input_type_registry.keys()))
OUTPUT_FEATURE_TYPES = sorted(list(output_type_registry.keys()))


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
                        'preprocessing': {},
                        'encoder': {'type': 'string'}
                    },
                    'allOf': get_input_encoder_conds(),
                    'required': ['name', 'type']
                }
            },
            'combiner': {},
            'output_features': {},
            'training': {},
            'preprocessing': {},
        },
        'required': ['input_features', 'output_features']
    }
    return schema


def get_input_encoder_conds():
    conds = []
    for feature_type in INPUT_FEATURE_TYPES:
        feature_cls = input_type_registry[feature_type]
        encoder_names = sorted(list(feature_cls.encoder_registry.keys()))
        encoder_cond = create_cond(
            {'type': feature_type},
            {'encoder': {'enum': encoder_names}},
        )
        conds.append(encoder_cond)

        for encoder_name in encoder_names:
            encoder_cls = feature_cls.encoder_registry[encoder_name]
            signature = inspect.signature(encoder_cls.__init__)
            config_args = [
                k for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty
            ]
            encoder_arg_cond = create_cond(
                {'type': feature_type, 'encoder': encoder_cls},
                {arg: {} for arg in config_args}
            )
            conds.append(encoder_arg_cond)
    return conds


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
