#! /usr/bin/env python
# Copyright (c) 2022 Predibase, Inc.
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
import logging
from typing import Dict, List, Type, Union

from ludwig.encoders.base import Encoder
from ludwig.modules import serialization
from ludwig.modules.ludwig_module import register_module
from ludwig.utils.fs_utils import is_url
from ludwig.utils.registry import Registry

logger = logging.getLogger(__name__)

encoder_registry = Registry()

sequence_encoder_registry = Registry()


def register_sequence_encoder(name: str):
    def wrap(cls):
        register_module(cls)
        sequence_encoder_registry[name] = cls
        return cls

    return wrap


def register_encoder(name: str, features: Union[str, List[str]]):
    if isinstance(features, str):
        features = [features]

    def wrap(cls):
        register_module(cls)
        for feature in features:
            feature_registry = encoder_registry.get(feature, {})
            feature_registry[name] = cls
            encoder_registry[feature] = feature_registry
        return cls

    return wrap


def get_encoder_default_params(feature: str, encoder_name_or_url: str) -> Dict:
    """Get the default parameters dict for an encoder.

    Encoder may be the name of an encoder, or the URL of a pre-trained encoder.
    """
    # TODO(daniel): Cache referenced encoders so we don't load from disk multiple times.
    if is_url(encoder_name_or_url):
        try:
            encoder_state = serialization.load_state_from_file(encoder_name_or_url)
            return encoder_state.config
        except Exception as e:
            logger.error(f"Failed to load encoder from {encoder_name_or_url}.")
            logger.exception(e)
    else:
        encoder_cls = get_encoder_cls(feature, encoder_name_or_url)
        return getattr(encoder_cls, "default_params", {})


def get_encoder_cls(feature: str, name: str) -> Type[Encoder]:
    return encoder_registry[feature][name]


def get_encoder_classes(feature: str) -> Dict[str, Type[Encoder]]:
    return encoder_registry[feature]
