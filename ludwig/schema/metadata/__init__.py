import os
from typing import Any, Dict

import yaml

from ludwig.schema.metadata.parameter_metadata import ParameterMetadata


_PATH_HERE = os.path.abspath(os.path.dirname(__file__))
_CONFIG_DIR = os.path.join(_PATH_HERE, "configs")


def _load(fname: str) -> Dict[str, Any]:
    return yaml.load(os.path.join(_CONFIG_DIR, fname))


COMBINER_METADATA = _load("combiners.yaml")
DECODER_METADATA = _load("decoders.yaml")
ENCODER_METADATA = _load("encoders.yaml")
FEATURE_METADATA = _load("features.yaml")
PREPROCESSING_METADATA = _load("preprocessing.yaml")
TRAINER_METADATA = _load("trainer.yaml")


def get_combiner_metadata(*keys) -> ParameterMetadata:
    return _get_metadata(COMBINER_METADATA, *keys)


def get_decoder_metadata(*keys) -> ParameterMetadata:
    return _get_metadata(DECODER_METADATA, *keys)


def get_encoder_metadata(*keys) -> ParameterMetadata:
    return _get_metadata(ENCODER_METADATA, *keys)


def get_feature_metadata(*keys) -> ParameterMetadata:
    return _get_metadata(FEATURE_METADATA, *keys)


def get_preprocessing_metadata(*keys) -> ParameterMetadata:
    return _get_metadata(PREPROCESSING_METADATA, *keys)


def get_trainermetadata(*keys) -> ParameterMetadata:
    return _get_metadata(TRAINER_METADATA, *keys)


def _get_metadata(d: Dict[str, Any], *keys) -> ParameterMetadata:
    for k in keys:
        d = d[k]
    return ParameterMetadata.from_dict(d)
