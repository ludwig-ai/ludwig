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


def to_metadata(d: Dict[str, Any]) -> ParameterMetadata:
    return ParameterMetadata.from_dict(d)
