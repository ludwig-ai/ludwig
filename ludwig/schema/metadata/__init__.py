import os
from typing import Any, Dict, Union

import yaml

from ludwig.schema.metadata.parameter_metadata import ParameterMetadata

_PATH_HERE = os.path.abspath(os.path.dirname(__file__))
_CONFIG_DIR = os.path.join(_PATH_HERE, "configs")


def _to_metadata(d: Dict[str, Any]) -> Union[ParameterMetadata, Dict[str, Any]]:
    is_nested = False
    for k, v in list(d.items()):
        if isinstance(v, dict):
            d[k] = _to_metadata(v)
            is_nested = True

    if is_nested:
        return d

    return ParameterMetadata.from_dict(d)


def _load(fname: str) -> Dict[str, Any]:
    with open(os.path.join(_CONFIG_DIR, fname)) as f:
        return _to_metadata(yaml.safe_load(f))


COMMON_METADATA = _load("common.yaml")
COMBINER_METADATA = _load("combiners.yaml")
DECODER_METADATA = _load("decoders.yaml")
ENCODER_METADATA = _load("encoders.yaml")
FEATURE_METADATA = _load("features.yaml")
PREPROCESSING_METADATA = _load("preprocessing.yaml")
TRAINER_METADATA = _load("trainer.yaml")
OPTIMIZER_METADATA = _load("optimizers.yaml")
LOSS_METADATA = _load("loss.yaml")
