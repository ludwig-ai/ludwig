import copy
from typing import Any, Dict, Mapping
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import (
    DEFAULTS,
    INPUT_FEATURES,
    OUTPUT_FEATURES,
    TYPE,
)
from ludwig.schema.encoders.utils import get_encoder_cls
from ludwig.schema.features.utils import input_config_registry
from ludwig.types import ModelConfigDict
from ludwig.utils.misc_utils import merge_dict


@DeveloperAPI
def merge_with_defaults(config_dict: ModelConfigDict) -> ModelConfigDict:
    # Recursive merge of the features, except that if we find a dictionary containing
    # an explicit "type" key, we ignore defaults if they don't match.
    defaults = config_dict.get(DEFAULTS)
    if not defaults:
        return config_dict

    config_dict = copy.deepcopy(config_dict)
    for feature in config_dict.get(INPUT_FEATURES, []) + config_dict.get(OUTPUT_FEATURES, []):
        ftype = feature.get(TYPE)
        if not ftype:
            continue

        default_feature = defaults.get(ftype, {})
        merged_feature = _merge_dict_with_types(default_feature, feature)

        # In-place replacement of the old feature with the new
        feature.clear()
        feature.update(merged_feature)

    return config_dict


def _merge_dict_with_types(dct: Dict[str, Any], merge_dct: Dict[str, Any]) -> Dict[str, Any]:
    dct = copy.deepcopy(dct)
    for k, v in merge_dct.items():
        # TODO(travis): below type comparison is not perfect, as it doesn't consider the case where one of the types
        # is omitted and the other is the default, in which case they should resolve to equal, but will be considered
        # different.
        if (
            k in dct
            and isinstance(dct[k], dict)
            and isinstance(merge_dct[k], Mapping)
            and dct[k].get(TYPE) == merge_dct[k].get(TYPE)
        ):
            dct[k] = _merge_dict_with_types(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct


@DeveloperAPI
def merge_fixed_preprocessing_params(
    model_type: str, feature_type: str, preprocessing_params: Dict[str, Any], encoder_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Update preprocessing parameters if encoders require fixed preprocessing parameters."""
    feature_cls = input_config_registry(model_type)[feature_type]
    encoder_type = encoder_params.get(TYPE, feature_cls().encoder.type)
    encoder_class = get_encoder_cls(model_type, feature_type, encoder_type)
    encoder = encoder_class.from_dict(encoder_params)
    return merge_dict(preprocessing_params, encoder.get_fixed_preprocessing_params())
