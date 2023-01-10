import copy
from typing import Any, Dict, List, Mapping, TYPE_CHECKING

from marshmallow import ValidationError

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import COLUMN, COMBINED, DEFAULTS, INPUT_FEATURES, LOSS, NAME, OUTPUT_FEATURES, PROC_COLUMN, TYPE
from ludwig.features.feature_utils import compute_feature_hash
from ludwig.schema.encoders.utils import get_encoder_cls
from ludwig.schema.features.base import BaseOutputFeatureConfig, FeatureCollection
from ludwig.schema.features.utils import input_config_registry, output_config_registry
from ludwig.types import ModelConfigDict
from ludwig.utils.misc_utils import merge_dict

if TYPE_CHECKING:
    from ludwig.schema.model_types.base import ModelConfig


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
        # TODO(travis): below type comparison is not perfect, as it doesn't consider the case where the default type
        # is omitted while the encoder type is explicitly set to the default type, in which case they
        # should resolve to equal, but will be considered different.
        if (
            k in dct
            and isinstance(dct[k], dict)
            and isinstance(v, Mapping)
            and dct[k].get(TYPE) == v.get(TYPE, dct[k].get(TYPE))
        ):
            dct[k] = _merge_dict_with_types(dct[k], v)
        else:
            dct[k] = v
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


def set_validation_parameters(config: "ModelConfig"):
    """Sets validation-related parameters used for early stopping, determining the best hyperopt trial, etc."""
    if not config.output_features:
        return

    # The user has explicitly set validation_field. Don't override any validation parameters.
    if config.trainer.validation_field:
        # TODO(Justin): Validate that validation_field is valid.
        return

    # The user has not explicitly set the validation_metric.
    if not config.trainer.validation_metric:
        # The user has not explicitly set any validation fields.
        # Default to using the first output feature's default validation metric.
        config.trainer.validation_field = config.output_features[0].name
        out_type = config.output_features[0].type
        config.trainer.validation_metric = output_config_registry[out_type].default_validation_metric

    # The user has explicitly set the validation_metric.
    # Loss is valid for all features.
    if config.trainer.validation_metric == LOSS:
        return

    # Determine the proper validation field for the user, like if the user specifies "accuracy" but forgets to
    # change the validation field from "combined" to the name of the feature that produces accuracy metrics.
    feature_to_metric_names_map = get_feature_to_metric_names_map(config.output_features)
    validation_field = None
    for feature_name, metric_names in feature_to_metric_names_map.items():
        if config.trainer.validation_metric in metric_names:
            if validation_field is None:
                validation_field = feature_name
            else:
                raise ValidationError(
                    f"The validation_metric: '{config.trainer.validation_metric}' corresponds to multiple "
                    f"possible validation_fields, '{validation_field}' and '{feature_name}'. Please explicitly "
                    "specify the validation_field that should be used with the validation_metric "
                    f"'{config.trainer.validation_metric}'."
                )
    if validation_field is None:
        raise ValidationError("User-specified trainer.validation_metric is not valid for any output feature.")

    config.trainer.validation_field = validation_field


def get_feature_to_metric_names_map(
    output_features: FeatureCollection[BaseOutputFeatureConfig],
) -> Dict[str, List[str]]:
    """Returns a dict of output_feature_name -> list of metric names."""
    from ludwig.features.feature_registries import get_output_type_registry

    metrics_names = {}
    for output_feature in output_features:
        output_feature_name = output_feature.name
        output_feature_type = output_feature.type
        metrics_names[output_feature_name] = get_output_type_registry()[output_feature_type].metric_functions
    metrics_names[COMBINED] = [LOSS]
    return metrics_names


def set_derived_feature_columns_(config: ModelConfigDict):
    for feature in config.get(INPUT_FEATURES, []) + config.get(OUTPUT_FEATURES, []):
        if COLUMN not in feature:
            feature[COLUMN] = feature[NAME]
        if PROC_COLUMN not in feature:
            feature[PROC_COLUMN] = compute_feature_hash(feature)
