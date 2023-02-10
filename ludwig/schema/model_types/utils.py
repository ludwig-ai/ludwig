import copy
import logging
import sys
import warnings
from typing import Any, Dict, List, Mapping, Set, TYPE_CHECKING

from marshmallow import ValidationError

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import (
    COMBINED,
    DECODER,
    DEFAULTS,
    ENCODER,
    GRID_SEARCH,
    INPUT_FEATURES,
    LOSS,
    OUTPUT_FEATURES,
    PARAMETERS,
    PREPROCESSING,
    SPACE,
    TYPE,
)
from ludwig.features.feature_utils import compute_feature_hash
from ludwig.schema.encoders.utils import get_encoder_cls
from ludwig.schema.features.utils import input_config_registry, output_config_registry
from ludwig.schema.hyperopt.scheduler import BaseHyperbandSchedulerConfig
from ludwig.schema.trainer import ECDTrainerConfig
from ludwig.types import HyperoptConfigDict, ModelConfigDict
from ludwig.utils.misc_utils import merge_dict

if TYPE_CHECKING:
    from ludwig.schema.model_types.base import ModelConfig


logger = logging.getLogger(__name__)


@DeveloperAPI
def merge_with_defaults(config_dict: ModelConfigDict) -> ModelConfigDict:
    # Recursive merge of the features, except that if we find a dictionary containing
    # an explicit "type" key, we ignore defaults if they don't match.
    defaults = config_dict.get(DEFAULTS)
    if not defaults:
        return config_dict

    config_dict = copy.deepcopy(config_dict)
    _merge_features_(config_dict.get(INPUT_FEATURES, []), defaults, {DECODER, LOSS})
    _merge_features_(config_dict.get(OUTPUT_FEATURES, []), defaults, {ENCODER, PREPROCESSING})
    return config_dict


def _merge_features_(features: List[Dict[str, Any]], defaults: Dict[str, Any], exclude_keys: Set[str]):
    for feature in features:
        ftype = feature.get(TYPE)
        if not ftype:
            continue

        default_feature = defaults.get(ftype, {})
        merged_feature = _merge_dict_with_types(default_feature, feature, exclude_keys)

        # In-place replacement of the old feature with the new
        feature.clear()
        feature.update(merged_feature)


def _merge_dict_with_types(dct: Dict[str, Any], merge_dct: Dict[str, Any], exclude_keys: Set[str]) -> Dict[str, Any]:
    dct = copy.deepcopy(dct)
    dct = {k: v for k, v in dct.items() if k not in exclude_keys}

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
            dct[k] = _merge_dict_with_types(dct[k], v, exclude_keys)
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

    # First set the validation field so we know what feature we're validating on
    if not config.trainer.validation_field:
        if config.trainer.validation_metric is None or config.trainer.validation_metric == LOSS:
            # Loss is valid for all features.
            config.trainer.validation_field = config.output_features[0].name
        else:
            # Determine the proper validation field for the user, like if the user specifies "accuracy" but forgets to
            # change the validation field from "combined" to the name of the feature that produces accuracy metrics.
            from ludwig.utils.metric_utils import get_feature_to_metric_names_map

            feature_to_metric_names_map = get_feature_to_metric_names_map(config.output_features.to_list())
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

    # If the field is combined, then make sure the metric is loss and then return
    if config.trainer.validation_field == COMBINED:
        # Only loss is supported for combined
        if not config.trainer.validation_metric:
            config.trainer.validation_metric = LOSS
        elif config.trainer.validation_metric != LOSS:
            raise ValidationError(
                f"Must set validation_metric=loss when validation_field=combined, "
                f"found validation_metric={config.trainer.validation_metric}"
            )
        return

    # Field is not combined, so use the default validation metric for the single feature
    validation_features = [f for f in config.output_features if f.name == config.trainer.validation_field]
    if len(validation_features) > 1:
        raise ValidationError(
            f"Found more than one feature matching validation field: {config.trainer.validation_field}"
        )
    if len(validation_features) == 0:
        raise ValidationError(f"No output feature found matching validation field: {config.trainer.validation_field}")

    validation_feature = validation_features[0]
    if not config.trainer.validation_metric:
        # The user has not explicitly set any validation fields.
        # Default to using the first output feature's default validation metric.
        out_type = validation_feature.type
        config.trainer.validation_metric = output_config_registry[out_type].default_validation_metric


def set_derived_feature_columns_(config_obj: "ModelConfig"):
    """Assigns column and proc_column values to features that do not have them set.

    Proc_column is set to a hash of the feature's preprocessing configuration.
    """
    for feature in config_obj.input_features:
        if feature.column is None:
            feature.column = feature.name
        if feature.proc_column is None:
            feature.proc_column = compute_feature_hash(feature.to_dict())

    for feature in config_obj.output_features:
        if feature.column is None:
            feature.column = feature.name
        if feature.proc_column is None:
            feature.proc_column = compute_feature_hash(feature.to_dict())


def set_hyperopt_defaults_(config: "ModelConfig"):
    """This function was migrated from defaults.py with the intention of setting some hyperopt defaults while the
    hyperopt section of the config object is not fully complete.

    Returns:
        None -> modifies trainer and hyperopt sections
    """
    if not config.hyperopt:
        return

    # Set default num_samples based on search space if not set by user
    if config.hyperopt.executor.num_samples is None:
        _contains_grid_search_params = contains_grid_search_parameters(config.hyperopt.to_dict())
        if _contains_grid_search_params:
            logger.info(
                "Setting hyperopt num_samples to 1 to prevent duplicate trials from being run. Duplicate trials are"
                " created when there are hyperopt parameters that use the `grid_search` search space.",
            )
            config.hyperopt.executor.num_samples = 1
        else:
            logger.info("Setting hyperopt num_samples to 10.")
            config.hyperopt.executor.num_samples = 10

    scheduler = config.hyperopt.executor.scheduler
    if scheduler.type == "fifo":
        # FIFO scheduler has no constraints
        return

    # Disable early stopping when using a scheduler. We achieve this by setting the parameter
    # to -1, which ensures the condition to apply early stopping is never met.
    early_stop = config.trainer.early_stop
    if early_stop is not None and early_stop != -1:
        warnings.warn("Can't utilize `early_stop` while using a hyperopt scheduler. Setting early stop to -1.")
    config.trainer.early_stop = -1

    if isinstance(config.trainer, ECDTrainerConfig) and isinstance(scheduler, BaseHyperbandSchedulerConfig):
        # TODO(travis): explore similar contraints for GBMs, which don't have epochs
        max_t = scheduler.max_t
        time_attr = scheduler.time_attr
        epochs = config.trainer.epochs
        if max_t is not None:
            if time_attr == "time_total_s":
                if epochs is None:
                    # Continue training until time limit hit
                    config.trainer.epochs = sys.maxsize
                # else continue training until either time or trainer epochs limit hit
            elif epochs is not None and epochs != max_t:
                raise ValueError(
                    "Cannot set trainer `epochs` when using hyperopt scheduler w/different training_iteration `max_t`. "
                    "Unset one of these parameters in your config or make sure their values match."
                )
            else:
                # Run trainer until scheduler epochs limit hit
                config.trainer.epochs = max_t
        elif epochs is not None:
            scheduler.max_t = epochs  # run scheduler until trainer epochs limit hit


@DeveloperAPI
def contains_grid_search_parameters(hyperopt_config: HyperoptConfigDict) -> bool:
    """Returns True if any hyperopt parameter in the config is using the grid_search space."""
    for _, param_info in hyperopt_config[PARAMETERS].items():
        if param_info.get(SPACE, None) == GRID_SEARCH:
            return True
    return False
