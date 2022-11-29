"""Validation checks that are not easily covered by marshmallow schemas like parameter interdependencies.

As these are built out, these auxiliary validations can be gradually removed.
"""
from typing import Dict, Any

from ludwig.constants import OUTPUT_FEATURES, TRAINER, COMBINED


def check_feature_names_unique(config: Dict[str, Any]) -> None:
    """Checks that all feature names are unique."""
    pass


def check_tied_features_are_valid(config: Dict[str, Any]) -> None:
    """Checks that all 'tied' parameters map to existing input feature names."""
    pass


def check_dependent_features(config: Dict[str, Any]) -> None:
    """Checks that 'dependent' features map to existing output features, and no circular dependencies."""


def check_training_runway(config: Dict[str, Any]) -> None:
    """Checks that checkpoints_per_epoch and steps_per_checkpoint aren't simultaneously defined."""
    pass


def check_gbm_horovod_incompatibility(config: Dict[str, Any]) -> None:
    """Checks that GBM model type isn't being used with the horovod backend."""
    pass


def check_gbm_feature_types(config: Dict[str, Any]) -> None:
    """Checks that only tabular features are used with GBM models."""
    pass


def check_ray_backend_in_memory_preprocessing(config: Dict[str, Any]) -> None:
    """Checks if it's a ray backend, then feature[preprocessing][in_memory] must be true."""
    pass


def check_sequence_concat_combiner_requirements(config: Dict[str, Any]) -> None:
    """Checks sequence concat combiner requirements.

    At least one of the input features should be a sequence feature.
    """
    pass


def check_tabtransformer_combiner_requirements(config: Dict[str, Any]) -> None:
    """Checks TabTransformer requirements.

    reduce_output cannot be None.
    """
    pass


def check_comparator_combiner_requirements(config: Dict[str, Any]) -> None:
    """Checks ComparatorCombiner requirements.

    All of the feature names for entity_1 and entity_2 are valid features.
    """
    pass


def check_class_balance_preprocessing(config: Dict[str, Any]) -> None:
    """Class balancing is only available for datasets with a single output feature."""
    pass


def check_sampling_exclusivity(config: Dict[str, Any]) -> None:
    """Oversample minority and undersample majority are mutually exclusive."""
    pass


def check_hyperopt_search_space(config: Dict[str, Any]) -> None:
    """Check that all hyperopt parameters search spaces are valid."""
    pass


def check_hyperopt_metric_targets(config: Dict[str, Any]) -> None:
    """Check that hyperopt metric targets are valid."""
    pass


def check_gbm_single_output_feature(config: Dict[str, Any]) -> None:
    """GBM models only support a single output feature."""
    pass


def get_metric_names(output_features: List[Dict]) -> Dict[str, List[str]]:
    """Returns a dict of output_feature_name -> list of metric names."""
    metrics_names = {}
    for output_feature in output_features:
        output_feature_name = output_feature[NAME]
        output_feature_type = output_feature[TYPE]
        for metric in output_feature.metric_functions:
            metrics = metrics_names.get(output_feature_name, [])
            metrics.append(metric)
            metrics_names[output_feature_name] = metrics
    metrics_names[COMBINED] = [LOSS]
    return metrics_names


def check_validation_metrics_are_valid(config: Dict[str, Any]) -> None:
    """Checks that validation fields in cofnig.trainer are valid."""
    output_features = config[OUTPUT_FEATURES]
    metrics_names = get_metric_names_from_list(output_features)

    validation_field = config[TRAINER]["validation_field"]
    validation_metric = config[TRAINER]["validation_metric"]

    # Check validation_field.
    if validation_field not in output_features and validation_field != COMBINED:
        raise ValueError(
            f"The specified trainer.validation_field '{validation_field}' is not valid."
            f'Available validation fields are: {list(output_features.keys()) + ["combined"]}'
        )

    # Check validation_metric.
    valid_validation_metric = validation_metric in metrics_names[validation_field]
    if not valid_validation_metric:
        raise ValueError(
            f"The specified trainer.validation_metric '{validation_metric}' is not valid for the"
            f"trainer.validation_field '{validation_field}'. "
            f"Available (validation_field, validation_metric) pairs are {metrics_names}"
        )
