"""Checks that are not easily covered by marshmallow JSON schema validation like parameter interdependencies.

Assumes incoming configs are comprehensive (all parameters and defaults filled in), and has been schema-validated.
"""

from ludwig.constants import (
    AUDIO,
    BACKEND,
    BINARY,
    CATEGORY,
    COMBINER,
    IMAGE,
    IN_MEMORY,
    INPUT_FEATURES,
    MODEL_ECD,
    MODEL_GBM,
    MODEL_TYPE,
    NAME,
    NUMBER,
    OUTPUT_FEATURES,
    PREPROCESSING,
    SEQUENCE,
    SET,
    SPLIT,
    TEXT,
    TRAINER,
    TYPE,
    VECTOR,
)
from ludwig.error import ConfigValidationError
from ludwig.types import ModelConfigDict
from ludwig.utils.metric_utils import get_feature_to_metric_names_map

# Set of all sequence feature types.
SEQUENCE_OUTPUT_FEATURE_TYPES = {SEQUENCE, TEXT, SET, VECTOR}


def check_feature_names_unique(config: ModelConfigDict) -> None:
    """Checks that all feature names are unique."""
    input_features = config[INPUT_FEATURES]
    input_feature_names = {input_feature[NAME] for input_feature in input_features}

    output_features = config[OUTPUT_FEATURES]
    output_feature_names = {output_feature[NAME] for output_feature in output_features}

    if len(input_feature_names) + len(output_feature_names) != len(input_features) + len(output_features):
        raise ConfigValidationError("Feature names must be unique.")


def check_tied_features_are_valid(config: ModelConfigDict) -> None:
    """Checks that all tied features are valid."""
    input_features = config[INPUT_FEATURES]
    input_feature_names = {input_feature[NAME] for input_feature in input_features}

    for input_feature in input_features:
        if input_feature["tied"] and input_feature["tied"] not in input_feature_names:
            raise ConfigValidationError(
                f"Feature {input_feature[NAME]} is tied to feature {input_feature['tied']}, but the "
                f"'{input_feature['tied']}' feature does not exist."
            )


def check_training_runway(config: ModelConfigDict) -> None:
    """Checks that checkpoints_per_epoch and steps_per_checkpoint aren't simultaneously defined."""
    if config[MODEL_TYPE] == MODEL_ECD:
        if config[TRAINER]["checkpoints_per_epoch"] != 0 and config[TRAINER]["steps_per_checkpoint"] != 0:
            raise ConfigValidationError(
                "It is invalid to specify both trainer.checkpoints_per_epoch AND "
                "trainer.steps_per_checkpoint. Please specify one or the other, or specify neither to checkpoint/eval "
                "the model every epoch."
            )


def check_dependent_features(config: ModelConfigDict) -> None:
    """Checks that 'dependent' features map to existing output features, and no circular dependencies."""
    pass


def check_gbm_horovod_incompatibility(config: ModelConfigDict) -> None:
    """Checks that GBM model type isn't being used with the horovod backend."""
    if BACKEND not in config:
        return
    if config[MODEL_TYPE] == MODEL_GBM and config[BACKEND][TYPE] == "horovod":
        raise ConfigValidationError("Horovod backend does not support GBM models.")


def check_gbm_single_output_feature(config: ModelConfigDict) -> None:
    """GBM models only support a single output feature."""
    model_type = config[MODEL_TYPE]
    if model_type == MODEL_GBM:
        if len(config[OUTPUT_FEATURES]) != 1:
            raise ConfigValidationError("GBM models only support a single output feature.")


def check_gbm_feature_types(config: ModelConfigDict) -> None:
    """Checks that all input features for GBMs are of supported types."""
    if config[MODEL_TYPE] == MODEL_GBM:
        for input_feature in config[INPUT_FEATURES]:
            if input_feature[TYPE] not in {BINARY, CATEGORY, NUMBER}:
                raise ConfigValidationError("GBM Models currently only support Binary, Category, and Number features")


def check_ray_backend_in_memory_preprocessing(config: ModelConfigDict) -> None:
    """Checks that in memory preprocessing is used with Ray backend."""
    if BACKEND not in config:
        return

    if config[BACKEND][TYPE] == "ray" and not config[TRAINER][PREPROCESSING][IN_MEMORY]:
        raise ConfigValidationError(
            "RayBackend does not support lazy loading of data files at train time. "
            "Set preprocessing config `in_memory: True`"
        )

    for input_feature in config[INPUT_FEATURES]:
        if input_feature[TYPE] == AUDIO or input_feature[TYPE] == IMAGE:
            if not input_feature[PREPROCESSING][IN_MEMORY] and config[BACKEND][TYPE] != "ray":
                raise ConfigValidationError(
                    "RayBackend does not support lazy loading of data files at train time. "
                    f"Set preprocessing config `in_memory: True` for input feature {input_feature[NAME]}"
                )


def check_sequence_concat_combiner_requirements(config: ModelConfigDict) -> None:
    """Checks that sequence concat combiner has at least one input feature that's sequential."""
    if config[MODEL_TYPE] != MODEL_ECD:
        return
    if config[COMBINER] != "sequence_concat":
        return
    has_sequence_input = False
    for input_feature in config[INPUT_FEATURES]:
        if input_feature[TYPE] in SEQUENCE_OUTPUT_FEATURE_TYPES:
            has_sequence_input = True
            break
    if not has_sequence_input:
        raise ConfigValidationError(
            "Sequence concat combiner should only be used for at least one sequential input feature."
        )


def check_comparator_combiner_requirements(config: ModelConfigDict) -> None:
    """Checks ComparatorCombiner requirements.

    All of the feature names for entity_1 and entity_2 are valid features.
    """
    if config[MODEL_TYPE] != MODEL_ECD:
        return
    if config[COMBINER] != "comparator":
        return

    input_feature_names = {input_feature[NAME] for input_feature in config[INPUT_FEATURES]}
    for entity in ["entity_1", "entity_2"]:
        for feature_name in config[COMBINER][entity]:
            if feature_name not in input_feature_names:
                raise ConfigValidationError(
                    f"Feature {feature_name} in {entity} for the comparator combiner is not a valid "
                    "input feature name."
                )


def check_class_balance_preprocessing(config: ModelConfigDict) -> None:
    """Class balancing is only available for datasets with a single output feature."""
    if config[PREPROCESSING]["oversample_minority"] or config[PREPROCESSING]["undersample_majority"]:
        if len(config[OUTPUT_FEATURES]) != 1:
            raise ConfigValidationError("Class balancing is only available for datasets with a single output feature.")
        if config[OUTPUT_FEATURES][0][TYPE] != BINARY:
            raise ConfigValidationError("Class balancing is only supported for binary output features.")


def check_sampling_exclusivity(config: ModelConfigDict) -> None:
    """Oversample minority and undersample majority are mutually exclusive."""
    if config[PREPROCESSING]["oversample_minority"] and config[PREPROCESSING]["undersample_majority"]:
        raise ConfigValidationError(
            "Oversample minority and undersample majority are mutually exclusive. Specify only one method."
        )


def check_validation_metric_exists(config: ModelConfigDict) -> None:
    """Checks that validation fields in config.trainer are valid."""
    validation_metric_name = config[TRAINER]["validation_metric"]

    # Get all valid metrics.
    feature_to_metric_names_map = get_feature_to_metric_names_map(config[OUTPUT_FEATURES])
    all_valid_metrics = set()
    for metric_names in feature_to_metric_names_map.values():
        all_valid_metrics.update(metric_names)

    if validation_metric_name not in all_valid_metrics:
        raise ConfigValidationError(
            f"User-specified trainer.validation_metric '{validation_metric_name}' is not valid. "
            f"Available metrics are: {all_valid_metrics}"
        )


def check_splitter(config: ModelConfigDict) -> None:
    """Checks the validity of the splitter configuration."""
    from ludwig.data.split import get_splitter

    splitter = get_splitter(**config[PREPROCESSING][SPLIT])
    splitter.validate(config)
