"""Checks that are not easily covered by marshmallow JSON schema validation like parameter interdependencies."""

from abc import ABC, abstractmethod
from typing import Callable, TYPE_CHECKING

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import (
    AUDIO,
    BINARY,
    CATEGORY,
    COMBINER,
    DECODER,
    ENCODER,
    IMAGE,
    IN_MEMORY,
    INPUT_FEATURES,
    MODEL_ECD,
    MODEL_GBM,
    MODEL_TYPE,
    NAME,
    NUMBER,
    OUTPUT_FEATURES,
    SEQUENCE,
    SET,
    TEXT,
    TIMESERIES,
    TRAINER,
    TYPE,
    VECTOR,
)
from ludwig.decoders.registry import get_decoder_registry
from ludwig.encoders.registry import get_encoder_registry
from ludwig.error import ConfigValidationError
from ludwig.schema.combiners.utils import get_combiner_registry
from ludwig.schema.features.utils import input_config_registry
from ludwig.schema.optimizers import optimizer_registry
from ludwig.types import ModelConfigDict
from ludwig.utils.metric_utils import get_feature_to_metric_names_map_from_feature_collection

if TYPE_CHECKING:
    from ludwig.schema.model_config import ModelConfig

# Set of all sequence feature types.
SEQUENCE_OUTPUT_FEATURE_TYPES = {SEQUENCE, TEXT, SET, VECTOR}


class ConfigCheckRegistry:
    """A registry of configuration checks."""

    def __init__(self):
        self._registry = []

    def register(self, check_fn):
        self._registry.append(check_fn)

    def check_config(self, config: "ModelConfig") -> None:  # noqa: F821
        for check_fn in self._registry:
            check_fn(config)


_CONFIG_CHECK_REGISTRY = ConfigCheckRegistry()


def get_config_check_registry():
    """Returns the config check registry."""
    return _CONFIG_CHECK_REGISTRY


@DeveloperAPI
def register_config_check(fn) -> Callable:
    """Registers a config check function."""
    _CONFIG_CHECK_REGISTRY.register(fn)


class ConfigCheck(ABC):
    """Checks instances of comprehensive (all parameters and defaults filled in) schema-validated config."""

    @staticmethod
    @abstractmethod
    def check(config: "ModelConfig") -> None:  # noqa: F821
        """Checks config for validity."""
        raise NotImplementedError


def check_basic_required_parameters(config: ModelConfigDict) -> None:
    """Checks basic required parameters like that all features have names and types, and all types are valid."""
    model_type = config["model_type"]

    # Check input features.
    for input_feature in config[INPUT_FEATURES]:
        if NAME not in input_feature:
            raise ConfigValidationError("All input features must have a name.")
        if TYPE not in input_feature:
            raise ConfigValidationError(f"Input feature {input_feature[NAME]} must have a type.")
        if input_feature[TYPE] not in input_config_registry(model_type):
            raise ConfigValidationError(
                f"Input feature {input_feature[NAME]} uses an invalid/unsupported type "
                f"'{input_feature[TYPE]}'. Input feature types: {list(get_encoder_registry().keys())}."
            )
        if ENCODER in input_feature:
            if (
                TYPE in input_feature[ENCODER]
                and input_feature[ENCODER][TYPE] not in get_encoder_registry()[input_feature[TYPE]]
            ):
                raise ConfigValidationError(
                    f"Encoder type '{input_feature[ENCODER][TYPE]}' for input feature {input_feature[NAME]} must be "
                    f"one of: {list(get_encoder_registry()[input_feature[TYPE]].keys())}."
                )

    # Check output features.
    for output_feature in config[OUTPUT_FEATURES]:
        if NAME not in output_feature:
            raise ConfigValidationError("All output features must have a name.")
        if TYPE not in output_feature:
            raise ConfigValidationError(f"Output feature {output_feature[NAME]} must have a type.")
        if output_feature[TYPE] not in get_decoder_registry():
            raise ConfigValidationError(
                f"Output feature {output_feature[NAME]} uses an invalid/unsupported output type "
                f"'{output_feature[TYPE]}'. Supported output features: {list(get_decoder_registry().keys())}."
            )
        if DECODER in output_feature:
            if (
                TYPE in output_feature[DECODER]
                and output_feature[DECODER][TYPE] not in get_decoder_registry()[output_feature[TYPE]]
            ):
                raise ConfigValidationError(
                    f"Decoder type for output feature {output_feature[NAME]} must be one of: "
                    f"{list(get_decoder_registry()[output_feature[TYPE]].keys())}."
                )

    # Check combiners.
    if config.get(MODEL_TYPE, MODEL_ECD) == MODEL_ECD:
        if COMBINER not in config:
            return
        if TYPE not in config[COMBINER]:
            raise ConfigValidationError("Combiner must have a type.")
        if config[COMBINER][TYPE] not in get_combiner_registry():
            raise ConfigValidationError(f"Combiner type must be one of: {list(get_combiner_registry().keys())}.")

    # Check trainer.
    if TRAINER in config and config[TRAINER] is None:
        raise ConfigValidationError("Trainer cannot be None.")

    # Check optimizer.
    if TRAINER in config and "optimizer" in config[TRAINER]:
        if config[TRAINER]["optimizer"] is None:
            raise ConfigValidationError("Trainer.optimizer cannot be None.")
        if TYPE in config[TRAINER]["optimizer"]:
            if config[TRAINER]["optimizer"][TYPE] not in optimizer_registry:
                raise ConfigValidationError(
                    f"Trainer.optimizer.type must be one of: {list(optimizer_registry.keys())}."
                )


@register_config_check
def check_feature_names_unique(config: "ModelConfig") -> None:  # noqa: F821
    """Checks that all feature names are unique."""
    input_features = config.input_features
    input_feature_names = {input_feature.name for input_feature in input_features}

    output_features = config.output_features
    output_feature_names = {output_feature.name for output_feature in output_features}

    if len(input_feature_names) + len(output_feature_names) != len(input_features) + len(output_features):
        raise ConfigValidationError("Feature names must be unique.")


@register_config_check
def check_tied_features_valid(config: "ModelConfig") -> None:  # noqa: F821
    """Checks that all tied features are valid."""
    input_features = config.input_features
    input_feature_names = {input_feature.name for input_feature in input_features}

    for input_feature in input_features:
        if input_feature.tied and input_feature.tied not in input_feature_names:
            raise ConfigValidationError(
                f"Feature {input_feature.name} is tied to feature {input_feature.tied}, but the "
                f"'{input_feature.tied}' feature does not exist."
            )


@register_config_check
def check_training_runway(config: "ModelConfig") -> None:  # noqa: F821
    """Checks that checkpoints_per_epoch and steps_per_checkpoint aren't simultaneously defined."""
    if config.model_type == MODEL_ECD:
        if config.trainer.checkpoints_per_epoch != 0 and config.trainer.steps_per_checkpoint != 0:
            raise ConfigValidationError(
                "It is invalid to specify both trainer.checkpoints_per_epoch AND "
                "trainer.steps_per_checkpoint. Please specify one or the other, or specify neither to "
                "checkpoint/eval the model every epoch."
            )


@register_config_check
def check_gbm_horovod_incompatibility(config: "ModelConfig") -> None:  # noqa: F821
    """Checks that GBM model type isn't being used with the horovod backend.

    TODO(Justin): This is fine for now because we don't validate on the backend, but can be removed in the future when
    backend is schema-fied (separate schemas for ECD and GBM).
    """
    if config.backend is None:
        return
    # TODO (jeffkinnison): Revert to object access when https://github.com/ludwig-ai/ludwig/pull/3127 lands
    if config.model_type == MODEL_GBM and config.backend.get("type") == "horovod":
        raise ConfigValidationError("Horovod backend does not support GBM models.")


@register_config_check
def check_gbm_output_type(config: "ModelConfig") -> None:  # noqa: F821
    """Checks that the output features for GBMs are of supported types."""
    if config.model_type == MODEL_GBM:
        for output_feature in config.output_features:
            if output_feature.type not in {BINARY, CATEGORY, NUMBER}:
                raise ConfigValidationError(
                    "GBM Models currently only support Binary, Category, and Number output features."
                )


@register_config_check
def check_ray_backend_in_memory_preprocessing(config: "ModelConfig") -> None:  # noqa: F821
    """Checks that in memory preprocessing is used with Ray backend."""
    if config.backend is None:
        return
    if not hasattr(config.trainer, "preprocessing") or not hasattr(config.trainer.preprocessing, IN_MEMORY):
        return

    if config.backend.type == "ray" and not config.trainer.preprocessing.in_memory:
        raise ConfigValidationError(
            "RayBackend does not support lazy loading of data files at train time. "
            "Set preprocessing config `in_memory: True`"
        )

    for input_feature in config.input_features:
        if input_feature.type == AUDIO or input_feature.type == IMAGE:
            if not input_feature.preprocessing.in_memory and config.backend.type != "ray":
                raise ConfigValidationError(
                    "RayBackend does not support lazy loading of data files at train time. "
                    f"Set preprocessing config `in_memory: True` for input feature {input_feature.name}"
                )


def check_sequence_concat_combiner_requirements(config: "ModelConfig") -> None:  # noqa: F821
    """Checks that sequence concat combiner has at least one input feature that's sequential."""
    if config.model_type != MODEL_ECD:
        return
    if config.combiner != "sequence_concat":
        return
    has_sequence_input = False
    for input_feature in config.input_features:
        if input_feature.type in SEQUENCE_OUTPUT_FEATURE_TYPES:
            has_sequence_input = True
            break
    if not has_sequence_input:
        raise ConfigValidationError(
            "Sequence concat combiner should only be used for at least one sequential input feature."
        )


@register_config_check
def check_comparator_combiner_requirements(config: "ModelConfig") -> None:  # noqa: F821
    """Checks that all of the feature names for entity_1 and entity_2 are valid features."""
    if config.model_type != MODEL_ECD:
        return
    if config.combiner != "comparator":
        return

    input_feature_names = {input_feature.name for input_feature in config.input_features}
    for feature_name in config.combiner.entity_1:
        if feature_name not in input_feature_names:
            raise ConfigValidationError(
                f"Feature {feature_name} in entity_1 for the comparator combiner is not a valid " "input feature name."
            )
    for feature_name in config.combiner.entity_2:
        if feature_name not in input_feature_names:
            raise ConfigValidationError(
                f"Feature {feature_name} in entity_2 for the comparator combiner is not a valid " "input feature name."
            )


@register_config_check
def check_class_balance_preprocessing(config: "ModelConfig") -> None:  # noqa: F821
    """Class balancing is only available for datasets with a single output feature."""
    if config.preprocessing.oversample_minority or config.preprocessing.undersample_majority:
        if len(config.output_features) != 1:
            raise ConfigValidationError("Class balancing is only available for datasets with a single output feature.")
        if config.output_features[0].type != BINARY:
            raise ConfigValidationError("Class balancing is only supported for binary output features.")


@register_config_check
def check_sampling_exclusivity(config: "ModelConfig") -> None:  # noqa: F821
    """Oversample minority and undersample majority are mutually exclusive."""
    if config.preprocessing.oversample_minority and config.preprocessing.undersample_majority:
        raise ConfigValidationError(
            "Oversample minority and undersample majority are mutually exclusive. Specify only one method."
        )


@register_config_check
def check_validation_metric_exists(config: "ModelConfig") -> None:  # noqa: F821
    """Checks that the specified validation metric exists."""
    validation_metric_name = config.trainer.validation_metric

    # Get all valid metrics.
    feature_to_metric_names_map = get_feature_to_metric_names_map_from_feature_collection(config.output_features)
    all_valid_metrics = set()
    for metric_names in feature_to_metric_names_map.values():
        all_valid_metrics.update(metric_names)

    if validation_metric_name not in all_valid_metrics:
        raise ConfigValidationError(
            f"User-specified trainer.validation_metric '{validation_metric_name}' is not valid. "
            f"Available metrics are: {all_valid_metrics}"
        )


@register_config_check
def check_splitter(config: "ModelConfig") -> None:  # noqa: F821
    """Checks the validity of the splitter configuration."""
    from ludwig.data.split import get_splitter

    splitter = get_splitter(**config.preprocessing.split.to_dict())
    splitter.validate(config)


@register_config_check
def check_hf_tokenizer_requirements(config: "ModelConfig") -> None:  # noqa: F821
    """Checks that the HuggingFace tokenizer has a pretrained_model_name_or_path specified."""

    for input_feature in config.input_features:
        if input_feature.type == TEXT:
            if input_feature.preprocessing.tokenizer == "hf_tokenizer":
                if input_feature.preprocessing.pretrained_model_name_or_path is None:
                    raise ConfigValidationError(
                        "Pretrained model name or path must be specified for HuggingFace tokenizer."
                    )


@register_config_check
def check_hf_encoder_requirements(config: "ModelConfig") -> None:  # noqa: F821
    """Checks that a HuggingFace encoder has a pretrained_model_name_or_path specified."""

    for input_feature in config.input_features:
        if input_feature.type == TEXT:
            if hasattr(input_feature.encoder, "use_pretrained"):
                if input_feature.preprocessing.pretrained_model_name_or_path is None:
                    raise ConfigValidationError(
                        "Pretrained model name or path must be specified for HuggingFace encoder."
                    )


@register_config_check
def check_stacked_transformer_requirements(config: "ModelConfig") -> None:  # noqa: F821
    """Checks that the transformer encoder type correctly configures `num_heads` and `hidden_size`"""

    def is_divisible(hidden_size: int, num_heads: int) -> bool:
        """Checks that hidden_size is divisible by num_heads."""
        return hidden_size % num_heads == 0

    sequence_types = [SEQUENCE, TEXT, TIMESERIES]

    for input_feature in config.input_features:
        if_type = input_feature.type
        encoder = input_feature.encoder
        if (
            if_type in sequence_types
            and encoder.type == "transformer"
            and not is_divisible(encoder.hidden_size, encoder.num_heads)
        ):
            raise ConfigValidationError(
                f"Input feature {input_feature.name} transformer encoder requires encoder.hidden_size to be divisible "
                f"by encoder.num_heads. Found hidden_size {encoder.hidden_size} and num_heads {encoder.num_heads}."
            )
