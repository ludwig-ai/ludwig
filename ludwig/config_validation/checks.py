"""Checks that are not easily covered by marshmallow JSON schema validation like parameter interdependencies."""

from abc import ABC, abstractmethod
from re import findall
from typing import Callable, TYPE_CHECKING

from transformers import AutoConfig

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import (
    AUDIO,
    BINARY,
    CATEGORY,
    IMAGE,
    IN_MEMORY,
    MIN_QUANTIZATION_BITS_FOR_MERGE_AND_UNLOAD,
    MODEL_ECD,
    MODEL_GBM,
    MODEL_LLM,
    NUMBER,
    SEQUENCE,
    SET,
    TEXT,
    TIMESERIES,
    VECTOR,
)
from ludwig.error import ConfigValidationError
from ludwig.utils.metric_utils import get_feature_to_metric_names_map_from_feature_collection
from ludwig.utils.misc_utils import merge_dict

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
    if config.combiner.type != "comparator":
        return

    input_feature_names = [input_feature.name for input_feature in config.input_features]
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

    if sorted(config.combiner.entity_1 + config.combiner.entity_2) != sorted(input_feature_names):
        raise ConfigValidationError("Not all input features are present as entities in the comparator combiner.")


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


@register_config_check
def check_hyperopt_search_algorithm_dependencies_installed(config: "ModelConfig") -> None:  # noqa: F821
    """Check that the hyperopt search algorithm dependencies are installed."""
    if config.hyperopt is None:
        return

    try:
        config.hyperopt.search_alg.dependencies_installed()
    except ImportError as e:
        raise ConfigValidationError(e.msg)


@register_config_check
def check_hyperopt_scheduler_dependencies_installed(config: "ModelConfig") -> None:  # noqa: F821
    """Check that the hyperopt scheduler dependencies are installed."""
    if config.hyperopt is None:
        return

    try:
        config.hyperopt.executor.scheduler.dependencies_installed()
    except ImportError as e:
        raise ConfigValidationError(e.msg)


@register_config_check
def check_tagger_decoder_requirements(config: "ModelConfig") -> None:  # noqa: F821
    """Checks that the tagger decoder has at least one sequence, text or timeseries input feature where the
    encoder's reduce_output will produce a 3D shaped output from the combiner."""
    # Check if there is a text or sequence output feature using a tagger decoder
    output_feature_with_tagger_decoder = False
    for output_feature in config.output_features:
        if output_feature.type in {TEXT, SEQUENCE} and output_feature.decoder.type == "tagger":
            output_feature_with_tagger_decoder = True

    if not output_feature_with_tagger_decoder:
        return

    # Check that there is at least one sequence, text or timeseries input feature that doesn't reduce the
    # output of the encoder.
    has_sequence_feature = False
    for input_feature in config.input_features:
        if input_feature.type in {SEQUENCE, TEXT, TIMESERIES}:
            has_sequence_feature = True
            if input_feature.encoder.reduce_output is None:
                return

    if not has_sequence_feature:
        raise ConfigValidationError("Tagger decoder requires at least one text, sequence or timeseries input feature.")
    else:
        raise ConfigValidationError(
            "Tagger decoder requires at least one of the text, sequence or timeseries input feature encoders to have "
            "`reduce_output` set to `None`."
        )


@register_config_check
def check_hyperopt_parameter_dicts(config: "ModelConfig") -> None:  # noqa: F821
    """Checks for hyperopt parameter dicts against their config objects."""
    if config.hyperopt is None:
        return

    from ludwig.schema.hyperopt.utils import get_parameter_cls, parameter_config_registry  # noqa: F401

    for parameter, space in config.hyperopt.parameters.items():
        # skip nested hyperopt parameters
        if parameter != ".":
            parameter_attribute_path = parameter.split(".")
            passed = False

            for root in [config, config.input_features, config.output_features]:
                current = root
                for p in parameter_attribute_path:
                    try:
                        current = current.__getattribute__(p)
                        if p == parameter_attribute_path[-1]:
                            passed = True
                    except AttributeError:
                        break
                if passed:
                    break

            if not passed:
                raise ConfigValidationError(
                    f"The supplied hyperopt parameter {parameter} is not a valid config field. Check the Ludwig "
                    "docs for the list of valid parameters."
                )

            try:
                space_cls = get_parameter_cls(space["space"])
                space_cls.from_dict(space)
            except KeyError:
                space_types = ", ".join(parameter_config_registry.keys())
                raise ConfigValidationError(
                    f"Invalid hyperopt parameter space requested for `hyperopt.parameters.{parameter}`. Valid spaces "
                    f"are {space_types}."
                )


@register_config_check
def check_concat_combiner_requirements(config: "ModelConfig") -> None:  # noqa: F821
    """Checks that if the concat combiner receives a mixture of sequence and non-sequence features, that all
    sequence features are configured with reduce_output to be 2D tensors."""
    if config.model_type != MODEL_ECD:
        return
    if config.combiner.type != "concat":
        return

    has_unreduced_sequence_feature = False
    has_non_sequence_feature = False
    for input_feature in config.input_features:
        if (
            input_feature.type in {SEQUENCE, TEXT, TIMESERIES}
            and hasattr(input_feature.encoder, "reduce_output")
            and input_feature.encoder.reduce_output is None
        ):
            has_unreduced_sequence_feature = True
        else:
            has_non_sequence_feature = True

    if has_unreduced_sequence_feature and has_non_sequence_feature:
        raise ConfigValidationError(
            "The concat combiner cannot receive a mix of unreduced sequence features (3D) and non-sequence features "
            "(2D). Options: 1) Set reduce_output in sequence feature encoders to a value other than None to ensure 2D "
            "encoder outputs, 2) Choose a different combiner like `sequence_concat` which can handle a mix of 2D and "
            "3D encoder output shapes, or 3) Remove features to ensure that output shapes from all encoders are the "
            "same dimension (all 2D or all 3D)."
        )


@register_config_check
def check_hyperopt_nested_parameter_dicts(config: "ModelConfig") -> None:  # noqa: F821
    """Checks that all nested parameters in a hyperopt config exist."""
    if config.hyperopt is None or "." not in config.hyperopt.parameters:
        return

    from ludwig.schema.hyperopt.utils import get_parameter_cls  # noqa: F401
    from ludwig.schema.model_types.base import ModelConfig

    space = config.hyperopt.parameters["."]

    # Build the config that would be produced by each parameter dict to validate subsections that may be in
    config_dict = config.to_dict()
    del config_dict["hyperopt"]
    for category in space["categories"]:
        for i, k in enumerate(category.keys()):
            try:
                config.__getattribute__(k)
            except AttributeError:
                raise ConfigValidationError(f"Invalid config block {k} in nested hyperopt parameter dict {i}: {space}.")

        category_dict = merge_dict(config_dict, category)
        try:
            ModelConfig.from_dict(category_dict)
        except ConfigValidationError as e:
            raise ConfigValidationError(f"Invalid config in hyperopt nested parameter config: {category}. {e.message}")

    try:
        space_cls = get_parameter_cls("choice")
        space_cls.from_dict(space)
    except KeyError:
        raise ConfigValidationError(
            f"Nested hyperparameter search spaces must be of type 'choice'. Requested space type: {space['space']}"
        )


@register_config_check
def check_llm_exactly_one_input_text_feature(config: "ModelConfig"):  # noqa: F821
    if config.model_type != MODEL_LLM:
        return

    if len(config.input_features) == 1 and config.input_features[0].type == TEXT:
        return
    else:
        raise ConfigValidationError("LLM requires exactly one text input feature.")


@register_config_check
def check_llm_finetuning_output_feature_config(config: "ModelConfig"):  # noqa: F821
    """Checks that the output feature config for LLM finetuning is valid."""
    if config.model_type != MODEL_LLM:
        return

    if config.trainer.type != "finetune":
        return

    if config.output_features[0].type != TEXT:
        raise ConfigValidationError(
            "LLM finetuning requires the output feature to be a text feature. If you are trying to use a different "
            "output feature type such as category or binary, please change the output feature type to text."
        )


@register_config_check
def check_llm_finetuning_trainer_config(config: "ModelConfig"):  # noqa: F821
    """Ensures that trainer type is finetune if adapter is not None."""
    if config.model_type != MODEL_LLM:
        return

    if (
        config.trainer.type == "none"
        and config.adapter is not None
        and config.adapter.pretrained_adapter_weights is not None
    ):
        # If performing zero-shot, we must specify pretrained adapter weights
        return

    if config.adapter is not None and config.trainer.type != "finetune":
        raise ConfigValidationError("LLM finetuning requires trainer type to be finetune.")


@register_config_check
def check_llm_finetuning_backend_config(config: "ModelConfig"):  # noqa: F821
    """Checks that the LLM finetuning using Ray is configured correctly.

    DDP strategy is not supported for LLM finetuning because it leads to OOMs since the model is large and DDP strategy
    requires a copy of the model on each GPU.
    """
    if config.model_type != MODEL_LLM:
        return

    # LLM finetuning is only supported by the finetune trainer type
    if (
        config.trainer.type != "finetune"
        and config.adapter is not None
        and config.adapter.pretrained_adapter_weights is not None
    ):
        return

    # Using local backend, so skip the checks below
    if not hasattr(config.backend, "type"):
        return

    backend = config.backend
    if not hasattr(backend.trainer, "strategy") or backend.trainer.strategy != "deepspeed":
        raise ConfigValidationError("LLM finetuning with Ray requires the DeepSpeed strategy.")

    # Deepspeed requires GPU
    if not backend.trainer.use_gpu or backend.trainer.resources_per_worker.GPU < 1:
        raise ConfigValidationError("LLM finetuning with DeepSpeed requires GPU.")


@register_config_check
def check_llm_finetuning_adalora_config(config: "ModelConfig"):
    """Checks that the adalora adapter is configured correctly.

    We check against PEFT's predefined target module list for ADALORA to see if this target_modules is present there. If
    not, AdaloraModel will run into issues downstream.
    """
    if config.model_type != MODEL_LLM:
        return

    if not config.adapter:
        return

    if config.adapter.type != "adalora":
        return

    from peft.utils import TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING

    model_config = _get_llm_model_config(config.base_model)
    if model_config.model_type not in TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING:
        raise ConfigValidationError(
            f"Adalora adapter is not supported for {model_config.model_type} model. "
            f"Supported model types are: {list(TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING.keys())}. "
            "If you know the target modules for your model, please specify them in the config through the "
            "`target_modules` key."
        )


@register_config_check
def check_llm_finetuning_adaption_prompt_parameters(config: "ModelConfig"):
    """Checks that the adaption_prompt adapter is configured correctly.

    Adaption prompt is only supported for Llama models.
    """
    if config.model_type != MODEL_LLM:
        return

    if not config.adapter:
        return

    if config.adapter.type != "adaption_prompt":
        return

    from peft.tuners.adaption_prompt import TRANSFORMERS_MODEL_CONFIG

    # Adaption Config is currently only supported for Llama model types
    model_config = _get_llm_model_config(config.base_model)
    if model_config.model_type not in TRANSFORMERS_MODEL_CONFIG:
        raise ConfigValidationError(
            f"Adaption prompt adapter is not supported for {model_config.model_type} model. "
            f"Supported model types are: {list(TRANSFORMERS_MODEL_CONFIG.keys())}."
        )


def _get_llm_model_config(model_name: str) -> AutoConfig:
    """Returns the LLM model config."""
    return AutoConfig.from_pretrained(model_name)


# TODO(geoffrey, arnav): uncomment this when we have reconciled the config with the backend kwarg in api.py
# @register_config_check
def check_llm_quantization_backend_incompatibility(config: "ModelConfig") -> None:  # noqa: F821
    """Checks that LLM model type with quantization uses the local backend."""
    if config.model_type != MODEL_LLM:
        return

    if config.quantization is None:
        return

    backend_type = None
    if config.backend:
        backend_type = config.backend.get("type", None)

    # If backend was explicitly set to Ray, then we need to raise an error
    if backend_type == "ray":
        raise ConfigValidationError(f"LLM with quantization requires the 'local' backend, found: '{backend_type}'")

    # If the backend is not explicitly set, then we need to check if a Ray process is running
    # If a Ray process is running, then we need to raise an error because the backend will be set to Ray
    if config.backend is None:
        try:
            # May not be installed, so we need to catch the ImportError
            import ray

            if ray.is_initialized():
                raise ConfigValidationError(
                    "LLM with quantization requires the 'local' backend, but backend will be set "
                    "to Ray since Ray is already running locally."
                )
        except ImportError:
            pass


@register_config_check
def check_qlora_requirements(config: "ModelConfig") -> None:  # noqa: F821
    """Checks that all the necessary settings are in place for QLoRA."""
    if config.model_type != MODEL_LLM or config.trainer.type == "none":
        return

    if config.quantization and (not config.adapter or config.adapter.type != "lora"):
        raise ConfigValidationError("Fine-tuning and LLM with quantization requires using the 'lora' adapter")


@register_config_check
def check_qlora_merge_and_unload_compatibility(config: "ModelConfig") -> None:  # noqa: F821
    """Checks that model.merge_and_unload() is supported by underlying model.save_pretrained() when merging QLoRA
    layers."""
    if config.model_type != MODEL_LLM or config.trainer.type == "none":
        return

    if not (
        config.adapter
        and config.adapter.type in ["lora", "adalora"]
        and config.adapter.postprocessor
        and config.adapter.postprocessor.merge_adapter_into_base_model
        and config.quantization
    ):
        return

    if config.quantization.bits < MIN_QUANTIZATION_BITS_FOR_MERGE_AND_UNLOAD:
        raise ConfigValidationError(
            f"""This operation will entail merging LoRA layers on a {config.quantization.bits}-bit \
quantized model.  Calling "save_pretrained()" on that model is currently unsupported.  If you want to merge the LoRA \
adapter weights into the base model, you need to use 8-bit quantization or do non-quantized based training by removing \
the quantization section from your Ludwig configuration."""
        )


@register_config_check
def check_prompt_requirements(config: "ModelConfig") -> None:  # noqa: F821
    """Checks that prompt's template and task properties are valid, according to the description on the schema."""
    if config.model_type != MODEL_LLM:
        return

    # TODO: `prompt` by default should be set to null, not a default dict:
    # # If no prompt is provided, no validation necessary:
    # if not config.prompt:
    #     return
    from ludwig.schema.llms.prompt import PromptConfig, RetrievalConfig

    if config.prompt == PromptConfig():
        return

    template = config.prompt.template
    task = config.prompt.task
    retrieval = config.prompt.retrieval

    # If template is NOT provided, then task is required for zero/few shot learning:
    if not template and not task:
        raise ConfigValidationError("A prompt task is required if no template is provided!")

    template_refs = set(findall(r"\{(.*?)\}", template)) if isinstance(template, str) else set()

    # If a template IS provided (i.e. we are not doing a built-in zero/few-shot learning), then...
    if template:
        # If task is also provided, the template must contain it:
        if task and "__task__" not in template_refs:
            raise ConfigValidationError(
                "When providing a task, you must make sure that the task keyword `{__task__} is "
                "present somewhere in the template string!"
            )

        # If retrieval is also provided, the template must reference it:
        # TODO: retrieval by default should be set to null, not a default dict:
        if retrieval and retrieval != RetrievalConfig() and "__context__" not in template_refs:
            raise ConfigValidationError(
                "When providing a retrieval config, you must make sure that the task keyword `{__context__}` is "
                "present somewhere in the template string!"
            )

        # Otherwise, the template should at least contain the sample keyword or some input column:
        # TODO: len(template_refs) is a hacky attempt to check that there are references to *something* in the
        # string. The proper validation is to check the references against the features in the user's dataset - but we
        # do not have access to the dataset in this code path right now.
        if not task:
            if len(template_refs) == 0 and "__sample__" not in template_refs:
                raise ConfigValidationError(
                    "A template must contain at least one reference to a column or the sample keyword {__sample__} for "
                    "a JSON-serialized representation of non-output feature columns."
                )
