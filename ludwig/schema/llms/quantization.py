import warnings

from transformers import BitsAndBytesConfig

from ludwig.api_annotations import DeveloperAPI
from ludwig.error import ConfigValidationError
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata import LLM_METADATA
from ludwig.schema.metadata.parameter_metadata import convert_metadata_to_json

warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    module="bitsandbytes.cuda_setup.main",
)


# Valid torchao modes. Keep in sync with `ludwig/utils/quantization.py::quantize_model`.
_TORCHAO_MODES = ["int4_weight_only", "int8_weight_only", "int8_dynamic", "float8"]


@DeveloperAPI
class QuantizationConfig(schema_utils.LudwigBaseConfig):
    backend: str = schema_utils.StringOptions(
        options=["bitsandbytes", "torchao"],
        default="bitsandbytes",
        allow_none=False,
        description=(
            "Quantization backend. 'bitsandbytes' (default) applies 4-bit / 8-bit quantization at "
            "model load time via the bitsandbytes library — the existing QLoRA fine-tuning path. "
            "'torchao' applies PyTorch-native quantization via torchao after model load, and can "
            "additionally run quantization-aware training (QAT) when `qat: true` is set."
        ),
    )

    mode: str | None = schema_utils.StringOptions(
        options=_TORCHAO_MODES,
        default=None,
        allow_none=True,
        description=(
            "torchao-only quantization mode. Ignored when `backend` is 'bitsandbytes'. "
            "'int4_weight_only' and 'int8_weight_only' quantize only the weight matrices (activations "
            "stay in fp16/bf16). 'int8_dynamic' quantizes activations to int8 dynamically per-forward. "
            "'float8' stores weights in fp8 (useful on H100+)."
        ),
    )

    qat: bool = schema_utils.Boolean(
        default=False,
        description=(
            "torchao-only. When true, inserts fake-quant observers into the model before training "
            "(QAT). The model is trained in the target low-precision numerical regime, then converted "
            "to actually-quantized weights at save time. Ignored when `backend` is 'bitsandbytes'."
        ),
    )

    bits: int = schema_utils.IntegerOptions(
        options=[4, 8],
        default=4,
        description="The quantization level to apply to weights on load.",
        parameter_metadata=LLM_METADATA["quantization"]["bits"],
    )

    llm_int8_threshold: float = schema_utils.NonNegativeFloat(
        default=6.0,
        description=(
            "This corresponds to the outlier threshold for outlier detection as described in `LLM.int8() : 8-bit "
            "Matrix Multiplication for Transformers at Scale` paper: https://arxiv.org/abs/2208.07339. Any hidden "
            "states value that is above this threshold will be considered an outlier and the operation on those "
            "values will be done in fp16. Values are usually normally distributed, that is, most values are in the "
            "range [-3.5, 3.5], but there are some exceptional systematic outliers that are very differently "
            "distributed for large models. These outliers are often in the interval [-60, -6] or [6, 60]. Int8 "
            "quantization works well for values of magnitude ~5, but beyond that, there is a significant performance "
            "penalty. A good default threshold is 6, but a lower threshold might be needed for more unstable models "
            "(small models, fine-tuning)."
        ),
    )

    llm_int8_has_fp16_weight: bool = schema_utils.Boolean(
        default=False,
        description=(
            "This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do "
            "not have to be converted back and forth for the backward pass."
        ),
    )

    bnb_4bit_compute_dtype: str = schema_utils.StringOptions(
        options=["float32", "float16", "bfloat16"],
        default="float16",
        description=(
            "This sets the computational type which might be different than the input type. For example, inputs "
            "might be fp32, but computation can be set to bf16 for speedups."
        ),
    )

    bnb_4bit_use_double_quant: bool = schema_utils.Boolean(
        default=True,
        description=(
            "This flag is used for nested quantization where the quantization constants from the first quantization "
            "are quantized again."
        ),
    )

    bnb_4bit_quant_type: str = schema_utils.StringOptions(
        options=["fp4", "nf4"],
        default="nf4",
        description="This sets the quantization data type in the bnb.nn.Linear4Bit layers.",
    )

    def to_bitsandbytes(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=self.bits == 4,
            load_in_8bit=self.bits == 8,
            llm_int8_threshold=self.llm_int8_threshold,
            llm_int8_has_fp16_weight=self.llm_int8_has_fp16_weight,
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
        )

    def validate_backend(self) -> None:
        """Cross-validate backend-specific fields.

        Called from :class:`~ludwig.schema.model_types.llm.LLMModelConfig.__post_init__`
        rather than from this class's own ``__post_init__`` because errors raised from a
        nested config's post-init get wrapped into a generic "Invalid params" message by
        the parent's :class:`NestedConfigField`, losing the specific reason.
        """
        if self.backend == "torchao":
            if self.mode is None:
                raise ConfigValidationError(
                    "`quantization.mode` is required when `quantization.backend` is 'torchao'. "
                    f"Options: {_TORCHAO_MODES}."
                )
        else:  # bitsandbytes
            if self.mode is not None:
                raise ConfigValidationError(
                    "`quantization.mode` is only supported for `backend: torchao`. Remove the "
                    "field or switch the backend to 'torchao'."
                )
            if self.qat:
                raise ConfigValidationError(
                    "`quantization.qat: true` is only supported for `backend: torchao`. "
                    "The bitsandbytes backend does not implement quantization-aware training."
                )


@DeveloperAPI
class QuantizationConfigField(schema_utils.NestedConfigField):
    def __init__(self):
        super().__init__(QuantizationConfig, default_missing=True)

    def _jsonschema_type_mapping(self):
        return {
            "oneOf": [
                {
                    "type": "null",
                    "title": "disabled",
                    "description": "Disable quantization.",
                    "parameter_metadata": convert_metadata_to_json(LLM_METADATA["quantization"]["_oneOf"]["none"]),
                },
                {
                    **schema_utils.unload_jsonschema_from_config_class(QuantizationConfig),
                    "title": "enabled",
                    "description": "Set quantization options.",
                    "parameter_metadata": convert_metadata_to_json(LLM_METADATA["quantization"]["_oneOf"]["object"]),
                },
            ],
            "title": "quantization",
            "description": "Set quantization options.",
            "parameter_metadata": convert_metadata_to_json(LLM_METADATA["quantization"]["_meta"]),
        }
