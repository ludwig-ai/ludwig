"""PyTorch-native quantization via torchao.

Provides int4, int8, and float8 weight quantization without requiring
bitsandbytes. Exposes three operations:

* :func:`quantize_model` — post-training quantization (PTQ). Call on a trained
  fp16/bf16/fp32 model to produce a quantized model.
* :func:`prepare_qat_model` — insert fake-quant observers before training
  (quantization-aware training). Train as usual in the target low-precision
  regime.
* :func:`convert_qat_model` — after QAT training, convert the observed model
  to actually-quantized weights for inference.

Usage in Ludwig config (Post-training quantization / PTQ):

    quantization:
      backend: torchao
      mode: int4_weight_only    # or int8_weight_only, int8_dynamic, float8

Usage in Ludwig config (Quantization-aware training / QAT):

    quantization:
      backend: torchao
      mode: int4_weight_only
      qat: true
"""

import logging

logger = logging.getLogger(__name__)


# Canonical mode list. Keep in sync with ``_TORCHAO_MODES`` in ``ludwig/schema/llms/quantization.py``.
_VALID_MODES = ("int4_weight_only", "int8_weight_only", "int8_dynamic", "float8")


def _import_torchao_ptq():
    """Import torchao's PTQ API or raise a clear error if torchao is missing."""
    try:
        from torchao.quantization import (
            float8_weight_only,
            int4_weight_only,
            int8_dynamic_activation_int8_weight,
            int8_weight_only,
            quantize_,
        )
    except ImportError as exc:
        raise ImportError(
            "torchao is required for the 'torchao' quantization backend. "
            "Install with: pip install 'ludwig[llm]' or pip install torchao."
        ) from exc

    return quantize_, {
        "int4_weight_only": int4_weight_only,
        "int8_weight_only": int8_weight_only,
        "int8_dynamic": int8_dynamic_activation_int8_weight,
        "float8": float8_weight_only,
    }


def _import_torchao_qat():
    """Import torchao's QAT API or raise a clear error if torchao is missing.

    The QAT submodule has moved between torchao releases, so we try the modern namespace first and fall back to the
    pre-0.9 location.
    """
    try:
        from torchao.quantization.qat import (
            FakeQuantizeConfig,
            from_intx_quantization_aware_training,
            intx_quantization_aware_training,
        )
    except ImportError:
        try:
            from torchao.quantization.prototype.qat import (  # type: ignore[no-redef]
                FakeQuantizeConfig,
                from_intx_quantization_aware_training,
                intx_quantization_aware_training,
            )
        except ImportError as exc:
            raise ImportError(
                "torchao QAT is required for `quantization.qat: true`. "
                "Install torchao >= 0.9 with: pip install 'ludwig[llm]'."
            ) from exc

    return intx_quantization_aware_training, from_intx_quantization_aware_training, FakeQuantizeConfig


def _validate_mode(mode: str) -> None:
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown quantization mode '{mode}'. Options: {list(_VALID_MODES)}")


def quantize_model(model, quantization_type: str):
    """Apply torchao post-training quantization (PTQ) to ``model`` in-place.

    Args:
        model: PyTorch model to quantize.
        quantization_type: One of :data:`_VALID_MODES`.

    Returns:
        The quantized model (modified in-place).
    """
    _validate_mode(quantization_type)
    quantize_, config_map = _import_torchao_ptq()
    logger.info("Applying %s quantization via torchao (PTQ)", quantization_type)
    quantize_(model, config_map[quantization_type]())
    logger.info("Quantization complete")
    return model


def _qat_bit_width(mode: str) -> int:
    if mode.startswith("int4"):
        return 4
    if mode.startswith("int8"):
        return 8
    # float8 isn't a standard IntXQAT path; reject explicitly rather than silently quantize wrong.
    raise ValueError(
        f"QAT is not supported for quantization mode '{mode}'. "
        f"Use PTQ (qat: false) for 'float8', or pick an int4/int8 mode for QAT."
    )


def prepare_qat_model(model, quantization_type: str, group_size: int = 32):
    """Insert fake-quant observers into ``model`` for quantization-aware training.

    Call this once on the unquantized model **before** training starts. Ludwig's
    ``LLM.prepare_for_training`` handles this automatically when
    ``quantization.qat: true`` is set in the user's config.

    After training completes, call :func:`convert_qat_model` to convert the
    observed model to actually-quantized weights for inference / export.

    Args:
        model: PyTorch model to prepare for QAT.
        quantization_type: One of :data:`_VALID_MODES` (int4/int8 only; float8
            is PTQ-only in torchao).
        group_size: Group size for per-group weight quantization. 32 is a sane
            default for small / medium LMs.

    Returns:
        The model modified in-place with fake-quant observers inserted.
    """
    _validate_mode(quantization_type)
    intx_qat, _, FakeQuantizeConfig = _import_torchao_qat()

    bit_width = _qat_bit_width(quantization_type)
    weight_cfg = FakeQuantizeConfig(dtype=f"int{bit_width}", group_size=group_size)
    activation_cfg = None
    if quantization_type == "int8_dynamic":
        activation_cfg = FakeQuantizeConfig(dtype="int8", is_dynamic=True)

    logger.info("Preparing model for QAT via torchao (%s, group_size=%d)", quantization_type, group_size)
    # intx_quantization_aware_training returns a transform that quantize_ applies.
    from torchao.quantization import quantize_

    quantize_(model, intx_qat(activation_cfg, weight_cfg))
    return model


def convert_qat_model(model, quantization_type: str):
    """Convert a QAT-prepared model to actually-quantized weights.

    Call after QAT training finishes — typically from the LLM save / export path.
    Undoes the fake-quant observers inserted by :func:`prepare_qat_model` and replaces
    them with real quantized tensors matching the original ``quantization_type``.

    Args:
        model: Model previously prepared via :func:`prepare_qat_model`.
        quantization_type: The same mode that was passed to ``prepare_qat_model``.

    Returns:
        The model modified in-place.
    """
    _validate_mode(quantization_type)
    _, from_intx_qat, _ = _import_torchao_qat()
    quantize_, config_map = _import_torchao_ptq()

    logger.info("Converting QAT-prepared model to %s quantized weights", quantization_type)
    # First strip the fake-quant observers back to plain Linear...
    quantize_(model, from_intx_qat())
    # ...then apply real PTQ to get the final quantized tensors.
    quantize_(model, config_map[quantization_type]())
    return model
