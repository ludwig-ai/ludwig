"""PyTorch-native quantization via torchao.

Provides int4, int8, and float8 weight quantization without requiring
bitsandbytes. Uses torchao's quantize_() for in-place model quantization.

Usage in Ludwig config:
    trainer:
      quantization:
        type: int4_weight_only    # or int8_weight_only, int8_dynamic, float8

Or programmatically:
    from ludwig.utils.quantization import quantize_model
    quantize_model(model, "int4_weight_only")
"""

import logging

logger = logging.getLogger(__name__)


def quantize_model(model, quantization_type: str):
    """Apply torchao quantization to a model in-place.

    Args:
        model: PyTorch model to quantize.
        quantization_type: One of "int4_weight_only", "int8_weight_only",
            "int8_dynamic", "float8".

    Returns:
        The quantized model (modified in-place).
    """
    try:
        from torchao.quantization import (
            float8_weight_only,
            int4_weight_only,
            int8_dynamic_activation_int8_weight,
            int8_weight_only,
            quantize_,
        )
    except ImportError:
        logger.error("torchao is required for quantization. Install with: pip install torchao")
        raise

    config_map = {
        "int4_weight_only": int4_weight_only,
        "int8_weight_only": int8_weight_only,
        "int8_dynamic": int8_dynamic_activation_int8_weight,
        "float8": float8_weight_only,
    }

    if quantization_type not in config_map:
        raise ValueError(f"Unknown quantization type '{quantization_type}'. Options: {list(config_map.keys())}")

    config_fn = config_map[quantization_type]
    logger.info(f"Applying {quantization_type} quantization via torchao")
    quantize_(model, config_fn())
    logger.info("Quantization complete")

    return model
