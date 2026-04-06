"""Auto-generate Optuna search spaces from Ludwig Pydantic config field constraints.

Inspects a Ludwig config schema (Pydantic model) and generates an Optuna-compatible
search space dict based on field types, defaults, and constraints.

Usage:
    from ludwig.hyperopt.search_space_generator import generate_search_space

    space = generate_search_space(ECDTrainerConfig, fields=["learning_rate", "batch_size", "dropout"])
    # Returns: {"trainer.learning_rate": {"space": "loguniform", "lower": 1e-5, "upper": 0.1}, ...}
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def generate_search_space(
    config_class,
    fields: list[str] | None = None,
    prefix: str = "",
) -> dict[str, dict[str, Any]]:
    """Generate Optuna search space from Pydantic config field constraints.

    Inspects field types, defaults, and metadata to create appropriate search spaces:
    - float fields with range constraints -> uniform or loguniform
    - int fields with range constraints -> int (randint)
    - str fields with options -> categorical
    - bool fields -> categorical [True, False]

    Args:
        config_class: A Ludwig config class (LudwigBaseConfig subclass).
        fields: List of field names to include. None for all tunable fields.
        prefix: Prefix for parameter names (e.g., "trainer." for trainer params).

    Returns:
        Dict mapping parameter paths to Optuna search space definitions.
    """
    space = {}

    for field_name, field_info in config_class.model_fields.items():
        if fields is not None and field_name not in fields:
            continue

        param_name = f"{prefix}{field_name}" if prefix else field_name
        field_type = field_info.annotation
        default = field_info.default

        # Skip non-tunable fields
        if field_name in ("type", "model_type"):
            continue

        # Extract constraints from metadata
        metadata = field_info.metadata or []
        min_val = None
        max_val = None

        for meta in metadata:
            if hasattr(meta, "ge"):
                min_val = meta.ge
            if hasattr(meta, "gt"):
                min_val = meta.gt
            if hasattr(meta, "le"):
                max_val = meta.le
            if hasattr(meta, "lt"):
                max_val = meta.lt

        # Determine search space type from field type and constraints
        type_str = str(field_type) if field_type else ""

        if "float" in type_str:
            if default and isinstance(default, (int, float)) and default > 0 and default < 0.01:
                # Small float default suggests log-uniform (learning rates, etc.)
                space[param_name] = {
                    "space": "loguniform",
                    "lower": (min_val or default / 10) if default else 1e-6,
                    "upper": (max_val or default * 10) if default else 1.0,
                }
            elif min_val is not None or max_val is not None:
                space[param_name] = {
                    "space": "uniform",
                    "lower": min_val if min_val is not None else 0.0,
                    "upper": max_val if max_val is not None else 1.0,
                }
        elif "int" in type_str:
            if min_val is not None or max_val is not None:
                space[param_name] = {
                    "space": "int",
                    "lower": int(min_val) if min_val is not None else 1,
                    "upper": int(max_val) if max_val is not None else (int(default * 4) if default else 100),
                }
        elif "bool" in type_str:
            space[param_name] = {
                "space": "categorical",
                "categories": [True, False],
            }

    return space


def generate_trainer_search_space(
    model_type: str = "ecd",
    tunable_fields: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Generate a default search space for trainer hyperparameters.

    If no fields are specified, generates search spaces for commonly tuned parameters.

    Args:
        model_type: "ecd" or "llm".
        tunable_fields: Specific fields to tune. None for sensible defaults.

    Returns:
        Optuna-compatible search space dict.
    """
    if tunable_fields is None:
        tunable_fields = ["learning_rate", "batch_size", "dropout"]

    # Common sensible defaults that work regardless of schema inspection
    defaults = {
        "trainer.learning_rate": {"space": "loguniform", "lower": 1e-5, "upper": 0.01},
        "trainer.batch_size": {"space": "int", "lower": 16, "upper": 512},
        "trainer.dropout": {"space": "uniform", "lower": 0.0, "upper": 0.5},
        "combiner.num_layers": {"space": "int", "lower": 1, "upper": 6},
        "combiner.hidden_size": {"space": "categorical", "categories": [64, 128, 256, 512]},
        "combiner.num_heads": {"space": "categorical", "categories": [2, 4, 8]},
        "combiner.dropout": {"space": "uniform", "lower": 0.0, "upper": 0.5},
    }

    return {k: v for k, v in defaults.items() if any(f in k for f in tunable_fields)}
