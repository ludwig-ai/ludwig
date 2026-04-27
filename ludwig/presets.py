"""Quality presets for Ludwig ECD models.

Inspired by AutoGluon's quality presets, these provide sensible defaults for different quality/speed tradeoffs. User-
specified config values always take precedence over preset defaults.
"""

import copy
import logging

logger = logging.getLogger(__name__)


QUALITY_PRESETS = {
    "medium_quality": {
        "combiner": {"type": "concat", "num_fc_layers": 2, "output_size": 128},
        "trainer": {
            "epochs": 50,
            "early_stop": 5,
            "batch_size": 256,
        },
    },
    "high_quality": {
        "combiner": {"type": "transformer", "num_layers": 2, "hidden_size": 256, "num_heads": 8},
        "trainer": {
            "epochs": 100,
            "early_stop": 10,
            "batch_size": 128,
            "loss_balancing": "uncertainty",
        },
    },
    "best_quality": {
        "combiner": {"type": "ft_transformer", "num_layers": 4, "hidden_size": 256, "num_heads": 8},
        "trainer": {
            "epochs": 200,
            "early_stop": 20,
            "batch_size": 64,
            "loss_balancing": "uncertainty",
            "model_soup": "uniform",
            "model_soup_top_k": 5,
        },
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base.

    Override values take precedence.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_preset(config_dict: dict, preset_name: str) -> dict:
    """Apply a quality preset to a config dict.

    User config takes precedence over preset defaults.
        Args:
            config_dict: The user's config dict.
            preset_name: Name of the preset to apply.

        Returns:
            Config dict with preset defaults applied (user overrides win).
    """
    if preset_name not in QUALITY_PRESETS:
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {list(QUALITY_PRESETS.keys())}")

    preset = QUALITY_PRESETS[preset_name]
    # Merge: preset is the base, user config overrides
    result = _deep_merge(preset, config_dict)
    logger.info(f"Applied quality preset '{preset_name}'")
    return result
