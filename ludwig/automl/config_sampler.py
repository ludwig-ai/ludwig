# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Samples diverse Ludwig configs for a given feature schema."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass

import pandas as pd

from ludwig.api_annotations import DeveloperAPI
from ludwig.automl.config_enumerator import (
    ConfigSpec,
    FeatureSpec,
    get_valid_combiners,
    get_valid_decoders,
    get_valid_encoders,
)
from ludwig.constants import (
    BATCH_SIZE,
    COMBINER,
    INPUT_FEATURES,
    LEARNING_RATE,
    OUTPUT_FEATURES,
    TRAINER,
    TYPE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------

_LR_VALUES: list[float] = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
_BATCH_SIZES: list[int] = [64, 128, 256, 512]
_NUM_LAYERS: list[int] = [1, 2, 3]
_DROPOUTS: list[float] = [0.0, 0.1, 0.3]

# Maximum number of encoder choices sampled per feature type to keep configs tractable.
_MAX_ENCODERS_PER_TYPE: int = 3


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@DeveloperAPI
@dataclass
class SampledConfig:
    """A complete Ludwig-style config dict ready for training."""

    config_dict: dict  # Full Ludwig config as dict
    config_hash: str  # SHA-256 of canonical JSON
    spec: ConfigSpec
    trainer_params: dict  # LR, batch_size, etc.
    seed: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_json(obj: dict) -> str:
    """Returns a deterministic JSON string suitable for hashing."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=True)


def _hash_config(config_dict: dict) -> str:
    """Returns the SHA-256 hex digest of the canonical JSON representation of *config_dict*."""
    return hashlib.sha256(_canonical_json(config_dict).encode()).hexdigest()


def _build_config_dict(
    input_features: list[FeatureSpec],
    output_feature: FeatureSpec,
    spec: ConfigSpec,
    trainer_params: dict,
    combiner_params: dict | None = None,
) -> dict:
    """Builds a complete Ludwig config dict from a :class:`ConfigSpec` and trainer params."""
    input_feature_dicts = [
        {"name": f.name, TYPE: f.type, "encoder": {TYPE: spec.input_encoders[f.name]}} for f in input_features
    ]
    output_feature_dict = {
        "name": output_feature.name,
        TYPE: output_feature.type,
        "decoder": {TYPE: spec.output_decoder},
    }
    combiner_dict = combiner_params if combiner_params is not None else {TYPE: spec.combiner}
    return {
        INPUT_FEATURES: input_feature_dicts,
        OUTPUT_FEATURES: [output_feature_dict],
        COMBINER: combiner_dict,
        TRAINER: dict(trainer_params),
    }


def _sample_trainer_params(rng: random.Random, max_epochs: int, time_limit_s: int | None) -> dict:
    """Samples a single set of trainer hyperparameters from the predefined grids."""
    params: dict = {
        LEARNING_RATE: rng.choice(_LR_VALUES),
        BATCH_SIZE: rng.choice(_BATCH_SIZES),
        "epochs": max_epochs,
    }
    if time_limit_s is not None:
        params["time_limit_s"] = time_limit_s
    return params


def _sample_combiner_params(combiner: str, rng: random.Random) -> dict:
    """Samples combiner-level hyperparameters (FC layers, dropout, etc.)."""
    params: dict = {TYPE: combiner}
    # concat-like combiners support num_fc_layers / output_size / dropout.
    if combiner in ("concat", "tabtransformer", "ft_transformer", "project_aggregate", "gated_fusion", "hypernetwork"):
        params["num_fc_layers"] = rng.choice(_NUM_LAYERS)
        params["dropout"] = rng.choice(_DROPOUTS)
        params["output_size"] = rng.choice([64, 128, 256])
    elif combiner == "tabnet":
        params["size"] = rng.choice([8, 16, 32])
        params["output_size"] = rng.choice([8, 16, 32])
        params["num_steps"] = rng.choice([3, 5, 7])
    elif combiner == "transformer":
        params["num_heads"] = rng.choice([2, 4, 8])
        params["num_layers"] = rng.choice(_NUM_LAYERS)
        params["dropout"] = rng.choice(_DROPOUTS)
    return params


def _limited_encoders(feature_type: str, rng: random.Random) -> list[str]:
    """Returns up to *_MAX_ENCODERS_PER_TYPE* encoders for *feature_type*, sampled without replacement."""
    all_encoders = get_valid_encoders(feature_type)
    if len(all_encoders) <= _MAX_ENCODERS_PER_TYPE:
        return list(all_encoders)
    return rng.sample(all_encoders, _MAX_ENCODERS_PER_TYPE)


# ---------------------------------------------------------------------------
# Core sampling
# ---------------------------------------------------------------------------


@DeveloperAPI
def sample_configs(
    input_features: list[FeatureSpec],
    output_feature: FeatureSpec,
    n: int = 100,
    seed: int = 42,
    max_epochs: int = 50,
    time_limit_s: int | None = None,
) -> list[SampledConfig]:
    """Samples *n* diverse Ludwig configs for the given feature schema.

    Uses stratified sampling across combiners to ensure diversity:

    - At least 1 config per valid combiner (up to *n*).
    - Remaining configs distributed proportionally across combiners.
    - Trainer hyperparams sampled from predefined grids (LR, batch size, layers, dropout).
    - Encoder choices sampled per feature type (up to 3 options per type).
    - Configs are deduplicated by :attr:`SampledConfig.config_hash`.

    # Inputs
    :param input_features: (list[FeatureSpec]) The input feature specifications.
    :param output_feature: (FeatureSpec) The single output feature specification.
    :param n: (int) Target number of sampled configs (after deduplication may be lower).
    :param seed: (int) Random seed for reproducibility.
    :param max_epochs: (int) Maximum training epochs written into each config's trainer section.
    :param time_limit_s: (int | None) Optional wall-clock time limit passed to the trainer.

    # Return
    :return: (list[SampledConfig]) Deduplicated sampled configs (up to *n*).
    """
    rng = random.Random(seed)

    valid_combiners = get_valid_combiners(input_features)
    valid_decoders = get_valid_decoders(output_feature.type)

    if not valid_combiners:
        logger.warning("sample_configs: no valid combiners for the given input feature schema.")
        return []
    if not valid_decoders:
        logger.warning(f"sample_configs: no valid decoders for output type '{output_feature.type}'.")
        return []

    # Build combiner budget: at least 1 per combiner, remainder distributed evenly.
    n_combiners = len(valid_combiners)
    base_per_combiner = max(1, n // n_combiners)
    remainder = max(0, n - base_per_combiner * n_combiners)

    combiner_budget: dict[str, int] = {}
    for i, c in enumerate(valid_combiners):
        combiner_budget[c] = base_per_combiner + (1 if i < remainder else 0)

    seen_hashes: set[str] = set()
    results: list[SampledConfig] = []

    # Pre-sample limited encoder lists per feature type (same across all combiner iterations
    # for consistency, but regenerated each attempt inside the inner loop).
    for combiner, budget in combiner_budget.items():
        attempts = 0
        generated = 0
        max_attempts = budget * 20  # avoid infinite loop on very small spaces

        while generated < budget and attempts < max_attempts:
            attempts += 1

            # Sample one encoder per input feature.
            input_encoders: dict[str, str] = {}
            for feat in input_features:
                choices = _limited_encoders(feat.type, rng)
                if not choices:
                    logger.warning(f"sample_configs: no encoders for feature type '{feat.type}'; skipping feature.")
                    choices = ["passthrough"]
                input_encoders[feat.name] = rng.choice(choices)

            decoder = rng.choice(valid_decoders)
            trainer_params = _sample_trainer_params(rng, max_epochs, time_limit_s)
            combiner_params = _sample_combiner_params(combiner, rng)

            spec = ConfigSpec(
                input_encoders=input_encoders,
                combiner=combiner,
                output_decoder=decoder,
                output_type=output_feature.type,
            )
            config_dict = _build_config_dict(input_features, output_feature, spec, trainer_params, combiner_params)
            config_hash = _hash_config(config_dict)

            if config_hash in seen_hashes:
                continue

            seen_hashes.add(config_hash)
            results.append(
                SampledConfig(
                    config_dict=config_dict,
                    config_hash=config_hash,
                    spec=spec,
                    trainer_params=trainer_params,
                    seed=seed,
                )
            )
            generated += 1

        if generated < budget:
            logger.debug(
                f"sample_configs: combiner '{combiner}' only produced {generated}/{budget} unique configs "
                f"after {attempts} attempts (search space may be small)."
            )

    logger.info(f"sample_configs: produced {len(results)} unique configs (target={n}).")
    return results[:n]


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


@DeveloperAPI
def configs_from_dataframe(
    df: pd.DataFrame,
    target_column: str,
    n: int = 100,
    seed: int = 42,
) -> list[SampledConfig]:
    """Convenience function: infers the feature schema from *df* and samples *n* configs.

    Uses Ludwig's existing type-inference logic (:func:`~ludwig.utils.automl.type_inference.infer_type`)
    to determine feature types for every column.  The *target_column* is treated as the
    output feature; all other non-excluded columns become input features.

    # Inputs
    :param df: (pd.DataFrame) The dataset to infer types from.
    :param target_column: (str) Name of the output / target column.
    :param n: (int) Target number of sampled configs.
    :param seed: (int) Random seed for reproducibility.

    # Return
    :return: (list[SampledConfig]) Sampled configs for the inferred schema.
    """
    from ludwig.automl.base_config import convert_targets, get_dataset_info, get_field_metadata

    dataset_info = get_dataset_info(df)
    targets = convert_targets(target_column)
    metadata = get_field_metadata(dataset_info.fields, dataset_info.row_count, targets)

    input_features: list[FeatureSpec] = []
    output_feature: FeatureSpec | None = None

    for field_meta in metadata:
        if field_meta.name == target_column:
            output_feature = FeatureSpec(name=field_meta.name, type=field_meta.config.type)
        elif not field_meta.excluded and field_meta.mode == "input":
            input_features.append(FeatureSpec(name=field_meta.name, type=field_meta.config.type))

    if output_feature is None:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")
    if not input_features:
        raise ValueError("No valid input features could be inferred from the dataframe.")

    logger.info(
        f"configs_from_dataframe: inferred {len(input_features)} input features, "
        f"output='{output_feature.name}' (type={output_feature.type})."
    )

    return sample_configs(
        input_features=input_features,
        output_feature=output_feature,
        n=n,
        seed=seed,
    )
