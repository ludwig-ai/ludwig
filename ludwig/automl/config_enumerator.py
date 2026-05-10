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
"""Enumerates all valid (encoder, combiner, decoder) combinations for a given feature schema."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import NamedTuple

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import (
    AUDIO,
    BAG,
    BINARY,
    CATEGORY,
    DATE,
    H3,
    IMAGE,
    NUMBER,
    SEQUENCE,
    SET,
    TEXT,
    TIMESERIES,
    VECTOR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry: encoders per input feature type (ECD model)
# ---------------------------------------------------------------------------

ENCODER_REGISTRY: dict[str, list[str]] = {
    BINARY: ["passthrough", "dense"],
    CATEGORY: ["onehot", "passthrough", "dense", "sparse", "target", "hash"],
    NUMBER: ["passthrough", "dense", "ple", "periodic", "bins"],
    VECTOR: ["passthrough", "dense"],
    TEXT: [
        "bert",
        "distilbert",
        "roberta",
        "xlnet",
        "albert",
        "electra",
        "longformer",
        "auto_transformer",
        "deberta",
        "modernbert",
        "camembert",
        "gpt",
        "gpt2",
        "t5",
        "mt5",
        "xlm",
        "xlmroberta",
        "tf_idf",
        "embed",
        "rnn",
        "parallel_cnn",
        "stacked_cnn",
        "transformer",
        "mamba2",
    ],
    IMAGE: [
        "stacked_cnn",
        "resnet",
        "efficientnet",
        "vit",
        "densenet",
        "alexnet",
        "vgg",
        "googlenet",
        "inceptionv3",
        "mobilenetv2",
        "regnet",
        "convnext",
        "dinov2",
        "siglip",
        "swin_transformer",
        "maxvit",
    ],
    AUDIO: ["rnn", "stacked_cnn", "parallel_cnn", "wav2vec2", "hubert", "whisper", "mamba2"],
    TIMESERIES: [
        "dense",
        "rnn",
        "stacked_cnn",
        "parallel_cnn",
        "patchtst",
        "nbeats",
        "mamba2",
        "transformer",
        "passthrough",
    ],
    SEQUENCE: ["embed", "rnn", "stacked_cnn", "parallel_cnn", "transformer", "mamba2"],
    DATE: ["embed", "wave"],
    H3: ["embed", "rnn", "weighted_sum"],
    BAG: ["embed"],
    SET: ["embed"],
}

# ---------------------------------------------------------------------------
# Registry: decoders per output feature type (ECD model)
# ---------------------------------------------------------------------------

DECODER_REGISTRY: dict[str, list[str]] = {
    BINARY: ["mlp_classifier", "regressor"],
    CATEGORY: ["classifier", "mlp_classifier"],
    NUMBER: ["regressor"],
    TEXT: ["generator", "tagger", "transformer_generator"],
    SEQUENCE: ["generator", "tagger", "transformer_generator"],
    IMAGE: ["fpn", "segformer", "unet"],
    SET: ["classifier"],
    VECTOR: ["projector"],
    TIMESERIES: ["projector"],
}

# ---------------------------------------------------------------------------
# Combiner lists and compatibility constraints
# ---------------------------------------------------------------------------

ALL_COMBINERS: list[str] = [
    "concat",
    "tabnet",
    "transformer",
    "tabtransformer",
    "ft_transformer",
    "tabpfn_v2",
    "project_aggregate",
    "comparator",
    "sequence",
    "sequence_concat",
    "cross_attention",
    "perceiver",
    "gated_fusion",
    "hypernetwork",
]

# Feature types considered "tabular" for tabnet / tabpfn_v2 compatibility.
_TABULAR_TYPES: frozenset[str] = frozenset({BINARY, CATEGORY, NUMBER})

# Feature types that satisfy the sequence/timeseries combiner requirement.
_SEQUENTIAL_TYPES: frozenset[str] = frozenset({SEQUENCE, TEXT, TIMESERIES})


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@DeveloperAPI
class FeatureSpec(NamedTuple):
    """Minimal description of a feature: its column name and Ludwig feature type."""

    name: str
    type: str  # Ludwig feature type constant


@DeveloperAPI
@dataclass
class ConfigSpec:
    """A single valid (encoder assignments, combiner, decoder) combination."""

    input_encoders: dict[str, str]  # feature_name -> encoder_type
    combiner: str
    output_decoder: str
    output_type: str


# ---------------------------------------------------------------------------
# Core query functions
# ---------------------------------------------------------------------------


@DeveloperAPI
def get_valid_encoders(feature_type: str) -> list[str]:
    """Returns the list of valid encoder names for *feature_type*.

    # Inputs
    :param feature_type: (str) A Ludwig feature type constant (e.g. ``"text"``, ``"image"``).

    # Return
    :return: (list[str]) Encoder names registered for that feature type.  Returns an empty
        list for unknown feature types (with a warning logged).
    """
    encoders = ENCODER_REGISTRY.get(feature_type)
    if encoders is None:
        logger.warning(f"No encoder registry entry for feature type '{feature_type}'; returning empty list.")
        return []
    return list(encoders)


@DeveloperAPI
def get_valid_decoders(feature_type: str) -> list[str]:
    """Returns the list of valid decoder names for *feature_type*.

    # Inputs
    :param feature_type: (str) A Ludwig feature type constant used as an output feature.

    # Return
    :return: (list[str]) Decoder names registered for that feature type.  Returns an empty
        list for unknown/unsupported output feature types (with a warning logged).
    """
    decoders = DECODER_REGISTRY.get(feature_type)
    if decoders is None:
        logger.warning(f"No decoder registry entry for feature type '{feature_type}'; returning empty list.")
        return []
    return list(decoders)


@DeveloperAPI
def get_valid_combiners(input_features: list[FeatureSpec]) -> list[str]:
    """Returns the list of valid combiner names for the given input feature schema.

    Applies the following compatibility constraints:

    - ``"tabnet"`` and ``"tabpfn_v2"``: only when **all** input features are tabular
      (BINARY, CATEGORY, or NUMBER).
    - ``"comparator"``: only when there are **exactly 2** input features.
    - ``"sequence"`` and ``"sequence_concat"``: only when at least one input feature is
      SEQUENCE, TEXT, or TIMESERIES.
    - All other combiners are always compatible.

    # Inputs
    :param input_features: (list[FeatureSpec]) The input feature specifications.

    # Return
    :return: (list[str]) Compatible combiner names.
    """
    input_types = {f.type for f in input_features}
    n_inputs = len(input_features)
    all_tabular = input_types.issubset(_TABULAR_TYPES)
    has_sequential = bool(input_types & _SEQUENTIAL_TYPES)

    valid: list[str] = []
    for combiner in ALL_COMBINERS:
        if combiner in ("tabnet", "tabpfn_v2"):
            if not all_tabular:
                continue
        elif combiner == "comparator":
            if n_inputs != 2:
                continue
        elif combiner in ("sequence", "sequence_concat"):
            if not has_sequential:
                continue
        valid.append(combiner)

    return valid


# ---------------------------------------------------------------------------
# Full enumeration
# ---------------------------------------------------------------------------


@DeveloperAPI
def enumerate_config_specs(
    input_features: list[FeatureSpec],
    output_feature: FeatureSpec,
    max_configs: int | None = None,
) -> list[ConfigSpec]:
    """Enumerates all valid (encoder, combiner, decoder) combinations for the given schema.

    Each :class:`ConfigSpec` specifies one encoder per input feature, one combiner, and one
    decoder for the output feature.  The combinatorial space can be very large; *max_configs*
    caps the output using deterministic sampling across all axes.

    # Inputs
    :param input_features: (list[FeatureSpec]) The input feature specifications.
    :param output_feature: (FeatureSpec) The single output feature specification.
    :param max_configs: (int | None) Maximum number of ``ConfigSpec`` objects to return.
        ``None`` means unlimited.

    # Return
    :return: (list[ConfigSpec]) All (or up to *max_configs*) valid config specs.
    """
    # Collect per-feature encoder lists.
    per_feature_encoders: list[list[str]] = [get_valid_encoders(f.type) for f in input_features]

    valid_combiners = get_valid_combiners(input_features)
    valid_decoders = get_valid_decoders(output_feature.type)

    if not valid_combiners:
        logger.warning("No valid combiners for the given input feature schema; returning empty list.")
        return []
    if not valid_decoders:
        logger.warning(f"No valid decoders for output feature type '{output_feature.type}'; returning empty list.")
        return []

    # Total space size (may be huge).
    total = 1
    for enc_list in per_feature_encoders:
        total *= max(len(enc_list), 1)
    total *= len(valid_combiners) * len(valid_decoders)

    logger.debug(
        f"enumerate_config_specs: {total} total combinations "
        f"(features={len(input_features)}, combiners={len(valid_combiners)}, decoders={len(valid_decoders)})"
    )

    results: list[ConfigSpec] = []

    # Iterate deterministically: encoder combos × combiners × decoders.
    encoder_combos = itertools.product(*per_feature_encoders)

    if max_configs is not None and total > max_configs:
        # Deterministic stride-based sampling across the full Cartesian product.
        stride = max(1, total // max_configs)
        flat_idx = 0
        next_sample = 0
        for enc_combo in encoder_combos:
            for combiner in valid_combiners:
                for decoder in valid_decoders:
                    if flat_idx == next_sample:
                        results.append(
                            ConfigSpec(
                                input_encoders={f.name: enc for f, enc in zip(input_features, enc_combo)},
                                combiner=combiner,
                                output_decoder=decoder,
                                output_type=output_feature.type,
                            )
                        )
                        next_sample += stride
                        if len(results) >= max_configs:
                            return results
                    flat_idx += 1
    else:
        for enc_combo in encoder_combos:
            for combiner in valid_combiners:
                for decoder in valid_decoders:
                    results.append(
                        ConfigSpec(
                            input_encoders={f.name: enc for f, enc in zip(input_features, enc_combo)},
                            combiner=combiner,
                            output_decoder=decoder,
                            output_type=output_feature.type,
                        )
                    )

    return results
