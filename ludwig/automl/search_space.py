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
"""YAML-driven SearchSpace loader for AutoML hyperparameter grids."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ludwig.api_annotations import DeveloperAPI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EncoderSpec:
    """Specification for a single encoder type."""

    name: str
    feature_types: list[str]
    preprocessing: dict = field(default_factory=dict)
    hyperparameters: dict[str, list] = field(default_factory=dict)


@dataclass
class CombinerSpec:
    """Specification for a single combiner type."""

    name: str
    constraints: dict = field(default_factory=dict)
    hyperparameters: dict[str, list] = field(default_factory=dict)


@dataclass
class DecoderSpec:
    """Specification for a single decoder type."""

    name: str
    feature_types: list[str]
    hyperparameters: dict[str, list] = field(default_factory=dict)


@dataclass
class TrainerSpec:
    """Trainer hyperparameter grids."""

    learning_rate_values: list[float]
    batch_size_values: list[int]
    default_epochs: int


# ---------------------------------------------------------------------------
# Built-in defaults (must exactly match prior hardcoded values)
# ---------------------------------------------------------------------------

_DEFAULT_ENCODER_REGISTRY: dict[str, list[str]] = {
    "binary": ["passthrough", "dense"],
    "category": ["onehot", "passthrough", "dense", "sparse", "target", "hash"],
    "number": ["passthrough", "dense", "ple", "periodic", "bins"],
    "vector": ["passthrough", "dense"],
    "text": [
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
    "image": [
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
    "audio": ["rnn", "stacked_cnn", "parallel_cnn", "wav2vec2", "hubert", "whisper", "mamba2"],
    "timeseries": [
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
    "sequence": ["embed", "rnn", "stacked_cnn", "parallel_cnn", "transformer", "mamba2"],
    "date": ["embed", "wave"],
    "h3": ["embed", "rnn", "weighted_sum"],
    "bag": ["embed"],
    "set": ["embed"],
}

_DEFAULT_DECODER_REGISTRY: dict[str, list[str]] = {
    "binary": ["mlp_classifier", "regressor"],
    "category": ["classifier", "mlp_classifier"],
    "number": ["regressor"],
    "text": ["generator", "tagger", "transformer_generator"],
    "sequence": ["generator", "tagger", "transformer_generator"],
    "image": ["fpn", "segformer", "unet"],
    "set": ["classifier"],
    "vector": ["projector"],
    "timeseries": ["projector"],
}

_DEFAULT_ALL_COMBINERS: list[str] = [
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

_DEFAULT_COMBINER_CONSTRAINTS: dict[str, dict] = {
    "tabnet": {"requires_all_tabular": True},
    "tabpfn_v2": {"requires_all_tabular": True},
    "tabtransformer": {"requires_all_tabular": True},
    "ft_transformer": {"requires_all_tabular": True},
    "comparator": {"exact_n_inputs": 2},
    "sequence": {"requires_sequential": True},
    "sequence_concat": {"requires_sequential": True},
}

# Shared FC-layer hyperparam grid used by several combiners.
_FC_HYPERPARAMS: dict[str, list] = {
    "num_fc_layers": [1, 2, 3],
    "output_size": [64, 128, 256],
    "dropout": [0.0, 0.1, 0.3],
}

_DEFAULT_COMBINER_HYPERPARAMS: dict[str, dict[str, list]] = {
    "concat": _FC_HYPERPARAMS,
    "tabtransformer": _FC_HYPERPARAMS,
    "ft_transformer": _FC_HYPERPARAMS,
    "project_aggregate": _FC_HYPERPARAMS,
    "gated_fusion": _FC_HYPERPARAMS,
    "hypernetwork": _FC_HYPERPARAMS,
    "tabnet": {"size": [8, 16, 32], "output_size": [8, 16, 32], "num_steps": [3, 5, 7]},
    "transformer": {"num_heads": [2, 4, 8], "num_layers": [1, 2, 3], "dropout": [0.0, 0.1, 0.3]},
}

_DEFAULT_TRAINER_SPEC = TrainerSpec(
    learning_rate_values=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    batch_size_values=[64, 128, 256, 512],
    default_epochs=50,
)


def _build_default_search_space() -> SearchSpace:
    """Constructs a SearchSpace from the hardcoded defaults (no YAML loading)."""
    encoders: dict[str, EncoderSpec] = {}
    for feat_type, enc_names in _DEFAULT_ENCODER_REGISTRY.items():
        for enc_name in enc_names:
            if enc_name not in encoders:
                encoders[enc_name] = EncoderSpec(name=enc_name, feature_types=[])
            if feat_type not in encoders[enc_name].feature_types:
                encoders[enc_name].feature_types.append(feat_type)

    combiners: dict[str, CombinerSpec] = {
        name: CombinerSpec(
            name=name,
            constraints=_DEFAULT_COMBINER_CONSTRAINTS.get(name, {}),
            hyperparameters=_DEFAULT_COMBINER_HYPERPARAMS.get(name, {}),
        )
        for name in _DEFAULT_ALL_COMBINERS
    }

    decoders: dict[str, DecoderSpec] = {}
    for feat_type, dec_names in _DEFAULT_DECODER_REGISTRY.items():
        for dec_name in dec_names:
            if dec_name not in decoders:
                decoders[dec_name] = DecoderSpec(name=dec_name, feature_types=[])
            if feat_type not in decoders[dec_name].feature_types:
                decoders[dec_name].feature_types.append(feat_type)

    return SearchSpace._from_specs(encoders, combiners, decoders, _DEFAULT_TRAINER_SPEC)


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------


def _extract_hyperparameters(raw: dict[str, Any]) -> dict[str, list]:
    """Extracts only samplable (``values``-bearing) entries from a raw hyperparameters dict."""
    result: dict[str, list] = {}
    for param_name, param_spec in raw.items():
        if isinstance(param_spec, dict) and "values" in param_spec:
            result[param_name] = list(param_spec["values"])
        # entries with only ``default`` are intentionally omitted from the sampling pool
    return result


def _iter_yaml_dir(spec_dir: Path) -> Iterator[tuple[Path, dict]]:
    """Yields (path, data) for each valid named YAML file in *spec_dir*."""
    import yaml

    for yaml_path in sorted(spec_dir.glob("*.yaml")):
        with open(yaml_path) as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, dict) and "name" in data:
            yield yaml_path, data
        else:
            logger.warning(f"Skipping YAML without 'name' key: {yaml_path}")


def _load_encoders_from_dir(enc_dir: Path) -> dict[str, EncoderSpec]:
    """Loads all encoder YAML files from *enc_dir*."""
    return {
        data["name"]: EncoderSpec(
            name=data["name"],
            feature_types=list(data.get("feature_types", [])),
            preprocessing=dict(data.get("preprocessing", {})),
            hyperparameters=_extract_hyperparameters(data.get("hyperparameters", {})),
        )
        for _, data in _iter_yaml_dir(enc_dir)
    }


def _load_combiners_from_dir(comb_dir: Path) -> dict[str, CombinerSpec]:
    """Loads all combiner YAML files from *comb_dir*."""
    return {
        data["name"]: CombinerSpec(
            name=data["name"],
            constraints=dict(data.get("constraints", {})),
            hyperparameters=_extract_hyperparameters(data.get("hyperparameters", {})),
        )
        for _, data in _iter_yaml_dir(comb_dir)
    }


def _load_decoders_from_dir(dec_dir: Path) -> dict[str, DecoderSpec]:
    """Loads all decoder YAML files from *dec_dir*."""
    return {
        data["name"]: DecoderSpec(
            name=data["name"],
            feature_types=list(data.get("feature_types", [])),
            hyperparameters=_extract_hyperparameters(data.get("hyperparameters", {})),
        )
        for _, data in _iter_yaml_dir(dec_dir)
    }


def _load_trainer_from_yaml(trainer_path: Path) -> TrainerSpec:
    """Loads trainer hyperparameter grids from a YAML file."""
    import yaml

    with open(trainer_path) as fh:
        data = yaml.safe_load(fh)

    lr_spec = data.get("learning_rate", {})
    bs_spec = data.get("batch_size", {})
    epochs_spec = data.get("epochs", {})

    return TrainerSpec(
        learning_rate_values=list(lr_spec.get("values", _DEFAULT_TRAINER_SPEC.learning_rate_values)),
        batch_size_values=list(bs_spec.get("values", _DEFAULT_TRAINER_SPEC.batch_size_values)),
        default_epochs=int(epochs_spec.get("default", _DEFAULT_TRAINER_SPEC.default_epochs)),
    )


# ---------------------------------------------------------------------------
# SearchSpace
# ---------------------------------------------------------------------------


@DeveloperAPI
class SearchSpace:
    """Container for encoder, combiner, decoder, and trainer hyperparameter grids.

    Construct from YAML files in a directory hierarchy::

        search_space/
          encoders/   *.yaml
          combiners/  *.yaml
          decoders/   *.yaml
          trainer.yaml

    Or rely on the built-in defaults by passing ``search_space_dir=None``.

    # Inputs
    :param search_space_dir: (str | os.PathLike | None) Root directory containing the
        ``encoders/``, ``combiners/``, ``decoders/`` subdirectories and ``trainer.yaml``.
        Pass ``None`` to use the built-in hardcoded defaults.
    """

    def __init__(self, search_space_dir: str | os.PathLike | None = None) -> None:
        if search_space_dir is None:
            _ss = _build_default_search_space()
            self.encoders = _ss.encoders
            self.combiners = _ss.combiners
            self.decoders = _ss.decoders
            self.trainer = _ss.trainer
        else:
            root = Path(search_space_dir)
            self.encoders = _load_encoders_from_dir(root / "encoders")
            self.combiners = _load_combiners_from_dir(root / "combiners")
            self.decoders = _load_decoders_from_dir(root / "decoders")
            trainer_yaml = root / "trainer.yaml"
            self.trainer = _load_trainer_from_yaml(trainer_yaml) if trainer_yaml.exists() else _DEFAULT_TRAINER_SPEC

        self.encoder_registry: dict[str, list[str]] = self._build_encoder_registry()
        self.decoder_registry: dict[str, list[str]] = self._build_decoder_registry()

    @classmethod
    def _from_specs(
        cls,
        encoders: dict[str, EncoderSpec],
        combiners: dict[str, CombinerSpec],
        decoders: dict[str, DecoderSpec],
        trainer: TrainerSpec,
    ) -> SearchSpace:
        """Internal constructor: build a SearchSpace directly from spec dicts."""
        instance = cls.__new__(cls)
        instance.encoders = encoders
        instance.combiners = combiners
        instance.decoders = decoders
        instance.trainer = trainer
        instance.encoder_registry = instance._build_encoder_registry()
        instance.decoder_registry = instance._build_decoder_registry()
        return instance

    # ------------------------------------------------------------------
    # Registry builders
    # ------------------------------------------------------------------

    def _build_encoder_registry(self) -> dict[str, list[str]]:
        registry: dict[str, list[str]] = {}
        for spec in self.encoders.values():
            for ft in spec.feature_types:
                registry.setdefault(ft, []).append(spec.name)
        return registry

    def _build_decoder_registry(self) -> dict[str, list[str]]:
        registry: dict[str, list[str]] = {}
        for spec in self.decoders.values():
            for ft in spec.feature_types:
                registry.setdefault(ft, []).append(spec.name)
        return registry

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_hyperparams(
        self,
        spec: EncoderSpec | CombinerSpec | DecoderSpec,
        rng: random.Random,
    ) -> dict:
        """Returns a dict of sampled hyperparameter values for *spec*.

        For each param in ``spec.hyperparameters``, one value is chosen uniformly at random
        from its ``values`` list.  The ``type`` key is always set to ``spec.name``.

        # Inputs
        :param spec: An :class:`EncoderSpec`, :class:`CombinerSpec`, or :class:`DecoderSpec`.
        :param rng: (:class:`random.Random`) Random number generator.

        # Return
        :return: (dict) Sampled hyperparameter dict, always including ``{"type": spec.name}``.
        """
        params: dict = {"type": spec.name}
        for param_name, values in spec.hyperparameters.items():
            if values:
                params[param_name] = rng.choice(values)
        return params


# ---------------------------------------------------------------------------
# Module-level default instance (lazily created)
# ---------------------------------------------------------------------------

_DEFAULT_SEARCH_SPACE: SearchSpace | None = None


def _default_search_space() -> SearchSpace:
    """Returns the lazily-instantiated module-level default :class:`SearchSpace`."""
    global _DEFAULT_SEARCH_SPACE
    if _DEFAULT_SEARCH_SPACE is None:
        _DEFAULT_SEARCH_SPACE = SearchSpace()
    return _DEFAULT_SEARCH_SPACE
