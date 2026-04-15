#! /usr/bin/env python
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
import logging
from functools import partial

import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import (
    ANOMALY,
    BINARY,
    CATEGORY,
    CATEGORY_DISTRIBUTION,
    LOSS,
    NUMBER,
    SET,
    TIMESERIES,
    TYPE,
    VECTOR,
)
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder
from ludwig.schema.decoders.base import ClassifierConfig, PassthroughDecoderConfig, ProjectorConfig, RegressorConfig
from ludwig.utils.torch_utils import Dense, get_activation

logger = logging.getLogger(__name__)


@DeveloperAPI
# TODO(Arnav): Re-enable once we add DotProduct Combiner: https://github.com/ludwig-ai/ludwig/issues/3150
# @register_decoder("passthrough", [BINARY, CATEGORY, NUMBER, SET, VECTOR, SEQUENCE, TEXT])
class PassthroughDecoder(Decoder):
    def __init__(self, input_size: int = 1, num_classes: int = None, decoder_config=None, **kwargs):
        super().__init__()
        self.config = decoder_config

        logger.debug(f" {self.name}")
        self.input_size = input_size
        self.num_classes = num_classes

    def forward(self, inputs, **kwargs):
        return inputs

    @staticmethod
    def get_schema_cls():
        return PassthroughDecoderConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        return self.input_shape


@DeveloperAPI
@register_decoder("regressor", [BINARY, NUMBER])
class Regressor(Decoder):
    def __init__(
        self,
        input_size,
        use_bias=True,
        weights_initializer="xavier_uniform",
        bias_initializer="zeros",
        decoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = decoder_config

        logger.debug(f" {self.name}")

        logger.debug("  Dense")

        self.dense = Dense(
            input_size=input_size,
            output_size=1,
            use_bias=use_bias,
            weights_initializer=weights_initializer,
            bias_initializer=bias_initializer,
        )

    @staticmethod
    def get_schema_cls():
        return RegressorConfig

    @property
    def input_shape(self):
        return self.dense.input_shape

    def forward(self, inputs, **kwargs):
        return self.dense(inputs)


@DeveloperAPI
@register_decoder("projector", [VECTOR, TIMESERIES])
class Projector(Decoder):
    def __init__(
        self,
        input_size,
        output_size,
        use_bias=True,
        weights_initializer="xavier_uniform",
        bias_initializer="zeros",
        activation=None,
        multiplier=1.0,
        clip=None,
        decoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = decoder_config

        logger.debug(f" {self.name}")

        logger.debug("  Dense")
        self.dense = Dense(
            input_size=input_size,
            output_size=output_size,
            use_bias=use_bias,
            weights_initializer=weights_initializer,
            bias_initializer=bias_initializer,
        )

        self.activation = get_activation(activation)
        self.multiplier = multiplier

        if clip is not None:
            if isinstance(clip, (list, tuple)) and len(clip) == 2:
                self.clip = partial(torch.clip, min=clip[0], max=clip[1])
            else:
                raise ValueError(
                    "The clip parameter of {} is {}. "
                    "It must be a list or a tuple of length 2.".format(self.feature_name, self.clip)
                )
        else:
            self.clip = None

    @staticmethod
    def get_schema_cls():
        return ProjectorConfig

    @property
    def input_shape(self):
        return self.dense.input_shape

    def forward(self, inputs, **kwargs):
        values = self.activation(self.dense(inputs)) * self.multiplier
        if self.clip:
            values = self.clip(values)
        return values


@DeveloperAPI
@register_decoder("classifier", [CATEGORY, CATEGORY_DISTRIBUTION, SET])
class Classifier(Decoder):
    def __init__(
        self,
        input_size,
        num_classes,
        use_bias=True,
        weights_initializer="xavier_uniform",
        bias_initializer="zeros",
        decoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = decoder_config

        logger.debug(f" {self.name}")

        logger.debug("  Dense")
        self.num_classes = num_classes
        self.dense = Dense(
            input_size=input_size,
            output_size=num_classes,
            use_bias=use_bias,
            weights_initializer=weights_initializer,
            bias_initializer=bias_initializer,
        )

        self.sampled_loss = False
        if LOSS in kwargs and TYPE in kwargs[LOSS] and kwargs[LOSS][TYPE] is not None:
            self.sampled_loss = kwargs[LOSS][TYPE].startswith("sampled")

    @staticmethod
    def get_schema_cls():
        return ClassifierConfig

    @property
    def input_shape(self):
        return self.dense.input_shape

    def forward(self, inputs, **kwargs):
        return self.dense(inputs)


@DeveloperAPI
@register_decoder("anomaly", [ANOMALY])
class AnomalyDecoder(Decoder):
    """AnomalyDecoder: computes ||z - c||^2 as the anomaly score.

    The center ``c`` is the mean of all encoder outputs from the first training epoch,
    computed by ``AnomalyOutputFeature.initialize_center()`` and stored as a non-trainable
    ``register_buffer``.  Until that call the center is the zero vector, so anomaly scores
    are raw squared norms which is still a valid (if uncentered) distance metric.

    This implements the hard-boundary Deep SVDD objective from Ruff et al. (ICML 2018).
    With the ECD combiner you get *free* multimodal anomaly detection: feed tabular,
    image, text or any mix of input features and the combiner fuses them into a single
    latent vector that the decoder compares against the center.

    Args:
        input_size: Latent space dimensionality (set automatically from the FC stack).
        decoder_config: AnomalyDecoderConfig instance.
    """

    def __init__(self, input_size: int = None, decoder_config=None, **kwargs):
        super().__init__()
        self.config = decoder_config
        self.input_size = input_size or 1
        logger.debug(f" {self.name}")
        self.register_buffer("center", torch.zeros(self.input_size))
        self._center_initialized = False

    def initialize_center(self, center: torch.Tensor) -> None:
        """Set the hypersphere center from the mean of first-epoch encoder outputs.

        Args:
            center: Tensor of shape [input_size].
        """
        if center.shape != self.center.shape:
            raise ValueError(f"Center shape mismatch: expected {self.center.shape}, got {center.shape}.")
        self.center.copy_(center)
        self._center_initialized = True

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute anomaly scores as squared distance to hypersphere center.

        Args:
            inputs: Encoder output, shape [batch, input_size].

        Returns:
            Anomaly scores, shape [batch] (higher = more anomalous).
        """
        diff = inputs - self.center.unsqueeze(0)
        return (diff * diff).sum(dim=-1)

    @staticmethod
    def get_schema_cls():
        from ludwig.schema.decoders.base import AnomalyDecoderConfig

        return AnomalyDecoderConfig

    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([1])


@DeveloperAPI
@register_decoder("mlp_classifier", [CATEGORY, BINARY])
class MLPClassifier(Decoder):
    """Multi-layer perceptron classifier decoder.

    Stacks ``num_fc_layers`` fully-connected hidden layers (each of size ``output_size``, with
    configurable ``activation`` and ``dropout``) before a final linear projection to ``num_classes``
    logits.  When ``num_fc_layers=0`` this is equivalent to the standard ``Classifier`` decoder.
    Use this decoder when the combiner output benefits from additional non-linear transformation
    before the classification head -- for example, when using a simple concatenation combiner with
    heterogeneous input features.

    Supports two optional inference-time extensions:

    * **Temperature scaling** (``calibration='temperature_scaling'`` on the decoder config):
      After training, a single scalar *T* is learned on the validation set so that
      ``calibrated_logits = logits / T`` minimises negative log-likelihood.  This reliably
      reduces overconfidence without changing argmax predictions.
      See: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.

    * **MC Dropout uncertainty** (``mc_dropout_samples > 0``):
      At inference time the decoder is run ``mc_dropout_samples`` times with dropout layers kept
      in training mode.  The mean of the resulting probability distributions is returned as the
      prediction, and the per-class variance is reported as an ``uncertainty`` tensor.
      See: Gal & Ghahramani, "Dropout as a Bayesian Approximation: Representing Model
      Uncertainty in Deep Learning", ICML 2016.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int = None,
        num_fc_layers: int = 1,
        output_size: int = 256,
        activation: str = "relu",
        dropout: float = 0.0,
        use_bias: bool = True,
        weights_initializer: str = "xavier_uniform",
        bias_initializer: str = "zeros",
        mc_dropout_samples: int = 0,
        decoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = decoder_config

        logger.debug(f" {self.name}")

        # For binary features num_classes is not set by the caller; default to 1 (single logit).
        effective_num_classes = num_classes if num_classes is not None else 1
        self.num_classes = effective_num_classes
        self.mc_dropout_samples = mc_dropout_samples
        self._input_size = input_size

        # Build hidden FC layers.  FCLayer is imported here to avoid circular imports at
        # module load time (fully_connected_modules imports from torch_utils which imports Decoder).
        from ludwig.modules.fully_connected_modules import FCLayer

        self.fc_layers = torch.nn.ModuleList()
        current_size = input_size
        for i in range(num_fc_layers):
            logger.debug(f"  FCLayer {i}")
            layer = FCLayer(
                input_size=current_size,
                output_size=output_size,
                use_bias=use_bias,
                weights_initializer=weights_initializer,
                bias_initializer=bias_initializer,
                activation=activation,
                dropout=dropout,
            )
            self.fc_layers.append(layer)
            current_size = output_size

        # Final linear projection to logits.
        logger.debug("  Linear (classification head)")
        self.output_layer = torch.nn.Linear(current_size, effective_num_classes, bias=use_bias)
        if use_bias:
            torch.nn.init.zeros_(self.output_layer.bias)
        _init_fns = {
            "xavier_uniform": torch.nn.init.xavier_uniform_,
            "xavier_normal": torch.nn.init.xavier_normal_,
            "zeros": torch.nn.init.zeros_,
        }
        _init_fns.get(weights_initializer, torch.nn.init.xavier_uniform_)(self.output_layer.weight)

        self._hidden_size = current_size

    @staticmethod
    def get_schema_cls():
        from ludwig.schema.decoders.base import MLPClassifierConfig

        return MLPClassifierConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self._input_size])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.num_classes])

    def _single_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run one forward pass through hidden layers + classification head."""
        hidden = inputs
        for layer in self.fc_layers:
            hidden = layer(hidden)
        logits = self.output_layer(hidden)
        # For binary (num_classes=1), squeeze to match the expected 1-D logit shape.
        if self.num_classes == 1:
            logits = logits.squeeze(-1)
        return logits

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run the forward pass and return raw logits.

        Returns logits of shape ``[batch, num_classes]`` for category features,
        or ``[batch]`` for binary features.
        MC Dropout inference is available via :meth:`mc_forward`.
        """
        return self._single_forward(inputs)

    def mc_forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout inference: run ``mc_dropout_samples`` stochastic passes.

        Keeps dropout layers in training mode so each pass samples a different sub-network,
        then averages the resulting probability distributions.

        For category features returns (mean_probs [batch, num_classes], variance [batch, num_classes]).
        For binary features returns (mean_probs [batch, 2], variance [batch, 2]).

        Args:
            inputs: Float tensor of shape ``[batch, input_size]``.

        Returns:
            mean_probs: Averaged softmax/sigmoid probabilities over MC samples.
            uncertainty: Per-class variance across MC samples.

        References:
            Gal & Ghahramani, "Dropout as a Bayesian Approximation: Representing Model
            Uncertainty in Deep Learning", ICML 2016.
        """
        was_training = self.training
        # Enable dropout (train mode) while keeping BN in eval mode if present.
        self.train()
        with torch.no_grad():
            probs_list = []
            for _ in range(self.mc_dropout_samples):
                logits = self._single_forward(inputs)
                if self.num_classes == 1:
                    # Binary: convert scalar logit to 2-class probability pair for consistent output.
                    pos_prob = torch.sigmoid(logits)
                    probs = torch.stack([1 - pos_prob, pos_prob], dim=-1)
                else:
                    probs = torch.softmax(logits, dim=-1)
                probs_list.append(probs)
        if not was_training:
            self.eval()
        # Stack to [num_samples, batch, 2_or_num_classes]
        probs_stack = torch.stack(probs_list, dim=0)
        mean_probs = probs_stack.mean(dim=0)
        uncertainty = probs_stack.var(dim=0)
        return mean_probs, uncertainty
