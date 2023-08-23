# Copyright (c) 2019 Uber Technologies, Inc.
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
from abc import ABC, abstractmethod, abstractstaticmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from ludwig.constants import (
    ENCODER_OUTPUT,
    ENCODER_OUTPUT_STATE,
    HIDDEN,
    LENGTHS,
    LOGITS,
    LOSS,
    PREDICTIONS,
    PROBABILITIES,
)
from ludwig.decoders.registry import get_decoder_cls
from ludwig.encoders.registry import get_encoder_cls
from ludwig.features.feature_utils import get_input_size_with_dependencies
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.modules.loss_modules import create_loss
from ludwig.modules.metric_modules import LossMetric, LudwigMetric, MeanMetric
from ludwig.modules.metric_registry import get_metric_classes, get_metric_cls, get_metric_tensor_input
from ludwig.modules.reduction_modules import SequenceReducer
from ludwig.schema.features.base import BaseFeatureConfig, BaseOutputFeatureConfig
from ludwig.types import FeatureConfigDict, FeatureMetadataDict, PreprocessingConfigDict, TrainingSetMetadataDict
from ludwig.utils import output_feature_utils
from ludwig.utils.calibration import CalibrationModule
from ludwig.utils.torch_utils import LudwigModule
from ludwig.utils.types import DataFrame, TorchscriptPreprocessingInput

logger = logging.getLogger(__name__)


class BaseFeatureMixin(ABC):
    """Parent class for feature mixins.

    Feature mixins support preprocessing functionality shared across input and output features.
    """

    @abstractstaticmethod
    def type() -> str:
        """Returns the type of feature this mixin supports."""
        raise NotImplementedError

    @abstractstaticmethod
    def cast_column(column: DataFrame, backend) -> DataFrame:
        """Returns a copy of the dataset column for the given feature, potentially after a type cast.

        Args:
            column: Pandas column of values.
            backend: (Union[Backend, str]) Backend to use for feature data processing.
        """
        raise NotImplementedError

    @abstractstaticmethod
    def get_feature_meta(
        column: DataFrame, preprocessing_parameters: PreprocessingConfigDict, backend, is_input_feature: bool
    ) -> FeatureMetadataDict:
        """Returns a dictionary of feature metadata.

        Args:
            column: Pandas column of values.
            preprocessing_parameters: Preprocessing configuration for this feature.
            backend: (Union[Backend, str]) Backend to use for feature data processing.
        """
        raise NotImplementedError

    @abstractstaticmethod
    def add_feature_data(
        feature_config: FeatureConfigDict,
        input_df: DataFrame,
        proc_df: Dict[str, DataFrame],
        metadata: TrainingSetMetadataDict,
        preprocessing_parameters: PreprocessingConfigDict,
        backend,  # Union[Backend, str]
        skip_save_processed_input: bool,
    ) -> None:
        """Runs preprocessing on the input_df and stores results in the proc_df and metadata dictionaries.

        Args:
            feature_config: Feature configuration.
            input_df: Pandas column of values.
            proc_df: Dict of processed columns of data. Feature data is added to this.
            metadata: Metadata returned by get_feature_meta(). Additional information may be added to this.
            preprocessing_parameters: Preprocessing configuration for this feature.
            backend: (Union[Backend, str]) Backend to use for feature data processing.
            skip_save_processed_input: Whether to skip saving the processed input.
        """
        raise NotImplementedError


@dataclass
class ModuleWrapper:
    """Used to prevent the PredictModule from showing up an attribute on the feature module.

    This is necessary to avoid inflight errors from DeepSpeed. These errors occur when DeepSpeed believes that a param
    is still in the process of being processed asynchronously (allgathered, etc.).
    """

    module: torch.nn.Module


class PredictModule(torch.nn.Module):
    """Base class for all modules that convert model outputs to predictions.

    Explicit member variables needed here for scripting, as Torchscript will not be able to recognize global variables
    during scripting.
    """

    def __init__(self):
        super().__init__()
        self.predictions_key = PREDICTIONS
        self.probabilities_key = PROBABILITIES
        self.logits_key = LOGITS


class BaseFeature:
    """Base class for all features.

    Note that this class is not-cooperative (does not forward kwargs), so when constructing feature class hierarchies,
    there should be only one parent class that derives from base feature.  Other functionality should be put into mixin
    classes to avoid the diamond pattern.
    """

    def __init__(self, feature: BaseFeatureConfig):
        super().__init__()

        if not feature.name:
            raise ValueError("Missing feature name")
        self.feature_name = feature.name

        if not feature.column:
            feature.column = self.feature_name
        self.column = feature.column

        self.proc_column = feature.proc_column


class InputFeature(BaseFeature, LudwigModule, ABC):
    """Parent class for all input features."""

    def create_sample_input(self, batch_size: int = 2):
        # Used by get_model_inputs(), which is used for tracing-based torchscript generation.
        return torch.rand([batch_size, *self.input_shape]).to(self.input_dtype)

    def unskip(self) -> "InputFeature":
        """Convert feature using passthrough wrapper back to full encoder."""
        return self

    @staticmethod
    @abstractmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        pass

    def update_config_after_module_init(self, feature_config):
        """Updates the config after the torch.nn.Module objects have been initialized."""
        pass

    def initialize_encoder(self, encoder_config):
        encoder_cls = get_encoder_cls(self.type(), encoder_config.type)
        encoder_schema = encoder_cls.get_schema_cls().Schema()
        encoder_params_dict = encoder_schema.dump(encoder_config)
        return encoder_cls(encoder_config=encoder_config, **encoder_params_dict)

    @classmethod
    def get_preproc_input_dtype(cls, metadata: TrainingSetMetadataDict) -> str:
        return "string"

    @staticmethod
    def create_preproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
        raise NotImplementedError("Torchscript tracing not supported for feature")


class OutputFeature(BaseFeature, LudwigModule, ABC):
    """Parent class for all output features."""

    def __init__(
        self,
        feature: BaseOutputFeatureConfig,
        other_output_features: Dict[str, "OutputFeature"],
        *args,
        **kwargs,
    ):
        """Defines defaults, overwrites them based on the feature dictionary, and sets up dependencies.

        Any output feature can depend on one or more other output features. The `other_output_features` input dictionary
        should contain entries for any dependent output features, which is accomplished by constructing output features
        in topographically sorted order. Attributes of any dependent output features are used to properly initialize
        this feature's sizes.
        """
        super().__init__(feature)

        # List of names of metrics that this OutputFeature computes.
        self.metric_names = []
        self.loss = feature.loss
        self.reduce_input = feature.reduce_input
        self.reduce_dependencies = feature.reduce_dependencies

        # List of feature names that this output feature is dependent on.
        self.dependencies = feature.dependencies

        logger.debug(" output feature fully connected layers")
        logger.debug("  FCStack")

        self.input_size = get_input_size_with_dependencies(feature.input_size, self.dependencies, other_output_features)
        feature.input_size = self.input_size

        self.fc_stack = FCStack(
            first_layer_input_size=self.input_size,
            layers=feature.decoder.fc_layers,
            num_layers=feature.decoder.num_fc_layers,
            default_output_size=feature.decoder.fc_output_size,
            default_use_bias=feature.decoder.fc_use_bias,
            default_weights_initializer=feature.decoder.fc_weights_initializer,
            default_bias_initializer=feature.decoder.fc_bias_initializer,
            default_norm=feature.decoder.fc_norm,
            default_norm_params=feature.decoder.fc_norm_params,
            default_activation=feature.decoder.fc_activation,
            default_dropout=feature.decoder.fc_dropout,
        )
        self._calibration_module = self.create_calibration_module(feature)
        self._prediction_module = ModuleWrapper(self.create_predict_module())

        # set up two sequence reducers, one for inputs and other for dependencies
        self.reduce_sequence_input = SequenceReducer(reduce_mode=self.reduce_input)
        if self.dependencies:
            self.dependency_reducers = torch.nn.ModuleDict()
            # todo: re-evaluate need for separate handling of `attention` reducer
            #       currently this code does not support `attention`
            for dependency in self.dependencies:
                self.dependency_reducers[dependency] = SequenceReducer(reduce_mode=self.reduce_dependencies)

    def create_sample_output(self, batch_size: int = 2):
        output_shape = self.output_shape
        shape = [batch_size, *self.output_shape] if output_shape != torch.Size([1]) else [batch_size]
        return torch.rand(shape).to(self.get_output_dtype())

    @abstractmethod
    def get_prediction_set(self):
        """Returns the set of tensor keys returned by this feature's PredictModule.

        TODO(Justin): Move this to the PredictModule.
        """
        raise NotImplementedError("OutputFeature is missing implementation for get_prediction_set.")

    @classmethod
    @abstractmethod
    def get_output_dtype(cls):
        """Returns the Tensor data type feature outputs."""
        pass

    def initialize_decoder(self, decoder_config):
        # Input to the decoder is the output feature's FC hidden layer.
        decoder_config.input_size = self.fc_stack.output_shape[-1]
        decoder_cls = get_decoder_cls(self.type(), decoder_config.type)
        decoder_schema = decoder_cls.get_schema_cls().Schema()
        decoder_params_dict = decoder_schema.dump(decoder_config)
        return decoder_cls(decoder_config=decoder_config, **decoder_params_dict)

    def train_loss(self, targets: Tensor, predictions: Dict[str, Tensor], feature_name):
        loss_class = type(self.train_loss_function)
        prediction_key = output_feature_utils.get_feature_concat_name(feature_name, loss_class.get_loss_inputs())
        return self.train_loss_function(predictions[prediction_key], targets)

    def eval_loss(self, targets: Tensor, predictions: Dict[str, Tensor]):
        loss_class = type(self.train_loss_function)
        prediction_key = loss_class.get_loss_inputs()
        if isinstance(self.eval_loss_metric, MeanMetric):
            # MeanMetric's forward() implicitly updates the running average.
            # For MeanMetrics, we use get_current_value() to compute the loss without changing the state. All metrics
            # are updated at the BaseModel level as part of update_metrics().
            return self.eval_loss_metric.get_current_value(predictions[prediction_key].detach(), targets)
        return self.eval_loss_metric(predictions[prediction_key].detach(), targets)

    def _setup_loss(self):
        self.train_loss_function = create_loss(self.loss)
        self._eval_loss_metric = ModuleWrapper(get_metric_cls(self.type(), self.loss.type)(config=self.loss))

    def _setup_metrics(self):
        kwargs = {}
        for name, cls in get_metric_classes(self.type()).items():
            if cls.can_report(self) and isinstance(cls, LossMetric):
                kwargs[name] = cls(config=self.loss, **self.metric_kwargs())
            elif cls.can_report(self):
                kwargs[name] = cls(**self.metric_kwargs())
        self._metric_functions = {
            LOSS: self.eval_loss_metric,
            **kwargs,
        }
        self.metric_names = sorted(list(self._metric_functions.keys()))

    def create_calibration_module(self, feature: BaseOutputFeatureConfig) -> CalibrationModule:
        """Creates and returns a CalibrationModule that converts logits to a probability distribution."""
        return None

    @property
    def eval_loss_metric(self) -> LudwigMetric:
        return self._eval_loss_metric.module

    @property
    def calibration_module(self) -> torch.nn.Module:
        """Returns the CalibrationModule used to convert logits to a probability distribution."""
        return self._calibration_module

    @abstractmethod
    def create_predict_module(self) -> PredictModule:
        """Creates and returns a `nn.Module` that converts raw model outputs (logits) to predictions.

        This module is needed when generating the Torchscript model using scripting.
        """
        raise NotImplementedError()

    @property
    def prediction_module(self) -> PredictModule:
        """Returns the PredictModule used to convert model outputs to predictions."""
        return self._prediction_module.module

    def predictions(self, all_decoder_outputs: Dict[str, torch.Tensor], feature_name: str) -> Dict[str, torch.Tensor]:
        """Computes actual predictions from the outputs of feature decoders.

        TODO(Justin): Consider refactoring this to accept feature-specific decoder outputs.

        Args:
            all_decoder_outputs: A dictionary of {feature name}::{tensor_name} -> output tensor.
        Returns:
            Dictionary of tensors with predictions as well as any additional tensors that may be
            necessary for computing evaluation metrics.
        """
        return self.prediction_module(all_decoder_outputs, feature_name)

    @abstractmethod
    def logits(self, combiner_outputs: Dict[str, torch.Tensor], target=None, **kwargs) -> Dict[str, torch.Tensor]:
        """Unpacks and feeds combiner_outputs to the decoder. Invoked as part of the output feature's forward pass.

        If target is not None, then we are in training.

        Args:
            combiner_outputs: Dictionary of tensors from the combiner's forward pass.
        Returns:
            Dictionary of decoder's output tensors (non-normalized), as well as any additional
            tensors that may be necessary for computing predictions or evaluation metrics.
        """
        raise NotImplementedError("OutputFeature is missing logits() implementation.")

    def metric_kwargs(self) -> Dict[str, Any]:
        """Returns arguments that are used to instantiate an instance of each metric class."""
        return {}

    def update_metrics(self, targets: Tensor, predictions: Dict[str, Tensor]) -> None:
        """Updates metrics with the given targets and predictions.

        Args:
            targets: Tensor with target values for this output feature.
            predictions: Dict of tensors returned by predictions().
        """
        for metric_name, metric_fn in self._metric_functions.items():
            prediction_key = get_metric_tensor_input(metric_name)
            metric_fn = metric_fn.to(predictions[prediction_key].device)
            metric_fn.update(predictions[prediction_key].detach(), targets)

    def get_metrics(self):
        metric_vals = {}
        for metric_name, metric_fn in self._metric_functions.items():
            try:
                computed_metric = metric_fn.compute()
            except Exception as e:
                logger.exception(f"Caught exception computing metric: {metric_name} with error: {e}.")
                continue

            # Metrics from torchmetrics can be a straightforward tensor.
            if isinstance(computed_metric, Tensor):
                metric_vals[metric_name] = computed_metric.detach().cpu().numpy().item()
            else:
                # Metrics from torchmetrics can be a dict of tensors.
                # For example, ROUGE is returned as a dictionary of tensors.
                # Unpack.
                for sub_metric_name, metric in computed_metric.items():
                    metric_vals[sub_metric_name] = metric.detach().cpu().numpy().item()
        return metric_vals

    def reset_metrics(self):
        for _, metric_fn in self._metric_functions.items():
            if metric_fn is not None:
                metric_fn.reset()

    def forward(
        self,
        combiner_outputs: Dict[str, torch.Tensor],
        other_output_feature_outputs: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass that takes in output from the combiner, and passes it through to the decoder.

        Args:
            combiner_outputs: Dict of outputs from the combiner.
            other_output_feature_outputs: Dict of tensors from other output features. Used for resolving dependencies.
            mask: (Unused). Tensor for masking.
            target: Tensor with targets. During training, targets != None. During prediction, targets = None.

        Returns:
            Dict of output tensors, with at least 'last_hidden' and 'logits' as keys, as well as any additional tensor
            results from the decoder.
        """
        # extract the combined hidden layer
        combiner_hidden = combiner_outputs["combiner_output"]
        hidden = self.prepare_decoder_inputs(combiner_hidden, other_output_feature_outputs, mask=mask)

        # ================ Predictions ================
        logits_input = {HIDDEN: hidden}
        # pass supplemental data from encoders to decoder
        if ENCODER_OUTPUT_STATE in combiner_outputs:
            logits_input[ENCODER_OUTPUT_STATE] = combiner_outputs[ENCODER_OUTPUT_STATE]
        if LENGTHS in combiner_outputs:
            logits_input[LENGTHS] = combiner_outputs[LENGTHS]

        logits = self.logits(logits_input, target=target)

        # For binary and number features, self.logits() is a tensor.
        # There are two special cases where self.logits() is a dict:
        #   categorical
        #       keys: logits, projection_input
        #   sequence
        #       keys: logits
        # TODO(Justin): Clean this up.
        if isinstance(logits, Tensor):
            logits = {"logits": logits}

        # For multi-class features, we must choose a consistent tuple subset.
        return {
            # last_hidden used for dependencies processing
            "last_hidden": hidden,
            **logits,
        }

    @abstractmethod
    def postprocess_predictions(
        self,
        result: Dict[str, Tensor],
        metadata: TrainingSetMetadataDict,
    ):
        raise NotImplementedError

    @classmethod
    def get_postproc_output_dtype(cls, metadata: TrainingSetMetadataDict) -> str:
        return "string"

    @staticmethod
    def create_postproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
        raise NotImplementedError("Torchscript tracing not supported for feature")

    @staticmethod
    @abstractmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def calculate_overall_stats(predictions, targets, train_set_metadata):
        pass

    def output_specific_fully_connected(self, inputs, mask=None):
        feature_hidden = inputs
        original_feature_hidden = inputs

        # flatten inputs
        if len(original_feature_hidden.shape) > 2:
            feature_hidden = torch.reshape(feature_hidden, (-1, list(feature_hidden.shape)[-1]))

        # pass it through fc_stack
        feature_hidden = self.fc_stack(feature_hidden, mask=mask)
        feature_hidden_size = feature_hidden.shape[-1]

        # reshape back to original first and second dimension
        if len(original_feature_hidden.shape) > 2:
            sequence_length = original_feature_hidden.shape[1]
            feature_hidden = torch.reshape(feature_hidden, (-1, sequence_length, feature_hidden_size))

        return feature_hidden

    def prepare_decoder_inputs(
        self, combiner_hidden: Tensor, other_output_features: Dict[str, Tensor], mask=None
    ) -> Tensor:
        """Takes the combiner output and the outputs of other outputs features computed so far and performs:

        - reduction of combiner outputs (if needed)
        - concatenating the outputs of dependent features (if needed)
        - output_specific fully connected layers (if needed)

        Args:
            combiner_hidden: hidden state of the combiner
            other_output_features: output tensors from other output features
        """
        # ================ Reduce Inputs ================
        feature_hidden = combiner_hidden
        if self.reduce_input is not None and len(combiner_hidden.shape) > 2:
            feature_hidden = self.reduce_sequence_input(combiner_hidden)

        # ================ Concat Dependencies ================
        if self.dependencies:
            feature_hidden = output_feature_utils.concat_dependencies(
                self.column, self.dependencies, self.dependency_reducers, feature_hidden, other_output_features
            )

        # ================ Output-wise Fully Connected ================
        feature_hidden = self.output_specific_fully_connected(feature_hidden, mask=mask)

        return feature_hidden


class PassthroughPreprocModule(torch.nn.Module):
    """Combines preprocessing and encoding into a single module for TorchScript inference.

    For encoder outputs that were cached during preprocessing, the encoder is simply the identity function in the ECD
    module. As such, we need this module to apply the encoding that would normally be done during preprocessing for
    realtime inference.
    """

    def __init__(self, preproc: torch.nn.Module, encoder: torch.nn.Module):
        self.preproc = preproc
        self.encoder = encoder

    def forward(self, v: TorchscriptPreprocessingInput) -> torch.Tensor:
        preproc_v = self.preproc(v)
        return self.encoder(preproc_v)


def create_passthrough_input_feature(feature: InputFeature, config: BaseFeatureConfig) -> InputFeature:
    """Creates a shim input feature that acts as a transparent identifiy function on the input data.

    Used when the feature's encoder embeddings were cached in preprocessing. This way, we don't need to make any changes
    to the underlying interface in such cases other than to swap the feature that would normally do the encoding with
    this one.
    """

    class _InputPassthroughFeature(InputFeature):
        def __init__(self, config: BaseFeatureConfig):
            super().__init__(config)

        def forward(self, inputs, mask=None):
            assert isinstance(inputs, torch.Tensor)
            return {ENCODER_OUTPUT: inputs}

        @property
        def input_dtype(self):
            # Doesn't matter as combiner will need to cast them to float32 anyway
            return torch.float32

        @property
        def input_shape(self):
            return feature.encoder_obj.output_shape

        @property
        def output_shape(self) -> torch.Size:
            return feature.encoder_obj.output_shape

        @staticmethod
        def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
            return feature.update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs)

        @staticmethod
        def get_schema_cls():
            return feature.get_schema_cls()

        @staticmethod
        def create_preproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
            return PassthroughPreprocModule(feature.create_preproc_module(metadata), feature)

        @staticmethod
        def type():
            return feature.type()

        def unskip(self) -> InputFeature:
            return feature

        @property
        def encoder_obj(self) -> torch.nn.Module:
            return feature.encoder_obj

    return _InputPassthroughFeature(config)
