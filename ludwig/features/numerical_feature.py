#! /usr/bin/env python
# coding=utf-8
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
from typing import Dict
import random

import numpy as np
import torch

from ludwig.constants import *
from ludwig.decoders.generic_decoders import Regressor
from ludwig.encoders.generic_encoders import PassthroughEncoder, DenseEncoder
from ludwig.features.base_feature import InputFeature
from ludwig.features.base_feature import OutputFeature
from ludwig.modules.loss_modules import MSELoss, MAELoss, RMSELoss, RMSPELoss, get_loss_cls
from ludwig.modules.metric_modules import (
    MAEMetric,
    MSEMetric,
    RMSEMetric,
    RMSPEMetric,
    R2Score, get_metric_classes,
)
from ludwig.utils import output_feature_utils
from ludwig.utils.misc_utils import set_default_value
from ludwig.utils.misc_utils import set_default_values
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.registry import Registry, DEFAULT_KEYS

logger = logging.getLogger(__name__)


class ZScoreTransformer:
    def __init__(self, mean: float = None, std: float = None, **kwargs: dict):
        self.mu = mean
        self.sigma = std

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sigma

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.sigma + self.mu

    @staticmethod
    def fit_transform_params(column: np.ndarray, backend: "Backend") -> dict:
        compute = backend.df_engine.compute
        return {
            "mean": compute(column.astype(np.float32).mean()),
            "std": compute(column.astype(np.float32).std()),
        }


class MinMaxTransformer:
    def __init__(self, min: float = None, max: float = None, **kwargs: dict):
        self.min_value = min
        self.max_value = max
        self.range = None if min is None or max is None else max - min

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.min_value) / self.range

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.range is None:
            raise ValueError(
                "Numeric transformer needs to be instantiated with "
                "min and max values."
            )
        return x * self.range + self.min_value

    @staticmethod
    def fit_transform_params(column: np.ndarray, backend: "Backend") -> dict:
        compute = backend.df_engine.compute
        return {
            "min": compute(column.astype(np.float32).min()),
            "max": compute(column.astype(np.float32).max()),
        }


class Log1pTransformer:
    def __init__(self, **kwargs: dict):
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        if np.any(x <= 0):
            raise ValueError(
                "One or more values are non-positive.  "
                "log1p normalization is defined only for positive values."
            )
        return np.log1p(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.expm1(x)

    @staticmethod
    def fit_transform_params(column: np.ndarray, backend: "Backend") -> dict:
        return {}


class IdentityTransformer:
    def __init__(self, **kwargs):
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def fit_transform_params(column: np.ndarray, backend: "Backend") -> dict:
        return {}


numeric_transformation_registry = {
    "minmax": MinMaxTransformer,
    "zscore": ZScoreTransformer,
    "log1p": Log1pTransformer,
    None: IdentityTransformer,
}


class NumericalFeatureMixin:
    type = NUMERICAL
    preprocessing_defaults = {
        "missing_value_strategy": FILL_WITH_CONST,
        "fill_value": 0,
        "normalization": None,
    }

    preprocessing_schema = {
        "missing_value_strategy": {
            "type": "string",
            "enum": MISSING_VALUE_STRATEGY_OPTIONS,
        },
        "fill_value": {"type": "number"},
        "computed_fill_value": {"type": "number"},
        "normalization": {
            "type": ["string", "null"],
            "enum": list(numeric_transformation_registry.keys()),
        },
    }

    @staticmethod
    def cast_column(column, backend):
        return backend.df_engine.df_lib.to_numeric(
            column, errors="coerce"
        ).astype(np.float32)

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters, backend):
        numeric_transformer = get_from_registry(
            preprocessing_parameters.get("normalization", None),
            numeric_transformation_registry,
        )

        return numeric_transformer.fit_transform_params(column, backend)

    @staticmethod
    def add_feature_data(
            feature,
            input_df,
            proc_df,
            metadata,
            preprocessing_parameters,
            backend,
            skip_save_processed_input,
    ):
        proc_df[feature[PROC_COLUMN]] = (
            input_df[feature[COLUMN]].astype(np.float32).values
        )

        # normalize data as required
        numeric_transformer = get_from_registry(
            preprocessing_parameters.get("normalization", None),
            numeric_transformation_registry,
        )(**metadata[feature[NAME]])

        proc_df[feature[PROC_COLUMN]] = numeric_transformer.transform(
            proc_df[feature[PROC_COLUMN]]
        )

        return proc_df


class NumericalInputFeature(NumericalFeatureMixin, InputFeature):
    encoder = "passthrough"

    def __init__(self, feature, encoder_obj=None):
        # Required for certain encoders, maybe pass into initialize_encoder
        super().__init__(feature)
        self.overwrite_defaults(feature)
        feature['input_size'] = self.input_shape[-1]
        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(feature)

    def forward(self, inputs):
        assert isinstance(inputs, torch.Tensor)
        assert inputs.dtype == torch.float32 or inputs.dtype == torch.float64
        assert len(inputs.shape) == 1 or (
            len(inputs.shape) == 2 and inputs.shape[1] == 1)

        if len(inputs.shape) == 1:
            inputs = inputs[:, None]
        inputs_encoded = self.encoder_obj(inputs)

        return inputs_encoded

    def create_sample_input(self):
        # Used by get_model_inputs(), which is used for tracing-based torchscript generation.
        return torch.Tensor([random.randint(1, 100), random.randint(1, 100)])

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([1])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size(self.encoder_obj.output_shape)

    @staticmethod
    def update_config_with_metadata(
            input_feature, feature_metadata, *args, **kwargs
    ):
        pass

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, TIED, None)

    encoder_registry = Registry({
        'dense': DenseEncoder,
        **{key: PassthroughEncoder for key in DEFAULT_KEYS + ['passthrough']}
    })


class NumericalOutputFeature(NumericalFeatureMixin, OutputFeature):
    decoder = "regressor"
    loss = {TYPE: MEAN_SQUARED_ERROR}
    metric_functions = {
        LOSS: None,
        MEAN_SQUARED_ERROR: None,
        MEAN_ABSOLUTE_ERROR: None,
        ROOT_MEAN_SQUARED_ERROR: None,
        ROOT_MEAN_SQUARED_PERCENTAGE_ERROR: None,
        R2: None,
    }
    default_validation_metric = MEAN_SQUARED_ERROR
    clip = None

    def __init__(self, feature):
        super().__init__(feature)
        self.overwrite_defaults(feature)
        feature['input_size'] = self.input_shape[-1]
        self.decoder_obj = self.initialize_decoder(feature)
        self._setup_loss()
        self._setup_metrics()

    def logits(self, inputs, **kwargs):  # hidden
        hidden = inputs[HIDDEN]
        return self.decoder_obj(hidden)

    def predictions(self, inputs: Dict[str, torch.Tensor], feature_name: str, **kwargs):
        logits = output_feature_utils.get_output_feature_tensor(
            inputs, feature_name, LOGITS)
        predictions = logits

        if self.clip is not None:
            if isinstance(self.clip, (list, tuple)) and len(self.clip) == 2:
                predictions = torch.clamp(
                    logits,
                    self.clip[0],
                    self.clip[1]
                )

                logger.debug(
                    '  clipped_predictions: {0}'.format(predictions)
                )
            else:
                raise ValueError(
                    "The clip parameter of {} is {}. "
                    "It must be a list or a tuple of length 2.".format(
                        self.feature_name, self.clip
                    )
                )

        return {PREDICTIONS: predictions, LOGITS: logits}

    def get_prediction_set(self):
        return {PREDICTIONS, LOGITS}

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])

    @classmethod
    def get_output_dtype(cls):
        return torch.float32

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([1])

    @staticmethod
    def update_config_with_metadata(
            output_feature, feature_metadata, *args, **kwargs
    ):
        pass

    @staticmethod
    def calculate_overall_stats(predictions, targets, metadata):
        # no overall stats, just return empty dictionary
        return {}

    def postprocess_predictions(
            self,
            predictions,
            metadata,
            output_directory,
            backend,
    ):
        predictions_col = f"{self.feature_name}_{PREDICTIONS}"
        if predictions_col in predictions:
            # as needed convert predictions make to original value space
            numeric_transformer = get_from_registry(
                metadata["preprocessing"].get("normalization", None),
                numeric_transformation_registry,
            )(**metadata)
            predictions[predictions_col] = backend.df_engine.map_objects(
                predictions[predictions_col],
                lambda pred: numeric_transformer.inverse_transform(pred),
            )

        return predictions

    @staticmethod
    def populate_defaults(output_feature):
        set_default_value(
            output_feature, LOSS, {TYPE: "mean_squared_error", "weight": 1}
        )
        set_default_value(output_feature[LOSS], TYPE, "mean_squared_error")
        set_default_value(output_feature[LOSS], "weight", 1)

        set_default_values(
            output_feature,
            {
                "clip": None,
                "dependencies": [],
                "reduce_input": SUM,
                "reduce_dependencies": SUM,
            },
        )

    decoder_registry = {
        "regressor": Regressor,
        "null": Regressor,
        "none": Regressor,
        "None": Regressor,
        None: Regressor,
    }
