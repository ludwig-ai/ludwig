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
from typing import Union, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import (
    MeanAbsoluteError as MeanAbsoluteErrorMetric,
)
from tensorflow.keras.metrics import (
    MeanSquaredError as MeanSquaredErrorMetric,
    RootMeanSquaredError as RootMeanSquaredErrorMetric,
)

from ludwig.constants import *
from ludwig.decoders.generic_decoders import Regressor
from ludwig.encoders.generic_encoders import PassthroughEncoder, DenseEncoder
from ludwig.features.base_feature import InputFeature
from ludwig.features.base_feature import OutputFeature
from ludwig.modules.loss_modules import MSELoss, MAELoss, RMSELoss, RMSPELoss
from ludwig.modules.metric_modules import (
    MAEMetric,
    MSEMetric,
    RMSEMetric,
    RMSPEMetric,
    R2Score,
)
from ludwig.utils.misc_utils import set_default_value
from ludwig.utils.misc_utils import set_default_values
from ludwig.utils.misc_utils import get_from_registry

logger = logging.getLogger(__name__)


class ZScoreTransformer:
    def __init__(self, mean: float = None, std: float = None, **kwargs: dict):
        self.mu = mean
        self.sigma = std

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self._transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return self._inverse_transform(x)

    def transform_inference(self, x: tf.Tensor):
        return self._transform(x)

    def inverse_transform_inference(self, x: tf.Tensor):
        return self._inverse_transform(x)

    def _transform(self, x: Union[np.ndarray, tf.Tensor]):
        return (x - self.mu) / self.sigma

    def _inverse_transform(self, x: Union[np.ndarray, tf.Tensor]):
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
        return self._transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return self._inverse_transform(x)

    def transform_inference(self, x: tf.Tensor):
        return self._transform(x)

    def inverse_transform_inference(self, x: tf.Tensor):
        return self._inverse_transform(x)

    def _transform(self, x: Union[np.ndarray, tf.Tensor]):
        return (x - self.min_value) / self.range

    def _inverse_transform(self, x: Union[np.ndarray, tf.Tensor]):
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

    def transform_inference(self, x: tf.Tensor):
        return tf.math.log1p(x)

    def inverse_transform_inference(self, x: tf.Tensor):
        return tf.math.expm1(x)

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

    def transform_inference(self, x: tf.Tensor):
        return x

    def inverse_transform_inference(self, x: tf.Tensor):
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


def get_transformer(metadata, preprocessing_parameters):
    return get_from_registry(
        preprocessing_parameters.get("normalization", None),
        numeric_transformation_registry,
    )(**metadata)


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
        numeric_transformer = get_transformer(
            metadata[feature[NAME]], preprocessing_parameters
        )
        proc_df[feature[PROC_COLUMN]] = numeric_transformer.transform(
            proc_df[feature[PROC_COLUMN]]
        )

        return proc_df

    @staticmethod
    def preprocess_inference_graph(t: tf.Tensor, metadata: dict):
        numeric_transformer = get_transformer(
            metadata, metadata["preprocessing"]
        )
        return numeric_transformer.transform_inference(t)


class NumericalInputFeature(NumericalFeatureMixin, InputFeature):
    encoder = "passthrough"

    def __init__(self, feature, encoder_obj=None):
        super().__init__(feature)
        self.overwrite_defaults(feature)
        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(feature)

    def call(self, inputs, training=None, mask=None):
        assert isinstance(inputs, tf.Tensor)
        assert inputs.dtype == tf.float32 or inputs.dtype == tf.float64
        assert len(inputs.shape) == 1

        inputs_exp = inputs[:, tf.newaxis]
        inputs_encoded = self.encoder_obj(
            inputs_exp, training=training, mask=mask
        )

        return inputs_encoded

    @classmethod
    def get_input_dtype(cls):
        return tf.float32

    def get_input_shape(self):
        return ()

    @classmethod
    def get_inference_dtype(cls):
        return tf.float32

    @classmethod
    def get_inference_shape(cls):
        return ()

    @staticmethod
    def update_config_with_metadata(
            input_feature, feature_metadata, *args, **kwargs
    ):
        pass

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, TIED, None)

    encoder_registry = {
        "dense": DenseEncoder,
        "passthrough": PassthroughEncoder,
        "null": PassthroughEncoder,
        "none": PassthroughEncoder,
        "None": PassthroughEncoder,
        None: PassthroughEncoder,
    }


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
        self.decoder_obj = self.initialize_decoder(feature)
        self._setup_loss()
        self._setup_metrics()

    def logits(self, inputs, **kwargs):  # hidden
        hidden = inputs[HIDDEN]
        return self.decoder_obj(hidden)

    def predictions(self, inputs, **kwargs):  # logits
        logits = inputs[LOGITS]
        predictions = logits

        if self.clip is not None:
            if isinstance(self.clip, (list, tuple)) and len(self.clip) == 2:
                predictions = tf.clip_by_value(
                    predictions, self.clip[0], self.clip[1]
                )
                logger.debug("  clipped_predictions: {0}".format(predictions))
            else:
                raise ValueError(
                    "The clip parameter of {} is {}. "
                    "It must be a list or a tuple of length 2.".format(
                        self.feature_name, self.clip
                    )
                )

        return {PREDICTIONS: predictions, LOGITS: logits}

    def _setup_loss(self):
        if self.loss[TYPE] == "mean_squared_error":
            self.train_loss_function = MSELoss()
        elif self.loss[TYPE] == "mean_absolute_error":
            self.train_loss_function = MAELoss()
        elif self.loss[TYPE] == "root_mean_squared_error":
            self.train_loss_function = RMSELoss()
        elif self.loss[TYPE] == "root_mean_squared_percentage_error":
            self.train_loss_function = RMSPELoss()
        else:
            raise ValueError(
                "Unsupported loss type {}".format(self.loss[TYPE])
            )

        self.eval_loss_function = self.train_loss_function

    def _setup_metrics(self):
        self.metric_functions = {}  # needed to shadow class variable
        if self.loss[TYPE] == "mean_squared_error":
            self.metric_functions[LOSS] = MSEMetric(name="eval_loss")
        elif self.loss[TYPE] == "mean_absolute_error":
            self.metric_functions[LOSS] = MAEMetric(name="eval_loss")
        elif self.loss[TYPE] == "root_mean_squared_error":
            self.metric_functions[LOSS] = RMSEMetric(name="eval_loss")
        elif self.loss[TYPE] == "root_mean_squared_percentage_error":
            self.metric_functions[LOSS] = RMSPEMetric(name="eval_loss")

        self.metric_functions[MEAN_SQUARED_ERROR] = MeanSquaredErrorMetric(
            name="metric_mse"
        )
        self.metric_functions[MEAN_ABSOLUTE_ERROR] = MeanAbsoluteErrorMetric(
            name="metric_mae"
        )
        self.metric_functions[
            ROOT_MEAN_SQUARED_ERROR
        ] = RootMeanSquaredErrorMetric(name="metric_rmse")
        self.metric_functions[
            ROOT_MEAN_SQUARED_PERCENTAGE_ERROR
        ] = RMSPEMetric(name="metric_rmspe")
        self.metric_functions[R2] = R2Score(name="metric_r2")

    def get_prediction_set(self):
        return {PREDICTIONS, LOGITS}

    @classmethod
    def get_output_dtype(cls):
        return tf.float32

    def get_output_shape(self):
        return ()

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
            numeric_transformer = get_transformer(
                metadata, metadata["preprocessing"]
            )
            predictions[predictions_col] = backend.df_engine.map_objects(
                predictions[predictions_col],
                lambda pred: numeric_transformer.inverse_transform(pred),
            )

        return predictions

    def postprocess_inference_graph(self, preds: Dict[str, tf.Tensor], metadata: dict):
        numeric_transformer = get_transformer(
            metadata, metadata["preprocessing"]
        )
        return {
            PREDICTIONS: numeric_transformer.inverse_transform_inference(preds[PREDICTIONS])
        }

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
