#! /usr/bin/env python
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
import copy
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import torch
from torch import nn

from ludwig.constants import COLUMN, HIDDEN, LOGITS, NAME, NUMBER, PREDICTIONS, PROC_COLUMN
from ludwig.features.base_feature import BaseFeatureMixin, InputFeature, OutputFeature, PredictModule
from ludwig.schema.features.number_feature import NumberInputFeatureConfig, NumberOutputFeatureConfig
from ludwig.types import (
    FeatureMetadataDict,
    FeaturePostProcessingOutputDict,
    PreprocessingConfigDict,
    TrainingSetMetadataDict,
)
from ludwig.utils import output_feature_utils
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.types import TorchscriptPreprocessingInput

logger = logging.getLogger(__name__)


class NumberTransformer(nn.Module, ABC):
    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def transform_inference(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def inverse_transform_inference(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def fit_transform_params(column: np.ndarray, backend: Any) -> Dict[str, Any]:
        pass


class ZScoreTransformer(NumberTransformer):
    def __init__(self, mean: float = None, std: float = None, **kwargs: dict):
        super().__init__()
        self.mu = float(mean) if mean is not None else mean
        self.sigma = float(std) if std is not None else std
        self.feature_name = kwargs.get(NAME, "")
        if self.sigma == 0:
            raise RuntimeError(
                f"Cannot apply zscore normalization to `{self.feature_name}` since it has a standard deviation of 0. "
                f"This is most likely because `{self.feature_name}` has a constant value of {self.mu} for all rows in "
                "the dataset. Consider removing this feature from your Ludwig config since it is not useful for "
                "your machine learning model."
            )

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sigma

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.sigma + self.mu

    def transform_inference(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) / self.sigma

    def inverse_transform_inference(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigma + self.mu

    @staticmethod
    def fit_transform_params(column: np.ndarray, backend: "Backend") -> Dict[str, Any]:  # noqa
        compute = backend.df_engine.compute
        return {
            "mean": compute(column.astype(np.float32).mean()),
            "std": compute(column.astype(np.float32).std()),
        }


class MinMaxTransformer(NumberTransformer):
    def __init__(self, min: float = None, max: float = None, **kwargs: dict):
        super().__init__()
        self.min_value = float(min) if min is not None else min
        self.max_value = float(max) if max is not None else max
        if self.min_value is None or self.max_value is None:
            self.range = None
        else:
            self.range = self.max_value - self.min_value

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.min_value) / self.range

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.range is None:
            raise ValueError("Numeric transformer needs to be instantiated with " "min and max values.")
        return x * self.range + self.min_value

    def transform_inference(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.min_value) / self.range

    def inverse_transform_inference(self, x: torch.Tensor) -> torch.Tensor:
        if self.range is None:
            raise ValueError("Numeric transformer needs to be instantiated with " "min and max values.")
        return x * self.range + self.min_value

    @staticmethod
    def fit_transform_params(column: np.ndarray, backend: "Backend") -> Dict[str, Any]:  # noqa
        compute = backend.df_engine.compute
        return {
            "min": compute(column.astype(np.float32).min()),
            "max": compute(column.astype(np.float32).max()),
        }


class InterQuartileTransformer(NumberTransformer):
    def __init__(self, q1: float = None, q2: float = None, q3: float = None, **kwargs: dict):
        super().__init__()
        self.q1 = float(q1) if q1 is not None else q1
        self.q2 = float(q2) if q2 is not None else q2
        self.q3 = float(q3) if q3 is not None else q3
        if self.q1 is None or self.q3 is None:
            self.interquartile_range = None
        else:
            self.interquartile_range = self.q3 - self.q1
        self.feature_name = kwargs.get(NAME, "")
        if self.interquartile_range == 0:
            raise RuntimeError(
                f"Cannot apply InterQuartileNormalization to `{self.feature_name}` since"
                "the interquartile range is 0, which will result in a ZeroDivisionError."
            )

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.q2) / self.interquartile_range

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.interquartile_range + self.q2

    def transform_inference(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.q2) / self.interquartile_range

    def inverse_transform_inference(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.interquartile_range + self.q2

    @staticmethod
    def fit_transform_params(column: np.ndarray, backend: "Backend") -> Dict[str, Any]:  # noqa
        compute = backend.df_engine.compute
        return {
            "q1": compute(np.percentile(column.astype(np.float32), 25)),
            "q2": compute(np.percentile(column.astype(np.float32), 50)),
            "q3": compute(np.percentile(column.astype(np.float32), 75)),
        }


class Log1pTransformer(NumberTransformer):
    def __init__(self, **kwargs: dict):
        super().__init__()

    def transform(self, x: np.ndarray) -> np.ndarray:
        if np.any(x <= 0):
            raise ValueError(
                "One or more values are non-positive.  " "log1p normalization is defined only for positive values."
            )
        return np.log1p(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.expm1(x)

    def transform_inference(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(x)

    def inverse_transform_inference(self, x: torch.Tensor) -> torch.Tensor:
        return torch.expm1(x)

    @staticmethod
    def fit_transform_params(column: np.ndarray, backend: "Backend") -> Dict[str, Any]:  # noqa
        return {}


class IdentityTransformer(NumberTransformer):
    def __init__(self, **kwargs):
        super().__init__()

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def transform_inference(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse_transform_inference(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def fit_transform_params(column: np.ndarray, backend: "Backend") -> Dict[str, Any]:  # noqa
        return {}


numeric_transformation_registry = {
    "minmax": MinMaxTransformer,
    "zscore": ZScoreTransformer,
    "log1p": Log1pTransformer,
    "iq": InterQuartileTransformer,
    None: IdentityTransformer,
}


def get_transformer(metadata, preprocessing_parameters) -> NumberTransformer:
    return get_from_registry(
        preprocessing_parameters.get("normalization", None),
        numeric_transformation_registry,
    )(**metadata)


class _OutlierReplacer(torch.nn.Module):
    def __init__(self, metadata: TrainingSetMetadataDict):
        super().__init__()
        self.zscore_transformer = ZScoreTransformer(**metadata)
        self.outlier_threshold = metadata["preprocessing"].get("outlier_threshold")
        self.computed_outlier_fill_value = float(metadata["preprocessing"]["computed_outlier_fill_value"])

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        outliers = self.zscore_transformer.transform_inference(v).abs().gt(self.outlier_threshold)
        v_masked = torch.masked_fill(v, outliers, torch.nan)

        v = torch.nan_to_num(v_masked, nan=self.computed_outlier_fill_value)
        return v.to(dtype=torch.float32)


class _NumberPreprocessing(torch.nn.Module):
    def __init__(self, metadata: TrainingSetMetadataDict):
        super().__init__()
        self.computed_fill_value = float(metadata["preprocessing"]["computed_fill_value"])
        self.numeric_transformer = get_transformer(metadata, metadata["preprocessing"])

        # Optional outlier replacement
        self.outlier_replacer = None
        if metadata["preprocessing"].get("outlier_strategy") is not None:
            self.outlier_replacer = _OutlierReplacer(metadata)

    def forward(self, v: TorchscriptPreprocessingInput) -> torch.Tensor:
        if not torch.jit.isinstance(v, torch.Tensor):
            raise ValueError(f"Unsupported input: {v}")

        v = torch.nan_to_num(v, nan=self.computed_fill_value)
        v = v.to(dtype=torch.float32)

        # Handle outliers if needed
        if self.outlier_replacer is not None:
            v = self.outlier_replacer(v)

        return self.numeric_transformer.transform_inference(v)


class _NumberPostprocessing(torch.nn.Module):
    def __init__(self, metadata: TrainingSetMetadataDict):
        super().__init__()
        self.numeric_transformer = get_transformer(metadata, metadata["preprocessing"])
        self.predictions_key = PREDICTIONS

    def forward(self, preds: Dict[str, torch.Tensor], feature_name: str) -> FeaturePostProcessingOutputDict:
        predictions = output_feature_utils.get_output_feature_tensor(preds, feature_name, self.predictions_key)

        return {self.predictions_key: self.numeric_transformer.inverse_transform_inference(predictions)}


class _NumberPredict(PredictModule):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, inputs: Dict[str, torch.Tensor], feature_name: str) -> Dict[str, torch.Tensor]:
        logits = output_feature_utils.get_output_feature_tensor(inputs, feature_name, self.logits_key)
        predictions = logits

        if self.clip is not None:
            predictions = torch.clamp(logits, self.clip[0], self.clip[1])
            logger.debug(f"  clipped_predictions: {predictions}")

        return {self.predictions_key: predictions, self.logits_key: logits}


class NumberFeatureMixin(BaseFeatureMixin):
    @staticmethod
    def type():
        return NUMBER

    @staticmethod
    def cast_column(column, backend):
        return backend.df_engine.df_lib.to_numeric(column, errors="coerce").astype(np.float32)

    @staticmethod
    def get_feature_meta(
        column, preprocessing_parameters: PreprocessingConfigDict, backend, is_input_feature: bool
    ) -> FeatureMetadataDict:
        numeric_transformer: NumberTransformer = get_from_registry(
            preprocessing_parameters.get("normalization", None),
            numeric_transformation_registry,
        )

        params = numeric_transformer.fit_transform_params(column, backend)

        # Ensure mean and std are computed if we're removing outliers
        outlier_strategy = preprocessing_parameters.get("outlier_strategy")
        if outlier_strategy is not None and ("mean" not in params or "std" not in params):
            params.update(ZScoreTransformer.fit_transform_params(column, backend))

        return params

    @staticmethod
    def add_feature_data(
        feature_config,
        input_df,
        proc_df,
        metadata,
        preprocessing_parameters: PreprocessingConfigDict,
        backend,
        skip_save_processed_input,
    ):
        # Had to replace normalize() function due to issue #1911
        # this comment is to provide context for the change.
        # original code
        # def normalize(series: pd.Series) -> pd.Series:
        #     series = series.copy()
        #     numeric_transformer = get_transformer(metadata[feature_config[NAME]], preprocessing_parameters)
        #     series.update(numeric_transformer.transform(series.values))
        #     return series

        def normalize(series: pd.Series) -> pd.Series:
            _feature_metadata = copy.deepcopy(metadata[feature_config[NAME]])
            _feature_metadata.update({NAME: feature_config[NAME]})

            # retrieve request numeric transformer
            numeric_transformer = get_transformer(_feature_metadata, preprocessing_parameters)

            # transform input numeric values with specified transformer
            transformed_values = numeric_transformer.transform(series.values)

            # return transformed values with same index values as original series.
            return pd.Series(transformed_values, index=series.index)

        input_series = input_df[feature_config[COLUMN]].astype(np.float32)
        proc_df[feature_config[PROC_COLUMN]] = backend.df_engine.map_partitions(
            input_series, normalize, meta=input_series
        )

        return proc_df


class NumberInputFeature(NumberFeatureMixin, InputFeature):
    def __init__(self, input_feature_config: NumberInputFeatureConfig, encoder_obj=None, **kwargs):
        super().__init__(input_feature_config, **kwargs)
        input_feature_config.encoder.input_size = self.input_shape[-1]

        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(input_feature_config.encoder)

    def forward(self, inputs):
        assert isinstance(inputs, torch.Tensor)
        assert inputs.dtype == torch.float32 or inputs.dtype == torch.float64
        assert len(inputs.shape) == 1 or (len(inputs.shape) == 2 and inputs.shape[1] == 1)

        if len(inputs.shape) == 1:
            inputs = inputs[:, None]
        inputs_encoded = self.encoder_obj(inputs)

        return inputs_encoded

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([1])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size(self.encoder_obj.output_shape)

    @staticmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        pass

    @staticmethod
    def get_schema_cls():
        return NumberInputFeatureConfig

    def create_sample_input(self, batch_size: int = 2):
        return torch.rand([batch_size])

    @classmethod
    def get_preproc_input_dtype(cls, metadata: TrainingSetMetadataDict) -> str:
        return "float32"

    @staticmethod
    def create_preproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
        return _NumberPreprocessing(metadata)


class NumberOutputFeature(NumberFeatureMixin, OutputFeature):
    def __init__(
        self,
        output_feature_config: Union[NumberOutputFeatureConfig, Dict],
        output_features: Dict[str, OutputFeature],
        **kwargs,
    ):
        self.clip = output_feature_config.clip
        super().__init__(output_feature_config, output_features, **kwargs)
        self.decoder_obj = self.initialize_decoder(output_feature_config.decoder)
        self._setup_loss()
        self._setup_metrics()

    def logits(self, inputs, **kwargs):  # hidden
        hidden = inputs[HIDDEN]
        return self.decoder_obj(hidden)

    def create_predict_module(self) -> PredictModule:
        if getattr(self, "clip", None) and not (isinstance(self.clip, (list, tuple)) and len(self.clip) == 2):
            raise ValueError(
                f"The clip parameter of {self.feature_name} is {self.clip}. "
                f"It must be a list or a tuple of length 2."
            )
        return _NumberPredict(getattr(self, "clip", None))

    def get_prediction_set(self):
        return {PREDICTIONS, LOGITS}

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.decoder_obj.config.input_size])

    @classmethod
    def get_output_dtype(cls):
        return torch.float32

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([1])

    @staticmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        pass

    @staticmethod
    def calculate_overall_stats(predictions, targets, metadata):
        # no overall stats, just return empty dictionary
        return {}

    def postprocess_predictions(
        self,
        predictions,
        metadata,
    ):
        predictions_col = f"{self.feature_name}_{PREDICTIONS}"
        if predictions_col in predictions:
            # as needed convert predictions make to original value space
            numeric_transformer = get_from_registry(
                metadata["preprocessing"].get("normalization", None),
                numeric_transformation_registry,
            )(**metadata)
            predictions[predictions_col] = predictions[predictions_col].map(
                lambda pred: numeric_transformer.inverse_transform(pred)
            )

        return predictions

    @staticmethod
    def get_schema_cls():
        return NumberOutputFeatureConfig

    @classmethod
    def get_postproc_output_dtype(cls, metadata: TrainingSetMetadataDict) -> str:
        return "float32"

    @staticmethod
    def create_postproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
        return _NumberPostprocessing(metadata)
