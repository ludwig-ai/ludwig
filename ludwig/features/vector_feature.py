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
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ludwig.constants import (
    COLUMN,
    DECODER,
    ENCODER,
    ERROR,
    FILL_WITH_CONST,
    HIDDEN,
    LOGITS,
    LOSS,
    MEAN_ABSOLUTE_ERROR,
    MEAN_SQUARED_ERROR,
    NAME,
    PREDICTIONS,
    PROC_COLUMN,
    R2,
    TIED,
    TYPE,
    VECTOR,
)
from ludwig.features.base_feature import InputFeature, OutputFeature, PredictModule
from ludwig.schema.features.utils import register_input_feature, register_output_feature
from ludwig.schema.features.vector_feature import VectorInputFeatureConfig, VectorOutputFeatureConfig
from ludwig.utils import output_feature_utils
from ludwig.utils.misc_utils import set_default_value, set_default_values
from ludwig.utils.torch_utils import LudwigModule
from ludwig.utils.types import TorchscriptPreprocessingInput

logger = logging.getLogger(__name__)


class _VectorPreprocessing(torch.nn.Module):
    def forward(self, v: TorchscriptPreprocessingInput) -> torch.Tensor:
        if torch.jit.isinstance(v, torch.Tensor):
            out = v
        elif torch.jit.isinstance(v, List[torch.Tensor]):
            out = torch.stack(v)
        elif torch.jit.isinstance(v, List[str]):
            vectors = []
            for sample in v:
                vector = torch.tensor([float(x) for x in sample.split()], dtype=torch.float32)
                vectors.append(vector)
            out = torch.stack(vectors)
        else:
            raise ValueError(f"Unsupported input: {v}")

        if out.isnan().any():
            raise ValueError("Scripted NaN handling not implemented for Vector feature")
        return out


class _VectorPostprocessing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.predictions_key = PREDICTIONS
        self.logits_key = LOGITS

    def forward(self, preds: Dict[str, torch.Tensor], feature_name: str) -> Dict[str, Any]:
        predictions = output_feature_utils.get_output_feature_tensor(preds, feature_name, self.predictions_key)
        logits = output_feature_utils.get_output_feature_tensor(preds, feature_name, self.logits_key)

        return {self.predictions_key: predictions, self.logits_key: logits}


class _VectorPredict(PredictModule):
    def forward(self, inputs: Dict[str, torch.Tensor], feature_name: str) -> Dict[str, torch.Tensor]:
        logits = output_feature_utils.get_output_feature_tensor(inputs, feature_name, self.logits_key)

        return {self.predictions_key: logits, self.logits_key: logits}


class VectorFeatureMixin:
    @staticmethod
    def type():
        return VECTOR

    @staticmethod
    def preprocessing_defaults():
        return {
            "missing_value_strategy": FILL_WITH_CONST,
            "fill_value": "",
        }

    @staticmethod
    def cast_column(column, backend):
        return column

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters, backend):
        return {"preprocessing": preprocessing_parameters}

    @staticmethod
    def add_feature_data(
        feature_config, input_df, proc_df, metadata, preprocessing_parameters, backend, skip_save_processed_input
    ):
        """Expects all the vectors to be of the same size.

        The vectors need to be whitespace delimited strings. Missing values are not handled.
        """
        if len(input_df[feature_config[COLUMN]]) == 0:
            raise ValueError("There are no vectors in the dataset provided")

        # Convert the string of features into a numpy array
        try:
            proc_df[feature_config[PROC_COLUMN]] = backend.df_engine.map_objects(
                input_df[feature_config[COLUMN]], lambda x: np.array(x.split(), dtype=np.float32)
            )
        except ValueError:
            logger.error(
                "Unable to read the vector data. Make sure that all the vectors"
                " are of the same size and do not have missing/null values."
            )
            raise

        # Determine vector size
        vector_size = backend.df_engine.compute(proc_df[feature_config[PROC_COLUMN]].map(len).max())
        if "vector_size" in preprocessing_parameters:
            if vector_size != preprocessing_parameters["vector_size"]:
                raise ValueError(
                    "The user provided value for vector size ({}) does not "
                    "match the value observed in the data: {}".format(preprocessing_parameters, vector_size)
                )
        else:
            logger.debug(f"Observed vector size: {vector_size}")

        metadata[feature_config[NAME]]["vector_size"] = vector_size
        return proc_df


@register_input_feature(VECTOR)
class VectorInputFeature(VectorFeatureMixin, InputFeature):
    encoder = {TYPE: "dense"}
    vector_size = 0

    def __init__(self, feature: Dict[str, Any], encoder_obj: Optional[LudwigModule] = None):
        super().__init__(feature)
        self.overwrite_defaults(feature)
        self.encoder["input_size"] = feature["vector_size"]
        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert isinstance(inputs, torch.Tensor)
        assert inputs.dtype in [torch.float32, torch.float64]
        assert len(inputs.shape) == 2

        inputs_encoded = self.encoder_obj(inputs)

        return inputs_encoded

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.vector_size])

    @property
    def output_shape(self) -> torch.Size:
        return self.encoder_obj.output_shape

    @staticmethod
    def update_config_with_metadata(input_feature, feature_metadata, *args, **kwargs):
        for key in ["vector_size"]:
            input_feature[key] = feature_metadata[key]

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, TIED, None)
        set_default_value(input_feature, "preprocessing", {})
        set_default_values(input_feature, {ENCODER: {TYPE: "dense"}})

    @staticmethod
    def create_preproc_module(metadata: Dict[str, Any]) -> torch.nn.Module:
        return _VectorPreprocessing()

    @staticmethod
    def get_schema_cls():
        return VectorInputFeatureConfig


@register_output_feature(VECTOR)
class VectorOutputFeature(VectorFeatureMixin, OutputFeature):
    decoder = {TYPE: "projector"}
    loss = {TYPE: MEAN_SQUARED_ERROR}
    metric_functions = {LOSS: None, ERROR: None, MEAN_SQUARED_ERROR: None, MEAN_ABSOLUTE_ERROR: None, R2: None}
    default_validation_metric = MEAN_SQUARED_ERROR
    vector_size = 0

    def __init__(self, feature, output_features: Dict[str, OutputFeature]):
        super().__init__(feature, output_features)
        self.overwrite_defaults(feature)
        self._input_shape = feature[DECODER]["input_size"]
        self.decoder["output_size"] = feature["vector_size"]
        self.decoder_obj = self.initialize_decoder()
        self._setup_loss()
        self._setup_metrics()

    def logits(self, inputs, **kwargs):  # hidden
        hidden = inputs[HIDDEN]
        return self.decoder_obj(hidden)

    def loss_kwargs(self):
        return self.loss

    def metric_kwargs(self):
        return dict(num_outputs=self.output_shape[0])

    def create_predict_module(self) -> PredictModule:
        return _VectorPredict()

    def get_prediction_set(self):
        return {PREDICTIONS, LOGITS}

    @classmethod
    def get_output_dtype(cls):
        return torch.float32

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.vector_size])

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self._input_shape])

    @staticmethod
    def update_config_with_metadata(output_feature, feature_metadata, *args, **kwargs):
        output_feature["vector_size"] = feature_metadata["vector_size"]

    @staticmethod
    def calculate_overall_stats(predictions, targets, train_set_metadata):
        # no overall stats, just return empty dictionary
        return {}

    def postprocess_predictions(
        self,
        result,
        metadata,
    ):
        predictions_col = f"{self.feature_name}_{PREDICTIONS}"
        if predictions_col in result:
            result[predictions_col] = result[predictions_col].map(lambda pred: pred.tolist())
        return result

    @staticmethod
    def populate_defaults(output_feature):
        set_default_value(output_feature, LOSS, {})
        set_default_value(output_feature[LOSS], TYPE, MEAN_SQUARED_ERROR)
        set_default_value(output_feature[LOSS], "weight", 1)
        set_default_value(output_feature, "reduce_input", None)
        set_default_value(output_feature, "reduce_dependencies", None)
        set_default_values(output_feature, {DECODER: {TYPE: "projector"}})
        set_default_value(output_feature, "dependencies", [])

    @staticmethod
    def create_postproc_module(metadata: Dict[str, Any]) -> torch.nn.Module:
        return _VectorPostprocessing()

    @staticmethod
    def get_schema_cls():
        return VectorOutputFeatureConfig
