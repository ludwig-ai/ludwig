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
from typing import Any, Dict, Optional

import numpy as np
import torch

from ludwig.constants import (
    COLUMN,
    ERROR,
    FILL_WITH_CONST,
    HIDDEN,
    LOGITS,
    LOSS,
    MEAN_ABSOLUTE_ERROR,
    MEAN_SQUARED_ERROR,
    MISSING_VALUE_STRATEGY_OPTIONS,
    NAME,
    PREDICTIONS,
    PROC_COLUMN,
    R2,
    TIED,
    TYPE,
    VECTOR,
)
from ludwig.features.base_feature import InputFeature, OutputFeature, PredictModule
from ludwig.utils import output_feature_utils
from ludwig.utils.misc_utils import set_default_value
from ludwig.utils.torch_utils import LudwigModule

logger = logging.getLogger(__name__)


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
    def preprocessing_schema():
        fill_value_schema = {"type": "string", "pattern": "^([0-9]+(\\.[0-9]*)?\\s*)*$"}
        return {
            "vector_size": {"type": "integer", "minimum": 0},
            "missing_value_strategy": {"type": "string", "enum": MISSING_VALUE_STRATEGY_OPTIONS},
            "fill_value": fill_value_schema,
            "computed_fill_value": fill_value_schema,
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


class VectorInputFeature(VectorFeatureMixin, InputFeature):
    encoder = "dense"
    vector_size = 0

    def __init__(self, feature: Dict[str, Any], encoder_obj: Optional[LudwigModule] = None):
        super().__init__(feature)
        self.overwrite_defaults(feature)
        feature["input_size"] = feature["vector_size"]
        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(feature)

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


class VectorOutputFeature(VectorFeatureMixin, OutputFeature):
    decoder = "projector"
    loss = {TYPE: MEAN_SQUARED_ERROR}
    metric_functions = {LOSS: None, ERROR: None, MEAN_SQUARED_ERROR: None, MEAN_ABSOLUTE_ERROR: None, R2: None}
    default_validation_metric = MEAN_SQUARED_ERROR
    vector_size = 0

    def __init__(self, feature, output_features: Dict[str, OutputFeature]):
        super().__init__(feature, output_features)
        self.overwrite_defaults(feature)
        self._input_shape = feature["input_size"]
        feature["output_size"] = feature["vector_size"]
        self.decoder_obj = self.initialize_decoder(feature)
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
        output_directory,
        backend,
    ):
        predictions_col = f"{self.feature_name}_{PREDICTIONS}"
        if predictions_col in result:
            result[predictions_col] = backend.df_engine.map_objects(result[predictions_col], lambda pred: pred.tolist())
        return result

    @staticmethod
    def populate_defaults(output_feature):
        set_default_value(output_feature, LOSS, {})
        set_default_value(output_feature[LOSS], TYPE, MEAN_SQUARED_ERROR)
        set_default_value(output_feature[LOSS], "weight", 1)
        set_default_value(output_feature, "reduce_input", None)
        set_default_value(output_feature, "reduce_dependencies", None)
        set_default_value(output_feature, "decoder", "projector")
        set_default_value(output_feature, "dependencies", [])
