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

import numpy as np
import torch

from ludwig.constants import *
from ludwig.decoders.generic_decoders import Projector
from ludwig.encoders.generic_encoders import PassthroughEncoder, \
    DenseEncoder
from ludwig.features.base_feature import InputFeature
from ludwig.features.base_feature import OutputFeature
from ludwig.modules.loss_modules import SoftmaxCrossEntropyLoss, MSELoss, \
    MAELoss
from ludwig.modules.metric_modules import (
    SoftmaxCrossEntropyMetric, MSEMetric, MAEMetric
)
from ludwig.modules.metric_modules import R2Score
from ludwig.utils.misc_utils import set_default_value

logger = logging.getLogger(__name__)


class VectorFeatureMixin:
    type = VECTOR
    preprocessing_defaults = {
        'missing_value_strategy': FILL_WITH_CONST,
        'fill_value': '',
    }

    fill_value_schema = {
        "type": "string",
        "pattern": "^([0-9]+(\\.[0-9]*)?\\s*)*$"
    }

    preprocessing_schema = {
        'output_size': {'type': 'integer', 'minimum': 0},
        'missing_value_strategy': {'type': 'string',
                                   'enum': MISSING_VALUE_STRATEGY_OPTIONS},
        'fill_value': fill_value_schema,
        'computed_fill_value': fill_value_schema,
    }

    @staticmethod
    def cast_column(column, backend):
        return column

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters, backend):
        return {
            'preprocessing': preprocessing_parameters
        }

    @staticmethod
    def add_feature_data(
            feature,
            input_df,
            proc_df,
            metadata,
            preprocessing_parameters,
            backend,
            skip_save_processed_input
    ):
        """
                Expects all the vectors to be of the same size. The vectors need to be
                whitespace delimited strings. Missing values are not handled.
                """
        if len(input_df[feature[COLUMN]]) == 0:
            raise ValueError("There are no vectors in the dataset provided")

        # Convert the string of features into a numpy array
        try:
            proc_df[feature[PROC_COLUMN]] = backend.df_engine.map_objects(
                input_df[feature[COLUMN]],
                lambda x: np.array(x.split(), dtype=np.float32)
            )
        except ValueError:
            logger.error(
                'Unable to read the vector data. Make sure that all the vectors'
                ' are of the same size and do not have missing/null values.'
            )
            raise

        # Determine vector size
        output_size = backend.df_engine.compute(
            proc_df[feature[PROC_COLUMN]].map(len).max())
        if 'output_size' in preprocessing_parameters:
            if output_size != preprocessing_parameters['output_size']:
                raise ValueError(
                    'The user provided value for vector size ({}) does not '
                    'match the value observed in the data: {}'.format(
                        preprocessing_parameters, output_size
                    )
                )
        else:
            logger.debug('Observed vector size: {}'.format(output_size))

        metadata[feature[NAME]]['output_size'] = output_size
        return proc_df


class VectorInputFeature(VectorFeatureMixin, InputFeature):
    encoder = 'dense'
    output_size = 0

    def __init__(self, feature, encoder_obj=None):
        super().__init__(feature)
        self.overwrite_defaults(feature)
        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(feature)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert isinstance(inputs, torch.Tensor)
        assert inputs.dtype in [torch.float32, torch.float64]
        assert len(inputs.shape) == 2

        inputs_encoded = self.encoder_obj(
            inputs
        )

        return inputs_encoded

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.output_size])

    @staticmethod
    def update_config_with_metadata(
            input_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        for key in ['output_size']:
            input_feature[key] = feature_metadata[key]

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, TIED, None)
        set_default_value(input_feature, 'preprocessing', {})

    encoder_registry = {
        'dense': DenseEncoder,
        'passthrough': PassthroughEncoder,
        'null': PassthroughEncoder,
        'none': PassthroughEncoder,
        'None': PassthroughEncoder,
        None: PassthroughEncoder
    }


class VectorOutputFeature(VectorFeatureMixin, OutputFeature):
    decoder = 'projector'
    loss = {TYPE: MEAN_SQUARED_ERROR}
    metric_functions = {LOSS: None, ERROR: None, MEAN_SQUARED_ERROR: None,
                        MEAN_ABSOLUTE_ERROR: None, R2: None}
    default_validation_metric = MEAN_SQUARED_ERROR
    output_size = 0

    def __init__(self, feature):
        super().__init__(feature)
        self.overwrite_defaults(feature)
        self.decoder_obj = self.initialize_decoder(feature)
        self._setup_loss()
        self._setup_metrics()

    def logits(
            self,
            inputs,  # hidden
            **kwargs
    ):
        hidden = inputs[HIDDEN]
        return self.decoder_obj(hidden)

    def predictions(
            self,
            inputs,  # logits
            **kwargs
    ):
        return {PREDICTIONS: inputs[LOGITS], LOGITS: inputs[LOGITS]}

    def _setup_loss(self):
        if self.loss[TYPE] == 'mean_squared_error':
            self.train_loss_function = MSELoss()
            self.eval_loss_function = MSEMetric(name='eval_loss')
        elif self.loss[TYPE] == 'mean_absolute_error':
            self.train_loss_function = MAELoss()
            self.eval_loss_function = MAEMetric(name='eval_loss')
        elif self.loss[TYPE] == SOFTMAX_CROSS_ENTROPY:
            self.train_loss_function = SoftmaxCrossEntropyLoss(
                num_classes=self.output_size,
                feature_loss=self.loss,
                name='train_loss'
            )
            self.eval_loss_function = SoftmaxCrossEntropyMetric(
                num_classes=self.output_size,
                feature_loss=self.loss,
                name='eval_loss'
            )
        else:
            raise ValueError(
                'Unsupported loss type {}'.format(self.loss[TYPE])
            )

    def _setup_metrics(self):
        self.metric_functions = {}  # needed to shadow class variable
        self.metric_functions[LOSS] = self.eval_loss_function
        self.metric_functions[MEAN_SQUARED_ERROR] = MSEMetric(name='metric_mse')
        self.metric_functions[MEAN_ABSOLUTE_ERROR] = MAEMetric(name='metric_mae')
        self.metric_functions[R2] = R2Score(name='metric_r2')

    def get_prediction_set(self):
        return {
            PREDICTIONS, LOGITS
        }

    @classmethod
    def get_output_dtype(cls):
        return torch.float32

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.output_size])

    @staticmethod
    def update_config_with_metadata(
            output_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        output_feature['output_size'] = feature_metadata['output_size']

    @staticmethod
    def calculate_overall_stats(
            predictions,
            targets,
            train_set_metadata
    ):
        # no overall stats, just return empty dictionary
        return {}

    def postprocess_predictions(
            self,
            result,
            metadata,
            output_directory,
            backend,
    ):
        predictions_col = f'{self.feature_name}_{PREDICTIONS}'
        if predictions_col in result:
            result[predictions_col] = backend.df_engine.map_objects(
                result[predictions_col],
                lambda pred: pred.tolist()
            )
        return result

    @staticmethod
    def populate_defaults(output_feature):
        set_default_value(output_feature, LOSS, {})
        set_default_value(output_feature[LOSS], TYPE, MEAN_SQUARED_ERROR)
        set_default_value(output_feature[LOSS], 'weight', 1)
        set_default_value(output_feature, 'reduce_input', None)
        set_default_value(output_feature, 'reduce_dependencies', None)
        set_default_value(output_feature, 'decoder', 'projector')
        set_default_value(output_feature, 'dependencies', [])

    decoder_registry = {
        'projector': Projector,
        'null': Projector,
        'none': Projector,
        'None': Projector,
        None: Projector
    }
