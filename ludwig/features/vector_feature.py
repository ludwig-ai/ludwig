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
import os

import numpy as np

import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import \
    MeanAbsoluteError as MeanAbsoluteErrorMetric
from tensorflow.keras.metrics import MeanSquaredError as MeanSquaredErrorMetric

from ludwig.constants import *
from ludwig.features.base_feature import BaseFeature
from ludwig.features.base_feature import InputFeature
from ludwig.features.base_feature import OutputFeature
from ludwig.models.modules.generic_decoders import Projector
from ludwig.models.modules.generic_encoders import PassthroughEncoder, \
    DenseEncoder
from ludwig.models.modules.loss_modules import SoftmaxCrossEntropyLoss
from ludwig.models.modules.metric_modules import ErrorScore, \
    SoftmaxCrossEntropyMetric
from ludwig.models.modules.metric_modules import R2Score
from ludwig.utils.misc import set_default_value

logger = logging.getLogger(__name__)


# TODO TF2 can we eliminate use of these customer wrapper classes?
#  These are copies of the classes in numerical_modules,
#  depending on what we end up doing with those, these will follow
# custom class to handle how Ludwig stores predictions
class MSELoss(MeanSquaredError):
    def __init__(self, **kwargs):
        super(MSELoss, self).__init__(**kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        logits = y_pred[LOGITS]
        loss = super().__call__(y_true, logits, sample_weight=sample_weight)
        return loss


class MSEMetric(MeanSquaredErrorMetric):
    def __init__(self, **kwargs):
        super(MSEMetric, self).__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(
            y_true, y_pred['predictions'], sample_weight=sample_weight
        )


class MAELoss(MeanAbsoluteError):
    def __init__(self, **kwargs):
        super(MAELoss, self).__init__(**kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        logits = y_pred[LOGITS]
        loss = super().__call__(y_true, logits, sample_weight=sample_weight)
        return loss


class MAEMetric(MeanAbsoluteErrorMetric):
    def __init__(self, **kwargs):
        super(MAEMetric, self).__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(
            y_true, y_pred['predictions'], sample_weight=sample_weight
        )


class VectorBaseFeature(BaseFeature):
    type = VECTOR
    preprocessing_defaults = {
        'missing_value_strategy': FILL_WITH_CONST,
        'fill_value': ""
    }

    def __init__(self, feature):
        super().__init__(feature)

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters):
        return {
            'preprocessing': preprocessing_parameters
        }

    @staticmethod
    def add_feature_data(
            feature,
            dataset_df,
            data,
            metadata,
            preprocessing_parameters,
    ):
        """
                Expects all the vectors to be of the same size. The vectors need to be
                whitespace delimited strings. Missing values are not handled.
                """
        if len(dataset_df) == 0:
            raise ValueError("There are no vectors in the dataset provided")

        # Convert the string of features into a numpy array
        try:
            data[feature['name']] = np.array(
                [x.split() for x in dataset_df[feature['name']]],
                dtype=np.double
            )
        except ValueError:
            logger.error(
                'Unable to read the vector data. Make sure that all the vectors'
                ' are of the same size and do not have missing/null values.'
            )
            raise

        # Determine vector size
        vector_size = len(data[feature['name']][0])
        if 'vector_size' in preprocessing_parameters:
            if vector_size != preprocessing_parameters['vector_size']:
                raise ValueError(
                    'The user provided value for vector size ({}) does not '
                    'match the value observed in the data: {}'.format(
                        preprocessing_parameters, vector_size
                    )
                )
        else:
            logger.warning('Observed vector size: {}'.format(vector_size))

        metadata[feature['name']]['vector_size'] = vector_size


class VectorInputFeature(VectorBaseFeature, InputFeature):
    encoder = 'dense'

    def __init__(self, feature, encoder_obj=None):
        VectorBaseFeature.__init__(self, feature)
        InputFeature.__init__(self)
        self.overwrite_defaults(feature)
        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(feature)

    def call(self, inputs, training=None, mask=None):
        assert isinstance(inputs, tf.Tensor)
        assert inputs.dtype == tf.float32 or inputs.dtype == tf.float64
        assert len(inputs.shape) == 2

        inputs_encoded = self.encoder_obj(
            inputs, training=training, mask=mask
        )

        return {'encoder_outputs': inputs_encoded}

    @staticmethod
    def update_model_definition_with_metadata(
            input_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        for key in ['vector_size']:
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


class VectorOutputFeature(VectorBaseFeature, OutputFeature):
    decoder = 'projector'
    loss = {'type': MEAN_SQUARED_ERROR}
    vector_size = 0

    def __init__(self, feature):
        VectorBaseFeature.__init__(self, feature)
        OutputFeature.__init__(self, feature)
        self.overwrite_defaults(feature)
        self.decoder_obj = self.initialize_decoder(feature)
        self._setup_loss()
        self._setup_metrics()

    def logits(
            self,
            inputs,  # hidden
    ):
        return self.decoder_obj(inputs)

    def predictions(
            self,
            inputs,  # logits
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
                num_classes=self.vector_size,
                feature_loss=self.loss,
                name='train_loss'
            )
            self.eval_loss_function = SoftmaxCrossEntropyMetric(
                num_classes=self.vector_size,
                feature_loss=self.loss,
                name='eval_loss'
            )
        else:
            raise ValueError(
                'Unsupported loss type {}'.format(self.loss[TYPE])
            )

    def _setup_metrics(self):
        self.metric_functions[LOSS] = self.eval_loss_function
        self.metric_functions[ERROR] = ErrorScore(name='metric_error')
        self.metric_functions[MEAN_SQUARED_ERROR] = MeanSquaredErrorMetric(
            name='metric_mse'
        )
        self.metric_functions[MEAN_ABSOLUTE_ERROR] = MeanAbsoluteErrorMetric(
            name='metric_mae'
        )
        self.metric_functions[R2] = R2Score(name='metric_r2')

    default_validation_metric = MEAN_SQUARED_ERROR

    @staticmethod
    def update_model_definition_with_metadata(
            output_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        output_feature['vector_size'] = feature_metadata['vector_size']

    @staticmethod
    def calculate_overall_stats(
            test_stats,
            output_feature,
            dataset,
            train_set_metadata
    ):
        pass

    @staticmethod
    def postprocess_results(
            output_feature,
            result,
            metadata,
            experiment_dir_name,
            skip_save_unprocessed_output=False,
    ):
        postprocessed = {}
        npy_filename = os.path.join(experiment_dir_name, '{}_{}.npy')
        name = output_feature['name']

        if PREDICTIONS in result and len(result[PREDICTIONS]) > 0:
            postprocessed[PREDICTIONS] = result[PREDICTIONS]
            if not skip_save_unprocessed_output:
                np.save(
                    npy_filename.format(name, PREDICTIONS),
                    result[PREDICTIONS]
                )
            del result[PREDICTIONS]

        return postprocessed

    @staticmethod
    def populate_defaults(output_feature):
        set_default_value(output_feature, LOSS, {})
        set_default_value(output_feature[LOSS], 'type', MEAN_SQUARED_ERROR)
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
