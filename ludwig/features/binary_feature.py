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
import tensorflow.compat.v1 as tf
from tensorflow.keras.metrics import Accuracy as BinaryAccuracy

from ludwig.constants import *
from ludwig.features.base_feature import BaseFeature
from ludwig.features.base_feature import InputFeature
from ludwig.features.base_feature import OutputFeature
from ludwig.models.modules.binary_decoders import Regressor
from ludwig.models.modules.binary_encoders import BinaryPassthroughEncoder
from ludwig.models.modules.fully_connected_modules import FCStack
from ludwig.models.modules.loss_modules import BWCEWLoss
from ludwig.models.modules.metric_modules import BWCEWLMetric
from ludwig.utils.metrics_utils import ConfusionMatrix
from ludwig.utils.metrics_utils import average_precision_score
from ludwig.utils.metrics_utils import precision_recall_curve
from ludwig.utils.metrics_utils import roc_auc_score
from ludwig.utils.metrics_utils import roc_curve
from ludwig.utils.misc import set_default_value, get_from_registry
from ludwig.utils.misc import set_default_values

logger = logging.getLogger(__name__)


class BinaryBaseFeature(BaseFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = BINARY

    preprocessing_defaults = {
        'missing_value_strategy': FILL_WITH_CONST,
        'fill_value': 0
    }

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters):
        return {}

    @staticmethod
    def add_feature_data(
            feature,
            dataset_df,
            data,
            metadata,
            preprocessing_parameters=None
    ):
        data[feature['name']] = dataset_df[feature['name']].astype(
            np.bool_).values


class BinaryInputFeature(BinaryBaseFeature, InputFeature):
    def __init__(self, feature, encoder_obj=None):
        BinaryBaseFeature.__init__(self, feature)
        InputFeature.__init__(self)

        self.encoder = 'passthrough'
        self.norm = None
        self.dropout = False

        encoder_parameters = self.overwrite_defaults(feature)

        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.get_binary_encoder(encoder_parameters)

    def get_binary_encoder(self, encoder_parameters):
        return get_from_registry(self.encoder, self.encoder_registry)(
            **encoder_parameters
        )

    def call(self, inputs, training=None, mask=None):
        assert isinstance(inputs, tf.Tensor)
        assert inputs.dtype == tf.bool
        assert len(inputs.shape) == 1

        inputs_exp = inputs[:, tf.newaxis]
        inputs_encoded = self.encoder_obj(
            inputs_exp, training=training, mask=mask
        )

        return inputs_encoded

    @staticmethod
    def update_model_definition_with_metadata(
            input_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        pass

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, TIED, None)

    encoder_registry = {
        'dense': FCStack,
        'passthrough': BinaryPassthroughEncoder,
        'null': BinaryPassthroughEncoder,
        'none': BinaryPassthroughEncoder,
        'None': BinaryPassthroughEncoder,
        None: BinaryPassthroughEncoder
    }


class BinaryOutputFeature(BinaryBaseFeature, OutputFeature):
    def __init__(self, feature):
        BinaryBaseFeature.__init__(self, feature)
        OutputFeature.__init__(self, feature)

        self.decoder = 'regressor'
        self.threshold = 0.5

        self.initializer = None
        self.regularize = True

        self.loss = {
            'robust_lambda': 0,
            'confidence_penalty': 0,
            'positive_class_weight': 1,
            'weight': 1
        }

        decoder_parameters = self.overwrite_defaults(feature)

        self.decoder = self.initialize_decoder(decoder_parameters)

        self._setup_loss()
        self._setup_metrics()

    def logits(
            self,
            inputs  # hidden
    ):
        return self.decoder(inputs)

    def predictions(
            self,
            inputs  # hidden
    ):
        logits = inputs

        probabilities = tf.nn.sigmoid(
            logits,
            name='probabilities_{}'.format(
                self.name)
        )
        predictions = tf.greater_equal(
            probabilities,
            self.threshold,
            name='predictions_{}'.format(
                self.name)
        )

        return {
            'probabilities': probabilities,
            'predictions': predictions,
            'logits': inputs
        }

    def _setup_loss(self):
        self.train_loss_function = BWCEWLoss(
            positive_class_weight=self.loss['positive_class_weight'],
            robust_lambda=self.loss['robust_lambda']
        )
        self.eval_loss_function = BWCEWLMetric(
            bwcew_loss_function=self.train_loss_function,
            name='eval_loss'
        )

    def _setup_metrics(self):
        self.metric_functions.update(
            {ACCURACY: BinaryAccuracy(name='metric_accuracy')}
        )
        self.metric_functions.update(
            {
                LOSS: BWCEWLMetric(
                    bwcew_loss_function=self.train_loss_function,
                    name='metric_bwcewl'
                )
             }
        )

    # override super class OutputFeature method to customize binary feature
    def update_metrics(self, targets, predictions):
        for metric, metric_fn in self.metric_functions.items():
            if metric == LOSS:
                metric_fn.update_state(
                    targets,
                    predictions['logits'],
                    positive_class_weight=self.loss['positive_class_weight'],
                    robust_lambda=self.loss['robust_lambda']
                )
            else:
                metric_fn.update_state(targets, predictions['predictions'])

    default_validation_metric = ACCURACY

    @staticmethod
    def update_model_definition_with_metadata(
            input_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        pass

    @staticmethod
    def calculate_overall_stats(
            test_stats,
            output_feature,
            dataset,
            train_set_metadata
    ):
        feature_name = output_feature['name']
        stats = test_stats[feature_name]

        confusion_matrix = ConfusionMatrix(
            dataset.get(feature_name),
            stats[PREDICTIONS],
            labels=['False', 'True']
        )
        stats['confusion_matrix'] = confusion_matrix.cm.tolist()
        stats['overall_stats'] = confusion_matrix.stats()
        stats['per_class_stats'] = confusion_matrix.per_class_stats()
        fpr, tpr, thresholds = roc_curve(
            dataset.get(feature_name),
            stats[PROBABILITIES]
        )
        stats['roc_curve'] = {
            'false_positive_rate': fpr.tolist(),
            'true_positive_rate': tpr.tolist()
        }
        stats['roc_auc_macro'] = roc_auc_score(
            dataset.get(feature_name),
            stats[PROBABILITIES],
            average='macro'
        )
        stats['roc_auc_micro'] = roc_auc_score(
            dataset.get(feature_name),
            stats[PROBABILITIES],
            average='micro'
        )
        ps, rs, thresholds = precision_recall_curve(
            dataset.get(feature_name),
            stats[PROBABILITIES]
        )
        stats['precision_recall_curve'] = {
            'precisions': ps.tolist(),
            'recalls': rs.tolist()
        }
        stats['average_precision_macro'] = average_precision_score(
            dataset.get(feature_name),
            stats[PROBABILITIES],
            average='macro'
        )
        stats['average_precision_micro'] = average_precision_score(
            dataset.get(feature_name),
            stats[PROBABILITIES],
            average='micro'
        )
        stats['average_precision_samples'] = average_precision_score(
            dataset.get(feature_name),
            stats[PROBABILITIES],
            average='samples'
        )

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
            postprocessed[PREDICTIONS] = result[PREDICTIONS].numpy()
            if not skip_save_unprocessed_output:
                np.save(
                    npy_filename.format(name, PREDICTIONS),
                    result[PREDICTIONS]
                )
            del result[PREDICTIONS]

        if PROBABILITIES in result and len(result[PROBABILITIES]) > 0:
            postprocessed[PROBABILITIES] = result[PROBABILITIES].numpy()
            if not skip_save_unprocessed_output:
                np.save(
                    npy_filename.format(name, PROBABILITIES),
                    result[PROBABILITIES]
                )
            del result[PROBABILITIES]

        return postprocessed

    @staticmethod
    def populate_defaults(output_feature):
        set_default_value(
            output_feature,
            LOSS,
            {
                'robust_lambda': 0,
                'confidence_penalty': 0,
                'positive_class_weight': 1,
                'weight': 1
            }
        )

        set_default_value(output_feature[LOSS], 'robust_lambda', 0)
        set_default_value(output_feature[LOSS], 'confidence_penalty', 0)
        set_default_value(output_feature[LOSS], 'positive_class_weight', 1)
        set_default_value(output_feature[LOSS], 'weight', 1)

        set_default_values(
            output_feature,
            {
                'threshold': 0.5,
                'dependencies': [],
                'reduce_input': SUM,
                'reduce_dependencies': SUM
            }
        )

    decoder_registry = {
        'regressor': Regressor,
        'null': Regressor,
        'none': Regressor,
        'None': Regressor,
        None: Regressor
    }
