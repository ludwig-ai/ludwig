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
from tensorflow.keras.metrics import Accuracy as BinaryAccuracy

from ludwig.constants import *
from ludwig.decoders.generic_decoders import Regressor
from ludwig.encoders.generic_encoders import PassthroughEncoder, \
    DenseEncoder
from ludwig.features.base_feature import InputFeature
from ludwig.features.base_feature import OutputFeature
from ludwig.globals import is_on_master
from ludwig.modules.loss_modules import BWCEWLoss
from ludwig.modules.metric_modules import BWCEWLMetric
from ludwig.utils.metrics_utils import ConfusionMatrix
from ludwig.utils.metrics_utils import average_precision_score
from ludwig.utils.metrics_utils import precision_recall_curve
from ludwig.utils.metrics_utils import roc_auc_score
from ludwig.utils.metrics_utils import roc_curve
from ludwig.utils.misc_utils import set_default_value
from ludwig.utils.misc_utils import set_default_values
from ludwig.utils.strings_utils import str2bool

logger = logging.getLogger(__name__)


class BinaryFeatureMixin(object):
    type = BINARY
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
        column = dataset_df[feature['name']]
        if column.dtype == object:
            column = column.map(str2bool)
        data[feature['name']] = column.astype(np.bool_).values


class BinaryInputFeature(BinaryFeatureMixin, InputFeature):
    encoder = 'passthrough'
    norm = None
    dropout = False

    def __init__(self, feature, encoder_obj=None):
        super().__init__(feature)
        self.overwrite_defaults(feature)
        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(feature)

    def call(self, inputs, training=None, mask=None):
        assert isinstance(inputs, tf.Tensor)
        assert inputs.dtype == tf.bool
        assert len(inputs.shape) == 1

        inputs = tf.cast(inputs, dtype=tf.float32)
        inputs_exp = inputs[:, tf.newaxis]
        encoder_outputs = self.encoder_obj(
            inputs_exp, training=training, mask=mask
        )

        return encoder_outputs

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
        'dense': DenseEncoder,
        'passthrough': PassthroughEncoder,
        'null': PassthroughEncoder,
        'none': PassthroughEncoder,
        'None': PassthroughEncoder,
        None: PassthroughEncoder
    }


class BinaryOutputFeature(BinaryFeatureMixin, OutputFeature):
    decoder = 'regressor'
    loss = {
        'robust_lambda': 0,
        'confidence_penalty': 0,
        'positive_class_weight': 1,
        'weight': 1
    }
    threshold = 0.5

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
            inputs,  # hidden
            **kwargs
    ):
        logits = inputs[LOGITS]

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
            PROBABILITIES: probabilities,
            PREDICTIONS: predictions,
            LOGITS: logits
        }

    def _setup_loss(self):
        self.train_loss_function = BWCEWLoss(
            positive_class_weight=self.loss['positive_class_weight'],
            robust_lambda=self.loss['robust_lambda'],
            confidence_penalty=self.loss['confidence_penalty']
        )
        self.eval_loss_function = BWCEWLMetric(
            positive_class_weight=self.loss['positive_class_weight'],
            robust_lambda=self.loss['robust_lambda'],
            confidence_penalty=self.loss['confidence_penalty'],
            name='eval_loss'
        )

    def _setup_metrics(self):
        self.metric_functions[LOSS] = self.eval_loss_function
        self.metric_functions[ACCURACY] = BinaryAccuracy(
            name='metric_accuracy')

    # def update_metrics(self, targets, predictions):
    #     for metric, metric_fn in self.metric_functions.items():
    #         if metric == LOSS:
    #             metric_fn.update_state(targets, predictions[LOGITS])
    #         else:
    #             metric_fn.update_state(targets, predictions[PREDICTIONS])

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
        name = output_feature['name']

        npy_filename = None
        if is_on_master():
            npy_filename = os.path.join(experiment_dir_name, '{}_{}.npy')
        else:
            skip_save_unprocessed_output = True

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
        # If Loss is not defined, set an empty dictionary
        set_default_value(output_feature, LOSS, {})
        set_default_values(
            output_feature[LOSS],
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
