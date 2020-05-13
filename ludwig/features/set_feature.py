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
from collections import OrderedDict

import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.keras.metrics import MeanIoU

from ludwig.constants import *
from ludwig.features.base_feature import BaseFeature
from ludwig.features.base_feature import InputFeature
from ludwig.features.base_feature import OutputFeature
from ludwig.features.feature_utils import set_str_to_idx
from ludwig.models.modules.initializer_modules import get_initializer
from ludwig.models.modules.loss_modules import SigmoidCrossEntropyLoss
from ludwig.models.modules.metric_modules import SigmoidCrossEntropyMetric
from ludwig.models.modules.set_encoders import SetSparseEncoder
from ludwig.models.modules.set_decoders import Classifier
from ludwig.utils.misc import set_default_value
from ludwig.utils.strings_utils import create_vocabulary

logger = logging.getLogger(__name__)


class SetBaseFeature(BaseFeature):
    type = SET
    preprocessing_defaults = {
        'tokenizer': 'space',
        'most_common': 10000,
        'lowercase': False,
        'missing_value_strategy': FILL_WITH_CONST,
        'fill_value': ''
    }

    def __init__(self, feature):
        super().__init__(feature)

    @staticmethod
    def get_feature_meta(column, preprocessing_parameters):
        idx2str, str2idx, str2freq, max_size = create_vocabulary(
            column,
            preprocessing_parameters['tokenizer'],
            num_most_frequent=preprocessing_parameters['most_common'],
            lowercase=preprocessing_parameters['lowercase']
        )
        return {
            'idx2str': idx2str,
            'str2idx': str2idx,
            'str2freq': str2freq,
            'vocab_size': len(str2idx),
            'max_set_size': max_size
        }

    @staticmethod
    def feature_data(column, metadata, preprocessing_parameters):
        feature_vector = np.array(
            column.map(
                lambda x: set_str_to_idx(
                    x,
                    metadata['str2idx'],
                    preprocessing_parameters['tokenizer']
                )
            )
        )

        set_matrix = np.zeros(
            (len(column),
             len(metadata['str2idx'])),
        )

        for i in range(len(column)):
            set_matrix[i, feature_vector[i]] = 1

        return set_matrix.astype(np.bool)

    @staticmethod
    def add_feature_data(
            feature,
            dataset_df,
            data,
            metadata,
            preprocessing_parameters,
    ):
        data[feature['name']] = SetBaseFeature.feature_data(
            dataset_df[feature['name']].astype(str),
            metadata[feature['name']],
            preprocessing_parameters
        )


class SetInputFeature(SetBaseFeature, InputFeature):
    encoder = 'embed'

    def __init__(self, feature, encoder_obj=None):
        super().__init__(feature)

        SetBaseFeature.__init__(self, feature)
        InputFeature.__init__(self)
        self.overwrite_defaults(feature)
        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(feature)

    def call(self, inputs, training=None, mask=None):
        assert isinstance(inputs, tf.Tensor)
        assert inputs.dtype == tf.bool

        encoder_output = self.encoder_obj(
            inputs, training=training, mask=mask
        )

        return {'encoder_output': encoder_output}

    @staticmethod
    def update_model_definition_with_metadata(
            input_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        input_feature['vocab'] = feature_metadata['idx2str']

    @staticmethod
    def populate_defaults(input_feature):
        set_default_value(input_feature, TIED, None)

    encoder_registry = {
        'embed': SetSparseEncoder,
        None: SetSparseEncoder
    }


class SetOutputFeature(SetBaseFeature, OutputFeature):
    decoder = 'classifier'
    num_classes = 0
    loss = {'type': SIGMOID_CROSS_ENTROPY}

    def __init__(self, feature):
        SetBaseFeature.__init__(self, feature)
        OutputFeature.__init__(self, feature)
        self.overwrite_defaults(feature)
        self.decoder_obj = self.initialize_decoder(feature)
        self._setup_loss()
        self._setup_metrics()
        # super().__init__(feature)
        # self.type = SET

        # self.loss = {'type': 'sigmoid_cross_entropy'}
        self.num_classes = 0
        self.threshold = 0.5
        # self.initializer = None
        # self.regularize = True

        # _ = self.overwrite_defaults(feature)

    # def _get_output_placeholder(self):
    #     return tf.placeholder(
    #         tf.bool,
    #         shape=[None, self.num_classes],
    #         name='{}_placeholder'.format(self.feature_name)
    #     )
    def logits(
            self,
            inputs,  # hidden
    ):
        return self.decoder_obj(inputs)

    def predictions(
            self,
            inputs,  # logits
    ):
        logits = inputs[LOGITS]

        probabilities = tf.nn.sigmoid(
            logits,
            name='probabilities_{}'.format(self.feature_name)
        )

        predictions = tf.greater_equal(
            probabilities,
            self.threshold,
            name='predictions_{}'.format(self.feature_name)
        )
        predictions = tf.cast(predictions, dtype=tf.int64)

        return {
            PREDICTIONS: predictions,
            PROBABILITIES: probabilities,
            LOGITS: logits
        }

    def _setup_loss(self):
        self.train_loss_function = SigmoidCrossEntropyLoss(
            name='train_loss'
        )

        self.eval_loss_function = SigmoidCrossEntropyMetric(
            name='eval_loss'
        )

    def _setup_metrics(self):
        self.metric_functions[LOSS] = self.eval_loss_function
        self.metric_functions[JACCARD] = MeanIoU(num_classes=self.num_classes)

    default_validation_metric = JACCARD

    output_config = OrderedDict([
        (LOSS, {
            'output': EVAL_LOSS,
            'aggregation': SUM,
            'value': 0,
            'type': METRIC
        }),
        (JACCARD, {
            'output': JACCARD,
            'aggregation': SUM,
            'value': 0,
            'type': METRIC
        }),
        (PREDICTIONS, {
            'output': PREDICTIONS,
            'aggregation': APPEND,
            'value': [],
            'type': PREDICTION
        }),
        (PROBABILITIES, {
            'output': PROBABILITIES,
            'aggregation': APPEND,
            'value': [],
            'type': PREDICTION
        })
    ])

    @staticmethod
    def update_model_definition_with_metadata(
            output_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        output_feature[LOSS]['type'] = None
        output_feature['num_classes'] = feature_metadata['vocab_size']

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
            preds = result[PREDICTIONS]
            if 'idx2str' in metadata:
                postprocessed[PREDICTIONS] = [
                    [metadata['idx2str'][i] for i, pred in enumerate(pred_set)
                     if pred == True] for pred_set in preds
                ]
            else:
                postprocessed[PREDICTIONS] = preds

            if not skip_save_unprocessed_output:
                np.save(npy_filename.format(name, PREDICTIONS), preds)

            del result[PREDICTIONS]

        if PROBABILITIES in result and len(result[PROBABILITIES]) > 0:
            probs = result[PROBABILITIES]
            prob = [[prob for prob in prob_set if
                     prob >= output_feature['threshold']] for prob_set in probs]
            postprocessed[PROBABILITIES] = probs
            postprocessed[PROBABILITY] = prob

            if not skip_save_unprocessed_output:
                np.save(npy_filename.format(name, PROBABILITIES), probs)
                np.save(npy_filename.format(name, PROBABILITY), probs)

            del result[PROBABILITIES]

        return postprocessed

    @staticmethod
    def populate_defaults(output_feature):
        set_default_value(output_feature, LOSS, {'weight': 1, 'type': None})
        set_default_value(output_feature[LOSS], 'weight', 1)

        set_default_value(output_feature, 'threshold', 0.5)
        set_default_value(output_feature, 'dependencies', [])
        set_default_value(output_feature, 'reduce_input', SUM)
        set_default_value(output_feature, 'reduce_dependencies', SUM)

    decoder_registry = {
        'classifier': Classifier,
        'null': Classifier,
        'none': Classifier,
        'None': Classifier,
        None: Classifier
    }
