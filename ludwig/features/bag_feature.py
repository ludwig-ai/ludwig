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
from collections import Counter

import numpy as np
import tensorflow as tf

from ludwig.constants import *
from ludwig.features.base_feature import BaseFeature
from ludwig.features.base_feature import InputFeature
from ludwig.features.feature_utils import set_str_to_idx
from ludwig.models.modules.embedding_modules import EmbedWeighted
from ludwig.utils.misc import set_default_value
from ludwig.utils.strings_utils import create_vocabulary

logger = logging.getLogger(__name__)


class BagBaseFeature(BaseFeature):
    def __init__(self, feature):
        super().__init__(feature)
        self.type = BAG

    preprocessing_defaults = {
        'tokenizer': 'space',
        'most_common': 10000,
        'lowercase': False,
        'missing_value_strategy': FILL_WITH_CONST,
        'fill_value': ''
    }

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
        bag_matrix = np.zeros(
            (len(column),
             len(metadata['str2idx'])),
            dtype=float
        )

        for i in range(len(column)):
            col_counter = Counter(set_str_to_idx(
                column[i],
                metadata['str2idx'],
                preprocessing_parameters['tokenizer'])
            )
            bag_matrix[i, list(col_counter.keys())] = list(col_counter.values())

        return bag_matrix

    @staticmethod
    def add_feature_data(
            feature,
            dataset_df,
            data,
            metadata,
            preprocessing_parameters=None
    ):
        data[feature['name']] = BagBaseFeature.feature_data(
            dataset_df[feature['name']].astype(str),
            metadata[feature['name']],
            preprocessing_parameters
        )


class BagInputFeature(BagBaseFeature, InputFeature):
    def __init__(self, feature):
        super().__init__(feature)

        self.vocab = []

        self.embedding_size = 50
        self.representation = 'dense'
        self.embeddings_trainable = True
        self.pretrained_embeddings = None
        self.embeddings_on_cpu = False
        self.dropout = False
        self.initializer = None
        self.regularize = True

        _ = self.overwrite_defaults(feature)

        self.embed_weighted = EmbedWeighted(
            self.vocab,
            self.embedding_size,
            representation=self.representation,
            embeddings_trainable=self.embeddings_trainable,
            pretrained_embeddings=self.pretrained_embeddings,
            embeddings_on_cpu=self.embeddings_on_cpu,
            dropout=self.dropout,
            initializer=self.initializer,
            regularize=self.regularize
        )

    def _get_input_placeholder(self):
        # None dimension is for dealing with variable batch size
        return tf.compat.v1.placeholder(
            tf.float32,
            shape=[None, len(self.vocab)],
            name=self.name
        )

    def build_input(
            self,
            regularizer,
            dropout_rate,
            is_training=False,
            **kwargs
    ):
        placeholder = self._get_input_placeholder()
        logger.debug('placeholder: {0}'.format(placeholder))

        embedded, embedding_size = self.embed_weighted(
            placeholder,
            regularizer,
            dropout_rate,
            is_training=False
        )
        logger.debug('feature_representation: {0}'.format(embedded))

        feature_representation = {
            'name': self.name,
            'type': self.type,
            'representation': embedded,
            'size': embedding_size,
            'placeholder': placeholder
        }
        return feature_representation

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
        set_default_value(input_feature, 'tied_weights', None)
