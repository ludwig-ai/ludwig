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
from abc import ABC, abstractmethod
from typing import Dict

import tensorflow as tf

from ludwig.constants import *
from ludwig.features.feature_utils import compute_feature_hash
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.modules.reduction_modules import SequenceReducer
from ludwig.utils.misc_utils import merge_dict, get_from_registry
from ludwig.utils.tf_utils import sequence_length_3D

logger = logging.getLogger(__name__)


class BaseFeature(object):
    """Base class for all features.

    Note that this class is not-cooperative (does not forward kwargs), so when constructing
    feature class hierarchies, there should be only one parent class that derives from base
    feature.  Other functionality should be put into mixin classes to avoid the diamond
    pattern.
    """

    def __init__(self, feature, *args, **kwargs):
        super().__init__()

        if NAME not in feature:
            raise ValueError('Missing feature name')
        self.feature_name = feature[NAME]

        if COLUMN not in feature:
            feature[COLUMN] = self.feature_name
        self.column = feature[COLUMN]

        if PROC_COLUMN not in feature:
            feature[PROC_COLUMN] = compute_feature_hash(feature)
        self.proc_column = feature[PROC_COLUMN]

        self.type = None

    def overwrite_defaults(self, feature):
        attributes = set(self.__dict__.keys())
        attributes.update(self.__class__.__dict__.keys())

        for k in feature.keys():
            if k in attributes:
                if (isinstance(feature[k], dict) and hasattr(self, k)
                        and isinstance(getattr(self, k), dict)):
                    setattr(self, k, merge_dict(getattr(self, k),
                                                feature[k]))
                else:
                    setattr(self, k, feature[k])


class InputFeature(BaseFeature, tf.keras.Model, ABC):
    """Parent class for all input features."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_input(self):
        return tf.keras.Input(shape=self.get_input_shape(),
                              dtype=self.get_input_dtype(),
                              name=self.name + '_input')

    @classmethod
    @abstractmethod
    def get_input_dtype(cls):
        """Returns the Tensor data type this input accepts."""
        pass

    @abstractmethod
    def get_input_shape(self):
        """Returns a tuple representing the Tensor shape this input accepts."""
        pass

    @staticmethod
    @abstractmethod
    def update_config_with_metadata(
            input_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        pass

    @staticmethod
    @abstractmethod
    def populate_defaults(input_feature):
        pass

    @property
    @abstractmethod
    def encoder_registry(self):
        pass

    def initialize_encoder(self, encoder_parameters):
        return get_from_registry(self.encoder, self.encoder_registry)(
            **encoder_parameters
        )


class OutputFeature(BaseFeature, tf.keras.Model, ABC):
    """Parent class for all output features."""

    train_loss_function = None
    eval_loss_function = None

    def __init__(self, feature, *args, **kwargs):
        super().__init__(*args, feature=feature, **kwargs)

        self.reduce_input = None
        self.reduce_dependencies = None
        self.dependencies = []

        self.fc_layers = None
        self.num_fc_layers = 0
        self.fc_size = 256
        self.use_bias = True
        self.weights_initializer = 'glorot_uniform'
        self.bias_initializer = 'zeros'
        self.weights_regularizer = None
        self.bias_regularizer = None
        self.activity_regularizer = None
        # self.weights_constraint=None
        # self.bias_constraint=None
        self.norm = None
        self.norm_params = None
        self.activation = 'relu'
        self.dropout = 0

        self.overwrite_defaults(feature)

        logger.debug(' output feature fully connected layers')
        logger.debug('  FCStack')
        self.fc_stack = FCStack(
            layers=self.fc_layers,
            num_layers=self.num_fc_layers,
            default_fc_size=self.fc_size,
            default_use_bias=self.use_bias,
            default_weights_initializer=self.weights_initializer,
            default_bias_initializer=self.bias_initializer,
            default_weights_regularizer=self.weights_regularizer,
            default_bias_regularizer=self.bias_regularizer,
            default_activity_regularizer=self.activity_regularizer,
            # default_weights_constraint=self.weights_constraint,
            # default_bias_constraint=self.bias_constraint,
            default_norm=self.norm,
            default_norm_params=self.norm_params,
            default_activation=self.activation,
            default_dropout=self.dropout,
        )

        # set up two sequence reducers, one for inputs and other for dependencies
        self.reduce_sequence_input = SequenceReducer(
            reduce_mode=self.reduce_input
        )
        if self.dependencies:
            self.dependency_reducers = {}
            for dependency in self.dependencies:
                self.dependency_reducers[dependency] = SequenceReducer(
                    reduce_mode=self.reduce_dependencies
                )

    def create_input(self):
        return tf.keras.Input(shape=self.get_output_shape(),
                              dtype=self.get_output_dtype(),
                              name=self.name + '_input')

    @classmethod
    @abstractmethod
    def get_output_dtype(cls):
        """Returns the Tensor data type feature outputs."""
        pass

    @abstractmethod
    def get_output_shape(self):
        """Returns a tuple representing the Tensor shape this feature outputs."""
        pass

    @property
    @abstractmethod
    def metric_functions(self) -> Dict:
        pass

    @property
    @abstractmethod
    def decoder_registry(self):
        pass

    def initialize_decoder(self, decoder_parameters):
        return get_from_registry(self.decoder, self.decoder_registry)(
            **decoder_parameters
        )

    def train_loss(self, targets, predictions):
        return self.train_loss_function(targets, predictions)

    def eval_loss(self, targets, predictions):
        return self.eval_loss_function(targets, predictions)

    def update_metrics(self, targets, predictions):
        for metric, metric_fn in self.metric_functions.items():
            if metric == LOSS or metric == HITS_AT_K:
                metric_fn.update_state(targets, predictions)
            else:
                metric_fn.update_state(targets, predictions[PREDICTIONS])

    def get_metrics(self):
        metric_vals = {}
        for metric_name, metric_onj in self.metric_functions.items():
            metric_vals[metric_name] = metric_onj.result().numpy()
        return metric_vals

    def reset_metrics(self):
        for of_name, metric_fn in self.metric_functions.items():
            if metric_fn is not None:
                metric_fn.reset_states()

    def call(
            self,
            inputs,
            # ((hidden, other_output_hidden), target) or (hidden, other_output_hidden)
            training=None,
            mask=None
    ):
        # account for output feature target
        if isinstance(inputs[0], tuple):
            local_inputs, target = inputs
        else:
            local_inputs = inputs
            target = None

        combiner_outputs, other_output_hidden = local_inputs

        # extract the combined hidden layer
        combiner_output = combiner_outputs['combiner_output']
        hidden = self.prepare_decoder_inputs(
            combiner_output,
            other_output_hidden,
            training=training,
            mask=mask
        )

        # ================ Predictions ================
        logits_input = {
            HIDDEN: hidden
        }
        # pass supplemental data from encoders to decoder
        if 'encoder_output_state' in combiner_outputs:
            logits_input['encoder_output_state'] = \
                combiner_outputs['encoder_output_state']
        if LENGTHS in combiner_outputs:
            logits_input[LENGTHS] = combiner_outputs[LENGTHS]
        logits = self.logits(logits_input, target=target, training=training)

        # most of the cases the output of self.logits is a tensor
        # in some cases like for sequence features, it can be  tuple of
        # logits, predictions, scores
        # The first element will be the logits tensor
        if isinstance(logits, tuple):
            logits = logits[0]
        if not isinstance(logits, dict):
            logits = {'logits': logits}

        return {
            'last_hidden': hidden,
            **logits
        }

    def overall_statistics_metadata(self):
        """Additional metadata used to extend `training_set_metadata`.

        Used when calculating the overall statistics.
        """
        return {}

    @property
    @abstractmethod
    def default_validation_metric(self):
        pass

    @abstractmethod
    def postprocess_predictions(
            self,
            result,
            metadata,
            output_directory,
            skip_save_unprocessed_output=False,
    ):
        pass

    @staticmethod
    @abstractmethod
    def update_config_with_metadata(
            output_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        pass

    @staticmethod
    @abstractmethod
    def calculate_overall_stats(
            predictions,
            targets,
            train_set_metadata
    ):
        pass

    @staticmethod
    @abstractmethod
    def populate_defaults(input_feature):
        pass

    def concat_dependencies(self, hidden, other_features_hidden):
        if len(self.dependencies) > 0:
            dependencies_hidden = []
            for dependency in self.dependencies:
                # the dependent feature is ensured to be present in final_hidden
                # because we did the topological sort of the features before
                dependency_final_hidden = other_features_hidden[dependency]

                if len(hidden.shape) > 2:
                    if len(dependency_final_hidden.shape) > 2:
                        # matrix matrix -> concat
                        assert hidden.shape[1] == \
                               dependency_final_hidden.shape[1]
                        dependencies_hidden.append(dependency_final_hidden)
                    else:
                        # matrix vector -> tile concat
                        sequence_max_length = hidden.shape[1]
                        multipliers = tf.concat(
                            [[1], [sequence_max_length], [1]],
                            0
                        )
                        tiled_representation = tf.tile(
                            tf.expand_dims(dependency_final_hidden, 1),
                            multipliers
                        )

                        # todo future: maybe modify this with TF2 mask mechanics
                        sequence_length = sequence_length_3D(hidden)
                        mask = tf.sequence_mask(
                            sequence_length,
                            sequence_max_length
                        )
                        tiled_representation = tf.multiply(
                            tiled_representation,
                            tf.cast(mask[:, :, tf.newaxis], dtype=tf.float32)
                        )

                        dependencies_hidden.append(tiled_representation)

                else:
                    if len(dependency_final_hidden.shape) > 2:
                        # vector matrix -> reduce concat
                        reducer = self.dependency_reducers[dependency]
                        dependencies_hidden.append(
                            reducer(dependency_final_hidden)
                        )
                    else:
                        # vector vector -> concat
                        dependencies_hidden.append(dependency_final_hidden)

            try:
                hidden = tf.concat([hidden] + dependencies_hidden, -1)
            except:
                raise ValueError(
                    'Shape mismatch while concatenating dependent features of '
                    '{}: {}. Concatenating the feature activations tensor {} '
                    'with activation tensors of dependencies: {}. The error is '
                    'likely due to a mismatch of the second dimension (sequence'
                    ' length) or a difference in ranks. Likely solutions are '
                    'setting the maximum_sequence_length of all sequential '
                    'features to be the same,  or reduce the output of some '
                    'features, or disabling the bucketing setting '
                    'bucketing_field to None / null, as activating it will '
                    'reduce the length of the field the bucketing is performed '
                    'on.'.format(
                        self.column,
                        self.dependencies,
                        hidden,
                        dependencies_hidden
                    )
                )

        return hidden

    def output_specific_fully_connected(
            self,
            inputs,  # feature_hidden
            training=None,
            mask=None
    ):
        feature_hidden = inputs
        original_feature_hidden = inputs

        # flatten inputs
        if len(original_feature_hidden.shape) > 2:
            feature_hidden = tf.reshape(
                feature_hidden,
                [-1, feature_hidden.shape[-1]]
            )

        # pass it through fc_stack
        feature_hidden = self.fc_stack(
            feature_hidden,
            training=training,
            mask=mask
        )
        feature_hidden_size = feature_hidden.shape[-1]

        # reshape back to original first and second dimension
        if len(original_feature_hidden.shape) > 2:
            sequence_length = original_feature_hidden.shape[1]
            feature_hidden = tf.reshape(
                feature_hidden,
                [-1, sequence_length, feature_hidden_size]
            )

        return feature_hidden

    def prepare_decoder_inputs(
            self,
            combiner_output,
            other_output_features,
            training=None,
            mask=None
    ):
        """
        Takes the combiner output and the outputs of other outputs features
        computed so far and performs:
        - reduction of combiner outputs (if needed)
        - concatenating the outputs of dependent features (if needed)
        - output_specific fully connected layers (if needed)

        :param combiner_output: output tensor of the combiner
        :param other_output_features: output tensors from other features
        :return: tensor
        """
        feature_hidden = combiner_output

        # ================ Reduce Inputs ================
        if self.reduce_input is not None and len(feature_hidden.shape) > 2:
            feature_hidden = self.reduce_sequence_input(
                feature_hidden
            )

        # ================ Concat Dependencies ================
        feature_hidden = self.concat_dependencies(
            feature_hidden,
            other_output_features
        )

        # ================ Output-wise Fully Connected ================
        feature_hidden = self.output_specific_fully_connected(
            feature_hidden,
            training=training,
            mask=mask
        )

        return feature_hidden
