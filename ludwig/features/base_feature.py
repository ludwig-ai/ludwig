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
from abc import ABC, abstractmethod

import tensorflow.compat.v1 as tf

from ludwig.models.modules.fully_connected_modules import FCStack
from ludwig.models.modules.reduction_modules import reduce_sequence
from ludwig.utils.misc import merge_dict, get_from_registry
from ludwig.utils.tf_utils import sequence_length_3D


class BaseFeature:
    def __init__(self, feature):
        if 'name' not in feature:
            raise ValueError('Missing feature name')

        self.feature_name = feature['name']
        self.type = None

    def overwrite_defaults(self, feature):
        attributes = self.__dict__.keys()

        remaining_dict = dict(feature)

        for k in feature.keys():
            if k in attributes:
                if (isinstance(feature[k], dict) and hasattr(self, k)
                        and isinstance(getattr(self, k), dict)):
                    setattr(self, k, merge_dict(getattr(self, k),
                                                feature[k]))
                else:
                    setattr(self, k, feature[k])
                del remaining_dict[k]

        return remaining_dict


class InputFeature(ABC, tf.keras.Model):

    @staticmethod
    @abstractmethod
    def update_model_definition_with_metadata(
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


class OutputFeature(ABC, BaseFeature, tf.keras.Model):

    def __init__(self, feature):
        BaseFeature.__init__(self, feature)
        tf.keras.Model.__init__(self)

        self.loss = None
        self.train_loss_function = None
        self.eval_loss_function = None
        self.metric_functions = {}

        self.reduce_input = None
        self.reduce_dependencies = None
        self.dependencies = []

        self.fc_layers = None
        self.num_fc_layers = 0
        self.fc_size = 256
        self.activation = 'relu'
        self.norm = None
        self.dropout = False
        self.regularize = True
        self.initializer = None

        self.overwrite_defaults(feature)

        self.fc_stack = FCStack(
            layers=self.fc_layers,
            num_layers=self.num_fc_layers,
            default_fc_size=self.fc_size,
            default_activation=self.activation,
            default_use_bias=True,
            default_norm=self.norm,
            # default_dropout_rate=self.dropout_rate,
            default_weights_initializer=self.initializer,
            # default_bias_initializer='zeros',
            # default_weights_regularizer=None,
            # default_bias_regularizer=None,
            # default_activity_regularizer=None,
            # default_weights_constraint=None,
            # default_bias_constraint=None,
        )

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
        return self.eval_loss_function(targets, predictions['logits'])

    def update_metrics(self, targets, predictions):
        for metric in self.metric_functions.values():
            metric.update_state(targets, predictions['predictions'])

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
            inputs,  # hidden, other_output_hidden
            training=None,
            mask=None
    ):
        combiner_output, other_output_hidden = inputs

        feature_hidden = self.prepare_decoder_inputs(
            combiner_output,
            other_output_hidden,
            training=training,
            mask=mask
        )

        # ================ Predictions ================
        predictions = self.logits(feature_hidden)

        return predictions

    @property
    @abstractmethod
    def default_validation_metric(self):
        pass

    @staticmethod
    @abstractmethod
    def update_model_definition_with_metadata(
            output_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        pass

    @staticmethod
    @abstractmethod
    def calculate_overall_stats(
            test_stats,
            output_feature,
            dataset,
            train_set_metadata
    ):
        pass

    @staticmethod
    @abstractmethod
    def postprocess_results(
            output_feature,
            result,
            metadata,
            experiment_dir_name,
            skip_save_unprocessed_output=False,
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

                # todo tf2: test all 4 branches, for now only vector x vector is tested
                if len(hidden.shape) > 2:
                    if len(dependency_final_hidden.shape) > 2:
                        # matrix matrix -> concat
                        assert hidden.shape[1] == dependency_final_hidden.shape[1]
                        dependencies_hidden.append(dependency_final_hidden)
                    else:
                        # matrix vector -> tile concat
                        sequence_max_length = hidden.shape[1]
                        multipliers = tf.concat(
                            [[1], sequence_max_length[:, tf.newaxis], [1]],
                            0
                        )
                        tiled_representation = tf.tile(
                            tf.expand_dims(dependency_final_hidden, 1),
                            multipliers
                        )

                        # todo tf2: modify this with TF2 mask mechanics
                        sequence_length = sequence_length_3D(hidden)
                        mask = tf.sequence_mask(
                            sequence_length,
                            sequence_max_length
                        )
                        tiled_representation = tf.multiply(
                            tiled_representation,
                            tf.cast(mask[:, tf.newaxis], dtype=tf.float32)
                        )

                        dependencies_hidden.append(tiled_representation)

                else:
                    if len(dependency_final_hidden.shape) > 2:
                        # vector matrix -> reduce concat
                        dependencies_hidden.append(
                            reduce_sequence(dependency_final_hidden,
                                            self.reduce_dependencies)
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
                        self.feature_name,
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
        :param kwargs:
        :return: tensor
        """
        feature_hidden = combiner_output

        # ================ Reduce Inputs ================
        if self.reduce_input is not None and len(feature_hidden.shape) > 2:
            feature_hidden = reduce_sequence(
                feature_hidden,
                self.reduce_input
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
        other_output_features[self.feature_name] = feature_hidden

        return feature_hidden
