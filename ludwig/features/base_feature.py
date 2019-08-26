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

import tensorflow as tf

from ludwig.models.modules.fully_connected_modules import FCStack
from ludwig.models.modules.reduction_modules import reduce_sequence
from ludwig.utils.misc import merge_dict
from ludwig.utils.tf_utils import sequence_length_3D


class BaseFeature:
    def __init__(self, feature):
        if 'name' not in feature:
            raise ValueError('Missing feature name')

        self.name = feature['name']
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


class InputFeature(ABC):

    @staticmethod
    @abstractmethod
    def update_model_definition_with_metadata(
            input_feature,
            feature_metadata,
            *args,
            **kwargs
    ):
        pass

    @abstractmethod
    def build_input(
            self,
            regularizer,
            dropout_rate,
            is_training=False,
            **kwargs
    ):
        pass

    @staticmethod
    @abstractmethod
    def populate_defaults(input_feature):
        pass


class OutputFeature(ABC, BaseFeature):

    def __init__(self, feature):
        super().__init__(feature)
        self.loss = None
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

    @property
    @abstractmethod
    def default_validation_measure(self):
        pass

    @property
    @abstractmethod
    def output_config(self):
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

    @abstractmethod
    def build_output(
            self,
            hidden,
            hidden_size,
            regularizer=None,
            dropout_rate=None,
            is_training=None,
            **kwargs
    ):
        pass

    @staticmethod
    @abstractmethod
    def populate_defaults(input_feature):
        pass

    def concat_dependencies(self, hidden, hidden_size, final_hidden):
        if len(self.dependencies) > 0:
            dependencies_hidden = []
            for dependency in self.dependencies:
                # the dependent feature is ensured to be present in final_hidden
                # because we did the topological sort of the features before
                dependency_final_hidden = final_hidden[dependency]

                if len(hidden.shape) > 2:
                    if len(dependency_final_hidden[0].shape) > 2:
                        # matrix matrix -> concat
                        dependencies_hidden.append(dependency_final_hidden[0])
                    else:
                        # matrix vector -> tile concat
                        sequence_max_length = tf.shape(hidden)[1]
                        multipliers = tf.concat(
                            [[1], tf.expand_dims(sequence_max_length, -1), [1]],
                            0
                        )
                        tiled_representation = tf.tile(
                            tf.expand_dims(dependency_final_hidden[0], 1),
                            multipliers
                        )

                        sequence_length = sequence_length_3D(hidden)
                        mask = tf.sequence_mask(
                            sequence_length,
                            sequence_max_length
                        )
                        tiled_representation = tf.multiply(
                            tiled_representation,
                            tf.cast(tf.expand_dims(mask, -1), dtype=tf.float32)
                        )

                        dependencies_hidden.append(tiled_representation)

                else:
                    if len(dependency_final_hidden[0].shape) > 2:
                        # vector matrix -> reduce concat
                        dependencies_hidden.append(
                            reduce_sequence(dependency_final_hidden[0],
                                            self.reduce_dependencies)
                        )
                    else:
                        # vector vector -> concat
                        dependencies_hidden.append(dependency_final_hidden[0])

                hidden_size += dependency_final_hidden[1]

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
                        self.name,
                        self.dependencies,
                        hidden,
                        dependencies_hidden
                    )
                )

        return hidden, hidden_size

    def output_specific_fully_connected(
            self,
            feature_hidden,
            feature_hidden_size,
            dropout_rate,
            regularizer,
            is_training=True
    ):
        original_feature_hidden = feature_hidden
        if len(original_feature_hidden.shape) > 2:
            feature_hidden = tf.reshape(
                feature_hidden,
                [-1, feature_hidden_size]
            )

        if self.fc_layers is not None or self.num_fc_layers > 0:
            fc_stack = FCStack(
                layers=self.fc_layers,
                num_layers=self.num_fc_layers,
                default_fc_size=self.fc_size,
                default_activation=self.activation,
                default_norm=self.norm,
                default_dropout=self.dropout,
                default_regularize=self.regularize,
                default_initializer=self.initializer
            )
            feature_hidden = fc_stack(
                feature_hidden,
                feature_hidden_size,
                regularizer,
                dropout_rate,
                is_training=is_training
            )
            feature_hidden_size = feature_hidden.shape.as_list()[-1]

        if len(original_feature_hidden.shape) > 2:
            sequence_length = tf.shape(original_feature_hidden)[1]
            feature_hidden = tf.reshape(
                feature_hidden,
                [-1, sequence_length, feature_hidden_size]
            )

        return feature_hidden, feature_hidden_size

    def concat_dependencies_and_build_output(
            self,
            combiner_hidden,
            combiner_hidden_size,
            final_hidden,
            regularizer=None,
            **kwargs
    ):
        # ================ Reduce Inputs ================
        if self.reduce_input is not None and len(combiner_hidden.shape) > 2:
            combiner_hidden = reduce_sequence(
                combiner_hidden,
                self.reduce_input
            )

        # ================ Adding Dependencies ================
        feature_hidden, feature_hidden_size = self.concat_dependencies(
            combiner_hidden,
            combiner_hidden_size,
            final_hidden
        )

        # ================ Output-wise Fully Connected ================
        (
            feature_hidden,
            feature_hidden_size
        ) = self.output_specific_fully_connected(
            feature_hidden,
            feature_hidden_size,
            dropout_rate=kwargs['dropout_rate'],
            regularizer=regularizer,
            is_training=kwargs['is_training']
        )
        final_hidden[self.name] = (feature_hidden, feature_hidden_size)

        # ================ Outputs ================
        train_mean_loss, eval_loss, output_tensors = self.build_output(
            feature_hidden,
            feature_hidden_size,
            regularizer=regularizer,
            **kwargs
        )

        loss_weight = float(self.loss['weight'])
        weighted_train_mean_loss = train_mean_loss * loss_weight
        weighted_eval_loss = eval_loss * loss_weight

        return weighted_train_mean_loss, weighted_eval_loss, output_tensors
