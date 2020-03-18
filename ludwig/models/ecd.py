import logging
from collections import OrderedDict

import tensorflow as tf

from ludwig.constants import TIED
from ludwig.features.feature_registries import input_type_registry, output_type_registry
from ludwig.models.modules.combiners import get_combiner_class
from ludwig.utils.algorithms_utils import topological_sort_feature_dependencies
from ludwig.utils.misc import get_from_registry

logger = logging.getLogger(__name__)


class ECD(tf.keras.Model):
    def __init__(
            self,
            input_features_def,
            combiner_def,
            output_features_def,
            **kwargs
    ):
        super().__init__()

        # ================ Inputs ================
        self.input_features = build_inputs(
            input_features_def
        )

        # ================ Combiner ================
        logger.debug('- Combiner {}'.format(combiner_def['type']))
        combiner_class = get_combiner_class(combiner_def['type'])
        self.combiner = combiner_class(
            self.input_features,
            self.regularizer,
            **combiner_def,
            **kwargs
        )

        # ================ Outputs ================
        self.output_features = build_outputs(
            output_features_def,
            self.combiner
        )

    def call(self, inputs, training=None, mask=None):
        # todo: tf2 proof-of-concept code
        # inputs is a dict feature_name -> tensor / ndarray
        assert inputs.keys() == self.input_features.keys()

        encoder_outputs = []
        for input_feature_name, input_values in inputs:
            encoder = self.input_features[input_feature_name]
            encoder_output = encoder(input_values)
            encoder_outputs.append(encoder_output)

        combiner_outputs = self.combiner(encoder_outputs)

        output_tensors = {}
        for output_feature_name, decoder in self.output_features:
            decoder_output = decoder(combiner_outputs)
            output_tensors[output_feature_name] = decoder_output

        return output_tensors


def build_inputs(
        input_features_def,
        **kwargs
):
    input_features = OrderedDict()
    input_features_def = topological_sort_feature_dependencies(input_features_def)
    for input_feature_def in input_features_def:
        input_features[input_feature_def['name']] = build_single_input(
            input_feature_def,
            input_features,
            **kwargs
        )
    return input_features


def build_single_input(input_feature_def, other_input_features, **kwargs):
    logger.debug('- Input {} feature {}'.format(
        input_feature_def['type'],
        input_feature_def['name']
    ))

    # todo tf2: tied encoder mechanism to be tested
    encoder_obj = None
    if input_feature_def.get(TIED, None) is not None:
        tied_input_feature_name = input_feature_def[TIED]
        if tied_input_feature_name in other_input_features:
            encoder_obj = other_input_features[tied_input_feature_name].encoder_obj

    input_feature_class = get_from_registry(
        input_feature_def['type'],
        input_type_registry
    )
    input_feature_obj = input_feature_class(input_feature_def, encoder_obj)

    return input_feature_obj


dynamic_length_encoders = {
    'rnn',
    'embed'
}


def build_outputs(
        output_features_def,
        combiner,
        **kwargs
):
    output_features_def = topological_sort_feature_dependencies(output_features_def)
    output_features = {}

    for output_feature_def in output_features_def:
        output_feature = build_single_output(
            output_feature_def,
            combiner,
            output_features,
            **kwargs
        )
        output_features[output_feature_def['name']] = output_feature

    return output_features


def build_single_output(
        output_feature_def,
        feature_hidden,
        other_output_features,
        regularizer,
        **kwargs
):
    logger.debug('- Output {} feature {}'.format(
        output_feature_def['type'],
        output_feature_def['name']
    ))

    output_feature_class = get_from_registry(
        output_feature_def['type'],
        output_type_registry
    )
    output_feature_obj = output_feature_class(output_feature_def)
    weighted_train_mean_loss, weighted_eval_loss, output_tensors = output_feature_obj.concat_dependencies_and_build_output(
        feature_hidden,
        other_output_features,
        **kwargs
    )

    return weighted_train_mean_loss, weighted_eval_loss, output_tensors


def calculate_combined_loss(output_feature):
    output_train_losses.append(output_feature)
    output_eval_losses.append(of_eval_loss)
    output_tensors.update(of_output_tensors)

    train_combined_mean_loss = tf.reduce_sum(
        tf.stack(output_train_losses),
        name='train_combined_mean_loss')

    # todo re-add later
    # if regularizer is not None:
    #    regularization_losses = tf.get_collection(
    #        tf.GraphKeys.REGULARIZATION_LOSSES)
    #   if regularization_losses:
    #        regularization_loss = tf.add_n(regularization_losses)
    #        logger.debug('- Regularization losses: {0}'.format(
    #            regularization_losses))
    #
    #   else:
    #        regularization_loss = tf.constant(0.0)
    # else:
    regularization_loss = tf.constant(0.0)

    train_reg_mean_loss = tf.add(train_combined_mean_loss,
                                 regularization_loss,
                                 name='train_combined_regularized_mean_loss')

    eval_combined_loss = tf.reduce_sum(tf.stack(output_eval_losses),
                                       axis=0,
                                       name='eval_combined_loss')

    return train_reg_mean_loss, eval_combined_loss, regularization_loss, output_tensors
