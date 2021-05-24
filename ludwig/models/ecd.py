import copy
import logging
from collections import OrderedDict

import tensorflow as tf

from ludwig.combiners.combiners import get_combiner_class
from ludwig.constants import *
from ludwig.features.feature_registries import input_type_registry, \
    output_type_registry
from ludwig.utils.algorithms_utils import topological_sort_feature_dependencies
from ludwig.utils.data_utils import clear_data_cache
from ludwig.utils.misc_utils import get_from_registry

logger = logging.getLogger(__name__)


class ECD(tf.keras.Model):

    def __init__(
            self,
            input_features_def,
            combiner_def,
            output_features_def,
            random_seed=None,
    ):
        # Deep copy to prevent TensorFlow from hijacking the dicts within the config and
        # transforming them into _DictWrapper classes, which are not JSON serializable.
        self._input_features_df = copy.deepcopy(input_features_def)
        self._combiner_def = copy.deepcopy(combiner_def)
        self._output_features_df = copy.deepcopy(output_features_def)

        self._random_seed = random_seed

        if random_seed is not None:
            tf.random.set_seed(random_seed)

        super().__init__()

        # ================ Inputs ================
        self.input_features = build_inputs(
            input_features_def
        )

        # ================ Combiner ================
        logger.debug('Combiner {}'.format(combiner_def[TYPE]))
        combiner_class = get_combiner_class(combiner_def[TYPE])
        self.combiner = combiner_class(
            input_features=self.input_features,
            **combiner_def,
        )

        # ================ Outputs ================
        self.output_features = build_outputs(
            output_features_def,
            self.combiner
        )

        # ================ Combined loss metric ================
        self.eval_loss_metric = tf.keras.metrics.Mean()

        # After constructing all layers, clear the cache to free up memory
        clear_data_cache()


    def get_model_inputs(self, training=True):
        inputs = {
            input_feature_name: input_feature.create_input()
            for input_feature_name, input_feature in
            self.input_features.items()
        }

        if not training:
            return inputs

        targets = {
            output_feature_name: output_feature.create_input()
            for output_feature_name, output_feature in
            self.output_features.items()
        }
        return inputs, targets

    def get_connected_model(self, training=True, inputs=None):
        inputs = inputs or self.get_model_inputs(training)
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def save_savedmodel(self, save_path):
        keras_model = self.get_connected_model(training=False)
        keras_model.save(save_path)

    def call(self, inputs, training=None, mask=None):
        # parameter inputs is a dict feature_name -> tensor / ndarray
        # or
        # parameter (inputs, targets) where
        #   inputs is a dict feature_name -> tensor/ndarray
        #   targets is dict feature_name -> tensor/ndarray

        if isinstance(inputs, tuple):
            inputs, targets = inputs
        else:
            targets = None
        assert inputs.keys() == self.input_features.keys()

        encoder_outputs = {}
        for input_feature_name, input_values in inputs.items():
            encoder = self.input_features[input_feature_name]
            encoder_output = encoder(input_values, training=training,
                                     mask=mask)
            encoder_outputs[input_feature_name] = encoder_output

        combiner_outputs = self.combiner(encoder_outputs)

        output_logits = {}
        output_last_hidden = {}
        for output_feature_name, decoder in self.output_features.items():
            # use presence or absence of targets
            # to signal training or prediction
            decoder_inputs = (combiner_outputs, copy.copy(output_last_hidden))
            if targets is not None:
                # targets are only used during training,
                # during prediction they are omitted
                decoder_inputs = (decoder_inputs, targets[output_feature_name])

            decoder_outputs = decoder(
                decoder_inputs,
                training=training,
                mask=mask
            )
            output_logits[output_feature_name] = decoder_outputs
            output_last_hidden[output_feature_name] = decoder_outputs[
                'last_hidden']

        return output_logits

    def predictions(self, inputs, output_features=None):
        # check validity of output_features
        if output_features is None:
            of_list = self.output_features
        elif isinstance(output_features, str):
            if output_features == 'all':
                of_list = set(self.output_features.keys())
            elif output_features in self.output_features:
                of_list = [output_features]
            else:
                raise ValueError(
                    "'output_features' {} is not a valid for this model. "
                    "Available ones are: {}".format(
                        output_features, set(self.output_features.keys())
                    )
                )
        elif isinstance(output_features, list or set):
            if output_features.issubset(self.output_features):
                of_list = output_features
            else:
                raise ValueError(
                    "'output_features' {} must be a subset of "
                    "available features {}".format(
                        output_features, set(self.output_features.keys())
                    )
                )
        else:
            raise ValueError(
                "'output_features' must be None or a string or a list "
                "of output features"
            )

        outputs = self.call(inputs, training=False)

        predictions = {}
        for of_name in of_list:
            predictions[of_name] = self.output_features[of_name].predictions(
                outputs[of_name],
                training=False
            )

        return predictions

    @tf.function
    def train_step(self, optimizer, inputs, targets,
                   regularization_lambda=0.0):
        with tf.GradientTape() as tape:
            model_outputs = self((inputs, targets), training=True)
            loss, all_losses = self.train_loss(
                targets, model_outputs, regularization_lambda
            )
        optimizer.minimize_with_tape(
            tape, loss, self.trainable_variables
        )
        # grads = tape.gradient(loss, model.trainable_weights)
        # optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss, all_losses

    @tf.function
    def evaluation_step(self, inputs, targets):
        predictions = self.predictions(inputs, output_features=None)
        self.update_metrics(targets, predictions)
        return predictions

    @tf.function
    def predict_step(self, inputs):
        return self.predictions(inputs, output_features=None)

    def train_loss(self, targets, predictions, regularization_lambda=0.0):
        train_loss = 0
        of_train_losses = {}
        for of_name, of_obj in self.output_features.items():
            of_train_loss = of_obj.train_loss(targets[of_name],
                                              predictions[of_name])
            train_loss += of_obj.loss['weight'] * of_train_loss
            of_train_losses[of_name] = of_train_loss
        train_loss += regularization_lambda * sum(
            self.losses)  # regularization / other losses
        return train_loss, of_train_losses

    def eval_loss(self, targets, predictions):
        eval_loss = 0
        of_eval_losses = {}
        for of_name, of_obj in self.output_features.items():
            of_eval_loss = of_obj.eval_loss(
                targets[of_name], predictions[of_name]
            )
            eval_loss += of_obj.loss['weight'] * of_eval_loss
            of_eval_losses[of_name] = of_eval_loss
        eval_loss += sum(self.losses)  # regularization / other losses
        return eval_loss, of_eval_losses

    def update_metrics(self, targets, predictions):
        for of_name, of_obj in self.output_features.items():
            of_obj.update_metrics(targets[of_name], predictions[of_name])
        self.eval_loss_metric.update_state(
            self.eval_loss(targets, predictions)[0]
        )

    def get_metrics(self):
        all_of_metrics = {}
        for of_name, of_obj in self.output_features.items():
            all_of_metrics[of_name] = of_obj.get_metrics()
        all_of_metrics[COMBINED] = {
            LOSS: self.eval_loss_metric.result().numpy()
        }
        return all_of_metrics

    def reset_metrics(self):
        for of_obj in self.output_features.values():
            of_obj.reset_metrics()
        self.eval_loss_metric.reset_states()

    def collect_weights(
            self,
            tensor_names=None,
            **kwargs
    ):
        def recurse_weights(model, prefix=None):
            results = []
            for layer in model.layers:
                layer_prefix = f'{prefix}/{layer.name}' if prefix else layer.name
                if isinstance(layer, tf.keras.Model):
                    results += recurse_weights(layer, layer_prefix)
                else:
                    results += [(f'{layer_prefix}/{w.name}', w) for w in
                                layer.weights]
            return results

        connected_model = self.get_connected_model()
        weights = recurse_weights(connected_model)
        if tensor_names:
            # Check for bad tensor names
            weight_set = set(name for name, w in weights)
            for name in tensor_names:
                if name not in weight_set:
                    raise ValueError(
                        f'Tensor {name} not present in the model graph')

            # Filter the weights
            tensor_set = set(tensor_names)
            weights = [(name, w) for name, w in weights if name in tensor_set]

        return weights

    def __setstate__(self, newstate):
        self.set_weights(newstate['weights'])

    def __reduce__(self):
        args = (self._input_features_df, self._combiner_def, self._output_features_df, self._random_seed)
        state = {'weights': self.get_weights()}
        return ECD, args, state


def build_inputs(
        input_features_def,
        **kwargs
):
    input_features = OrderedDict()
    input_features_def = topological_sort_feature_dependencies(
        input_features_def)
    for input_feature_def in input_features_def:
        input_features[input_feature_def[NAME]] = build_single_input(
            input_feature_def,
            input_features,
            **kwargs
        )
    return input_features


def build_single_input(input_feature_def, other_input_features, **kwargs):
    logger.debug('Input {} feature {}'.format(
        input_feature_def[TYPE],
        input_feature_def[NAME]
    ))

    encoder_obj = None
    if input_feature_def.get(TIED, None) is not None:
        tied_input_feature_name = input_feature_def[TIED]
        if tied_input_feature_name in other_input_features:
            encoder_obj = other_input_features[
                tied_input_feature_name].encoder_obj

    input_feature_class = get_from_registry(
        input_feature_def[TYPE],
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
    output_features_def = topological_sort_feature_dependencies(
        output_features_def)
    output_features = {}

    for output_feature_def in output_features_def:
        output_feature = build_single_output(
            output_feature_def,
            combiner,
            output_features,
            **kwargs
        )
        output_features[output_feature_def[NAME]] = output_feature

    return output_features


def build_single_output(
        output_feature_def,
        feature_hidden,
        other_output_features,
        **kwargs
):
    logger.debug('Output {} feature {}'.format(
        output_feature_def[TYPE],
        output_feature_def[NAME]
    ))

    output_feature_class = get_from_registry(
        output_feature_def[TYPE],
        output_type_registry
    )
    output_feature_obj = output_feature_class(output_feature_def)
    # weighted_train_mean_loss, weighted_eval_loss, output_tensors = output_feature_obj.concat_dependencies_and_build_output(
    #    feature_hidden,
    #    other_output_features,
    #    **kwargs
    # )

    return output_feature_obj
