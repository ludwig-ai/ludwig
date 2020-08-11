import copy
import logging
import sys
from collections import OrderedDict

import tensorflow as tf
from tqdm import tqdm

from ludwig.combiners.combiners import get_combiner_class
from ludwig.constants import TIED, LOSS, COMBINED, TYPE, LOGITS, LAST_HIDDEN
from ludwig.features.feature_registries import input_type_registry, \
    output_type_registry
from ludwig.globals import is_on_master, is_progressbar_disabled
from ludwig.utils.algorithms_utils import topological_sort_feature_dependencies
from ludwig.utils.batcher import initialize_batcher
from ludwig.utils.data_utils import clear_data_cache
from ludwig.utils.misc_utils import get_from_registry

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

        # ================ Keep Definition ================
        self.input_features_def = input_features_def
        self.combiner_def = combiner_def
        self.output_features_def = output_features_def

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
            **kwargs
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

    @tf.function
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
            # use presence or absence of targets to signal training or prediction
            if targets is not None:
                # doing training
                target_to_use = tf.cast(targets[output_feature_name],
                                        dtype=tf.int32)
            else:
                # doing prediction
                target_to_use = None

            decoder_logits, decoder_last_hidden = decoder(
                (
                    (combiner_outputs, output_last_hidden),
                    target_to_use
                ),
                training=training,
                mask=mask
            )
            output_logits[output_feature_name] = {}
            output_logits[output_feature_name][LOGITS] = decoder_logits
            output_logits[output_feature_name][
                LAST_HIDDEN] = decoder_last_hidden
            output_last_hidden[output_feature_name] = decoder_last_hidden

        return output_logits

    @tf.function
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

        logits = self.call(inputs, training=False)

        predictions = {}
        for of_name in of_list:
            predictions[of_name] = self.output_features[of_name].predictions(
                logits[of_name],
                training=False
            )

        return predictions

    @tf.function
    def train_step(self, optimizer, inputs, targets,
                   regularization_lambda=0.0):
        with tf.GradientTape() as tape:
            logits = self((inputs, targets), training=True)
            loss, all_losses = self.train_loss(
                targets, logits, regularization_lambda
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

    @tf.function
    def collect_activations_step(self, inputs):
        # todo tf2: to implement
        pass

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

    def batch_predict(
            self,
            dataset,
            batch_size,
            horovod=None,
            dataset_name=None
    ):
        batcher = initialize_batcher(
            dataset, batch_size,
            should_shuffle=False,
            horovod=horovod
        )

        if is_on_master():
            progress_bar = tqdm(
                desc='Prediction' if dataset_name is None
                else 'Prediction {0: <5.5}'.format(dataset_name),
                total=batcher.steps_per_epoch,
                file=sys.stdout,
                disable=is_progressbar_disabled()
            )

        predictions = {}
        while not batcher.last_batch():
            batch = batcher.next_batch()

            inputs = {i_feat.feature_name: batch[i_feat.feature_name]
                      for i_feat in self.input_features}

            preds = self.predict_step(inputs)

            # accumulate predictions from batch for each output feature
            for of_name, of_preds in preds.items():
                if of_name not in predictions:
                    predictions[of_name] = {}
                for pred_name, pred_values in of_preds.items():
                    if pred_name not in predictions[of_name]:
                        predictions[of_name][pred_name] = [pred_values]
                    else:
                        predictions[of_name][pred_name].append(pred_values)

            if is_on_master():
                progress_bar.update(1)

        if is_on_master():
            progress_bar.close()

        # consolidate predictions from each batch to a single tensor
        for of_name, of_predictions in predictions.items():
            for pred_name, pred_value_list in of_predictions.items():
                predictions[of_name][pred_name] = tf.concat(pred_value_list,
                                                            axis=0)

        return predictions

    def batch_evaluation(
            self,
            dataset,
            batch_size,
            collect_predictions=False,
            horovod=None,
            dataset_name=None
    ):
        batcher = initialize_batcher(
            dataset, batch_size,
            should_shuffle=False,
            horovod=horovod
        )

        if is_on_master():
            progress_bar = tqdm(
                desc='Evaluation' if dataset_name is None
                else 'Evaluation {0: <5.5}'.format(dataset_name),
                total=batcher.steps_per_epoch,
                file=sys.stdout,
                disable=is_progressbar_disabled()
            )

        predictions = {}
        while not batcher.last_batch():
            batch = batcher.next_batch()

            inputs = {i_feat.feature_name: batch[i_feat.feature_name]
                      for i_feat in self.input_features}
            targets = {o_feat.feature_name: batch[o_feat.feature_name]
                       for o_feat in self.output_features}

            preds = self.evaluation_step(inputs, targets)

            # accumulate predictions from batch for each output feature
            if collect_predictions:
                for of_name, of_preds in preds.items():
                    if of_name not in predictions:
                        predictions[of_name] = {}
                    for pred_name, pred_values in of_preds.items():
                        if pred_name not in predictions[of_name]:
                            predictions[of_name][pred_name] = [pred_values]
                        else:
                            predictions[of_name][pred_name].append(pred_values)

            if is_on_master():
                progress_bar.update(1)

        if is_on_master():
            progress_bar.close()

        # consolidate predictions from each batch to a single tensor
        if collect_predictions:
            for of_name, of_predictions in predictions.items():
                for pred_name, pred_value_list in of_predictions.items():
                    predictions[of_name][pred_name] = tf.concat(
                        pred_value_list, axis=0
                    )

        metrics = self.model.get_metrics()
        if self._horovod:
            metrics = self.merge_workers_metrics(metrics)
        self.model.reset_metrics()

        if collect_predictions:
            return metrics, predictions
        else:
            return metrics

    # todo tf2: reintroduce this functionality
    def batch_collect_activations(
            self,
            dataset,
            batch_size,
            tensor_names,
            horovod=None
    ):
        # output_nodes = {tensor_name: self.graph.get_tensor_by_name(tensor_name)
        #                 for tensor_name in tensor_names}
        # collected_tensors = {tensor_name: [] for tensor_name in tensor_names}

        batcher = initialize_batcher(
            dataset, batch_size,
            should_shuffle=False,
            horovod=horovod
        )

        progress_bar = tqdm(
            desc='Collecting Tensors',
            total=batcher.steps_per_epoch,
            file=sys.stdout,
            disable=is_progressbar_disabled()
        )

        while not batcher.last_batch():
            batch = batcher.next_batch()

            self.collect_activations_step(batch)
            # result = session.run(
            #     output_nodes,
            #     feed_dict=self.feed_dict(
            #         batch,
            #         is_training=False
            #     )
            # )
            #
            # for tensor_name in result:
            #     for row in result[tensor_name]:
            #         collected_tensors[tensor_name].append(row)

            progress_bar.update(1)

        progress_bar.close()

        collected_tensors = None
        return collected_tensors

    # todo tf2: reintroduce this functionality
    def collect_weights(
            self,
            tensor_names,
            **kwargs
    ):
        # if self.session is None:
        #     session = self.initialize_session(gpus, gpu_fraction)
        #
        #     # load parameters
        #     if self.weights_save_path:
        #         self.restore(session, self.weights_save_path)
        # else:
        #     session = self.session
        #
        # operation_names = set(
        #     [t.name for op in self.graph.get_operations() for t in op.values()]
        # )
        # for tensor_name in tensor_names:
        #     if tensor_name not in operation_names:
        #         raise ValueError(
        #             'Tensor / operation {} not present in the '
        #             'model graph'.format(tensor_name)
        #         )
        #
        # # collect tensors
        # collected_tensors = {
        #     tensor_name: session.run(self.graph.get_tensor_by_name(tensor_name))
        #     for tensor_name in tensor_names
        # }
        #
        # return collected_tensors
        pass

    def save_savedmodel(self, save_path):
        self.model.save(save_path)

    def get_definition(self):
        return {
            'input_features': self.input_features_def,
            'combiner': self.cobiner_def,
            'output_features': self.output_features_def,
        }

    def save_definition(self, save_path):
        # removing pretrained embeddings paths from hyperparameters
        # because the weights are already saved in the model, no need to reload
        # from their path when loading the model next time

        definition = copy.deepcopy(self.get_definition())
        for feature in (definition['input_features'] +
                        definition['output_features']):
            if 'pretrained_embeddings' in feature:
                feature['pretrained_embeddings'] = None
        save_json(save_path, definition, sort_keys=True, indent=4)


def build_inputs(
        input_features_def,
        **kwargs
):
    input_features = OrderedDict()
    input_features_def = topological_sort_feature_dependencies(
        input_features_def)
    for input_feature_def in input_features_def:
        input_features[input_feature_def['name']] = build_single_input(
            input_feature_def,
            input_features,
            **kwargs
        )
    return input_features


def build_single_input(input_feature_def, other_input_features, **kwargs):
    logger.debug('Input {} feature {}'.format(
        input_feature_def[TYPE],
        input_feature_def['name']
    ))

    # todo tf2: tied encoder mechanism needs to be tested
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
        output_features[output_feature_def['name']] = output_feature

    return output_features


def build_single_output(
        output_feature_def,
        feature_hidden,
        other_output_features,
        **kwargs
):
    logger.debug('Output {} feature {}'.format(
        output_feature_def[TYPE],
        output_feature_def['name']
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
