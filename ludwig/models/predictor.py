import logging
import os
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from pprint import pformat

import tensorflow as tf
from tqdm import tqdm

from ludwig.constants import COMBINED, LOGITS
from ludwig.globals import is_progressbar_disabled
from ludwig.utils.data_utils import save_csv, save_json
from ludwig.utils.horovod_utils import initialize_horovod, return_first
from ludwig.utils.misc_utils import sum_dicts
from ludwig.utils.print_utils import repr_ordered_dict
from ludwig.utils.tf_utils import initialize_tensorflow


EXCLUE_PRED_SET = {LOGITS}
SKIP_EVAL_METRICS = {'confusion_matrix', 'roc_curve'}

logger = logging.getLogger(__name__)


class BasePredictor(ABC):
    @abstractmethod
    def batch_predict(
            self,
            model,
            dataset,
            dataset_name=None
    ):
        raise NotImplementedError()

    @abstractmethod
    def batch_evaluation(
            self,
            model,
            dataset,
            collect_predictions=False,
            dataset_name=None
    ):
        raise NotImplementedError()

    @abstractmethod
    def batch_collect_activations(
            self,
            model,
            layer_names,
            dataset,
            bucketing_field=None
    ):
        raise NotImplementedError()

    # Remote implementations may override this
    def shutdown(self):
        pass

    # Functions needed to treat Trainer as a context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class Predictor(BasePredictor):
    """
    Predictor is a class that uses a model to predict and evaluate
    """

    def __init__(
            self,
            batch_size=128,
            horovod=None,
            debug=False,
            **kwargs
    ):
        self._batch_size = batch_size
        self._horovod = horovod
        self._debug = debug

    def batch_predict(
            self,
            model,
            dataset,
            dataset_name=None
    ):
        batcher = dataset.initialize_batcher(
            self._batch_size,
            should_shuffle=False,
            horovod=self._horovod
        )

        progress_bar = None
        if self.is_coordinator():
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

            inputs = {
                i_feat.feature_name: batch[i_feat.proc_column]
                for i_feat in model.input_features.values()
            }

            preds = model.predict_step(inputs)

            # accumulate predictions from batch for each output feature
            for of_name, of_preds in preds.items():
                if of_name not in predictions:
                    predictions[of_name] = {}
                for pred_name, pred_values in of_preds.items():
                    if pred_name not in EXCLUE_PRED_SET:
                        if pred_name not in predictions[of_name]:
                            predictions[of_name][pred_name] = [pred_values]
                        else:
                            predictions[of_name][pred_name].append(pred_values)

            if self.is_coordinator():
                progress_bar.update(1)

        if self.is_coordinator():
            progress_bar.close()

        # consolidate predictions from each batch to a single tensor
        for of_name, of_predictions in predictions.items():
            for pred_name, pred_value_list in of_predictions.items():
                predictions[of_name][pred_name] = tf.concat(pred_value_list,
                                                            axis=0)

        return predictions

    def batch_evaluation(
            self,
            model,
            dataset,
            collect_predictions=False,
            dataset_name=None
    ):
        batcher = dataset.initialize_batcher(
            self._batch_size,
            should_shuffle=False,
            horovod=self._horovod
        )

        progress_bar = None
        if self.is_coordinator():
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

            inputs = {
                i_feat.feature_name: batch[i_feat.proc_column]
                for i_feat in model.input_features.values()
            }
            targets = {
                o_feat.feature_name: batch[o_feat.proc_column]
                for o_feat in model.output_features.values()
            }

            preds = model.evaluation_step(inputs, targets)

            # accumulate predictions from batch for each output feature
            if collect_predictions:
                for of_name, of_preds in preds.items():
                    if of_name not in predictions:
                        predictions[of_name] = {}
                    for pred_name, pred_values in of_preds.items():
                        if pred_name not in EXCLUE_PRED_SET and pred_values is not None:
                            if pred_name not in predictions[of_name]:
                                predictions[of_name][pred_name] = [pred_values]
                            else:
                                predictions[of_name][pred_name].append(
                                    pred_values)

            if self.is_coordinator():
                progress_bar.update(1)

        if self.is_coordinator():
            progress_bar.close()

        # consolidate predictions from each batch to a single tensor
        if collect_predictions:
            for of_name, of_predictions in predictions.items():
                for pred_name, pred_value_list in of_predictions.items():
                    predictions[of_name][pred_name] = tf.concat(
                        pred_value_list, axis=0
                    )

        metrics = model.get_metrics()
        metrics = self.merge_workers_metrics(metrics)
        model.reset_metrics()

        return metrics, predictions

    def batch_collect_activations(
            self,
            model,
            layer_names,
            dataset,
            bucketing_field=None
    ):
        if bucketing_field:
            raise ValueError('BucketedBatcher is not supported yet')

        # Build static graph for the trained model
        tf.keras.backend.reset_uids()
        keras_model_inputs = model.get_model_inputs(training=False)
        keras_model = model.get_connected_model(inputs=keras_model_inputs,
                                                training=False)

        # Create a new model that routes activations to outputs
        tf.keras.backend.reset_uids()
        output_nodes = {layer_name: keras_model.get_layer(layer_name).output
                        for layer_name in layer_names}
        activation_model = tf.keras.Model(inputs=keras_model_inputs,
                                          outputs=output_nodes)

        batcher = dataset.initialize_batcher(
            self._batch_size,
            should_shuffle=False
        )

        progress_bar = tqdm(
            desc='Collecting Tensors',
            total=batcher.steps_per_epoch,
            file=sys.stdout,
            disable=is_progressbar_disabled()
        )

        collected_tensors = []
        while not batcher.last_batch():
            batch = batcher.next_batch()

            inputs = {
                i_feat.feature_name: batch[i_feat.proc_column]
                for i_feat in model.input_features.values()
            }
            outputs = activation_model(inputs)

            for layer_name, output in outputs.items():
                if isinstance(output, tuple):
                    output = list(output)

                if isinstance(output, tf.Tensor):
                    output = [('', output)]
                elif isinstance(output, dict):
                    output = [(f'_{key}', tensor)
                              for key, tensor in output.items()]
                elif isinstance(output, list):
                    output = [(f'_{idx}', tensor)
                              for idx, tensor in enumerate(output)]

                for suffix, tensor in output:
                    full_name = f'{layer_name}{suffix}'
                    collected_tensors.append((full_name, tensor))

            progress_bar.update(1)

        progress_bar.close()

        return collected_tensors

    def merge_workers_metrics(self, metrics):
        if not self._horovod:
            return metrics

        # gather outputs from all workers
        all_workers_output_metrics = self._horovod.allgather_object(metrics)

        # merge them into a single one
        merged_output_metrics = sum_dicts(
            all_workers_output_metrics,
            dict_type=OrderedDict
        )

        return merged_output_metrics

    def is_coordinator(self):
        if not self._horovod:
            return True
        return self._horovod.rank() == 0


class RemotePredictor(Predictor):
    def __init__(
        self,
        gpus=None,
        gpu_memory_limit=None,
        allow_parallel_threads=True,
        **kwargs
    ):
        horovod = initialize_horovod()
        initialize_tensorflow(gpus=gpus,
                              gpu_memory_limit=gpu_memory_limit,
                              allow_parallel_threads=allow_parallel_threads,
                              horovod=horovod)
        super().__init__(horovod=horovod, **kwargs)

        # Only return results from rank 0 to reduce network overhead
        self.batch_predict = return_first(self.batch_predict)
        self.batch_evaluation = return_first(self.batch_evaluation)


def calculate_overall_stats(
        output_features,
        predictions,
        dataset,
        training_set_metadata
):
    overall_stats = {}
    for of_name, output_feature in output_features.items():
        feature_metadata = output_feature.overall_statistics_metadata()
        feature_metadata.update(
            training_set_metadata[output_feature.feature_name])

        overall_stats[of_name] = output_feature.calculate_overall_stats(
            predictions[of_name],  # predictions
            dataset.get(output_feature.proc_column),  # target
            feature_metadata,  # output feature metadata
        )
    return overall_stats


def save_prediction_outputs(
        postprocessed_output,
        output_directory,
        skip_output_types=None
):
    if skip_output_types is None:
        skip_output_types = set()
    csv_filename = os.path.join(output_directory, '{}_{}.csv')
    for output_field, outputs in postprocessed_output.items():
        for output_type, values in outputs.items():
            if output_type not in skip_output_types:
                save_csv(
                    csv_filename.format(output_field, output_type),
                    values
                )


def save_evaluation_stats(test_stats, output_directory):
    test_stats_fn = os.path.join(
        output_directory,
        'test_statistics.json'
    )
    save_json(test_stats_fn, test_stats)


def print_evaluation_stats(test_stats):
    for output_field, result in test_stats.items():
        if (output_field != COMBINED or
                (output_field == COMBINED and len(test_stats) > 2)):
            logger.info('\n===== {} ====='.format(output_field))
            for metric in sorted(list(result)):
                if metric not in SKIP_EVAL_METRICS:
                    value = result[metric]
                    if isinstance(value, OrderedDict):
                        value_repr = repr_ordered_dict(value)
                    else:
                        value_repr = pformat(result[metric], indent=2)
                    logger.info('{0}: {1}'.format(metric, value_repr))
