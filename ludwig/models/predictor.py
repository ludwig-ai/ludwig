import logging
import os
import psutil
import sys
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from pprint import pformat

import torch
from tqdm import tqdm

from ludwig.constants import COMBINED, LOGITS, LAST_HIDDEN
from ludwig.data.dataset.base import Dataset
from ludwig.globals import is_progressbar_disabled
from ludwig.models.ecd import ECD
from ludwig.utils.data_utils import flatten_df, from_numpy_dataset, save_json
from ludwig.utils.horovod_utils import initialize_horovod, return_first
from ludwig.utils.misc_utils import sum_dicts
from ludwig.utils import output_feature_utils
from ludwig.utils.print_utils import repr_ordered_dict
from ludwig.utils.torch_utils import initialize_pytorch

EXCLUDE_PRED_SET = {LOGITS, LAST_HIDDEN}
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
    def predict_single(self, model, batch):
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

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def batch_predict(
            self,
            model: ECD,
            dataset: Dataset,
            dataset_name: str = None,
    ):
        with dataset.initialize_batcher(
            self._batch_size,
            should_shuffle=False,
            horovod=self._horovod
        ) as batcher:

            progress_bar = None
            if self.is_coordinator():
                progress_bar = tqdm(
                    desc='Prediction' if dataset_name is None
                    else 'Prediction {0: <5.5}'.format(dataset_name),
                    total=batcher.steps_per_epoch,
                    file=sys.stdout,
                    disable=is_progressbar_disabled()
                )

            predictions = defaultdict(list)
            while not batcher.last_batch():
                batch = batcher.next_batch()
                preds = self._predict(model, batch)
                self._accumulate_preds(preds, predictions)

                if self.is_coordinator():
                    progress_bar.update(1)

            if self.is_coordinator():
                progress_bar.close()

        # consolidate predictions from each batch to a single tensor
        self._concat_preds(predictions)

        return from_numpy_dataset(predictions)

    def predict_single(self, model, batch):
        predictions = defaultdict(list)
        preds = self._predict(model, batch)
        self._accumulate_preds(preds, predictions)
        self._concat_preds(predictions)
        return from_numpy_dataset(predictions)

    def _predict(self, model, batch):
        inputs = {
            i_feat.feature_name: torch.from_numpy(batch[i_feat.proc_column]).to(self.device)
            for i_feat in model.input_features.values()
        }

        return model.predict_step(inputs)

    def _accumulate_preds(self, preds, predictions):
        # accumulate predictions from batch for each output feature
        for of_name, of_preds in preds.items():
            for pred_name, pred_values in of_preds.items():
                if pred_name not in EXCLUDE_PRED_SET:
                    key = f'{of_name}_{pred_name}'
                    predictions[key].append(pred_values)

    def _concat_preds(self, predictions):
        for key, pred_value_list in predictions.items():
            # Without cloning and detaching, a runtime error is raised since pred_value_list
            # is a tensor that requires grad.
            predictions[key] = torch.cat(
                pred_value_list, dim=0).clone().detach().cpu().numpy()

    def batch_evaluation(
            self,
            model: ECD,
            dataset,
            collect_predictions=False,
            dataset_name=None
    ):
        with dataset.initialize_batcher(
            self._batch_size,
            should_shuffle=False,
            horovod=self._horovod
        ) as batcher:

            progress_bar = None
            if self.is_coordinator():
                progress_bar = tqdm(
                    desc='Evaluation' if dataset_name is None
                    else 'Evaluation {0: <5.5}'.format(dataset_name),
                    total=batcher.steps_per_epoch,
                    file=sys.stdout,
                    disable=is_progressbar_disabled()
                )

            predictions = defaultdict(list)
            while not batcher.last_batch():
                batch = batcher.next_batch()
                logger.debug(
                    f'evaluation for {dataset_name}: obtained next batch '
                    f'memory used: {psutil.Process(os.getpid()).memory_info()[0] / 1e6:0.2f}MB'
                )
                inputs = {
                    i_feat.feature_name: torch.from_numpy(
                        batch[i_feat.proc_column]).to(self.device)
                    for i_feat in model.input_features.values()
                }
                targets = {
                    o_feat.feature_name: torch.from_numpy(
                        batch[o_feat.proc_column]).to(self.device)
                    for o_feat in model.output_features.values()
                }

                preds = model.evaluation_step(inputs, targets)

                # accumulate predictions from batch for each output feature
                if collect_predictions:
                    for of_name, of_preds in preds.items():
                        for pred_name, pred_values in of_preds.items():
                            if pred_name not in EXCLUDE_PRED_SET:
                                key = f'{of_name}_{pred_name}'
                                predictions[key].append(pred_values)

                if self.is_coordinator():
                    progress_bar.update(1)
                    logger.debug(
                        f'evaluation for {dataset_name}: completed batch {progress_bar.n} '
                        f'memory used: {psutil.Process(os.getpid()).memory_info()[0] / 1e6:0.2f}MB'
                    )

            if self.is_coordinator():
                progress_bar.close()

        # consolidate predictions from each batch to a single tensor
        if collect_predictions:
            for key, pred_value_list in predictions.items():
                predictions[key] = torch.cat(
                    pred_value_list, dim=0).clone().detach().cpu().numpy()

        metrics = model.get_metrics()
        metrics = self.merge_workers_metrics(metrics)
        model.reset_metrics()

        return metrics, from_numpy_dataset(predictions)

    def batch_collect_activations(
            self,
            model,
            layer_names,
            dataset,
            bucketing_field=None
    ):
        if bucketing_field:
            raise ValueError('BucketedBatcher is not supported yet')

        activation_model = model

        with dataset.initialize_batcher(
            self._batch_size,
            should_shuffle=False
        ) as batcher:
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
                collected_tensors = [
                    (concat_name, tensor) for concat_name, tensor in outputs.items()]

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
        initialize_pytorch(
            gpus=gpus,
            gpu_memory_limit=gpu_memory_limit,
            allow_parallel_threads=allow_parallel_threads,
            horovod=horovod
        )
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

        feature_df = predictions.loc[:,
                                     predictions.columns.str.startswith(of_name)]
        feature_df = feature_df.rename(columns=lambda c: c[len(of_name) + 1:])

        overall_stats[of_name] = output_feature.calculate_overall_stats(
            feature_df,  # predictions
            dataset.get(output_feature.proc_column),  # target
            feature_metadata,  # output feature metadata
        )
    return overall_stats


def save_prediction_outputs(
        postprocessed_output,
        output_directory,
        backend,
):
    postprocessed_output, column_shapes = flatten_df(
        postprocessed_output, backend
    )
    postprocessed_output.to_parquet(
        os.path.join(output_directory, 'predictions.parquet')
    )
    save_json(
        os.path.join(output_directory, 'predictions.shapes.json'),
        column_shapes
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


def get_output_columns(output_features):
    output_columns = []
    for of_name, feature in output_features.items():
        for pred in feature.get_prediction_set():
            if pred not in EXCLUDE_PRED_SET:
                output_columns.append(f'{of_name}_{pred}')
    return output_columns
