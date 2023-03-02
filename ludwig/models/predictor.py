import logging
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from pprint import pformat
from typing import Dict

import numpy as np
import pandas as pd
import psutil
import torch

from ludwig.constants import COMBINED, LAST_HIDDEN, LOGITS
from ludwig.data.dataset.base import Dataset
from ludwig.data.utils import convert_to_dict
from ludwig.distributed.base import DistributedStrategy, LocalStrategy
from ludwig.globals import is_progressbar_disabled, PREDICTIONS_PARQUET_FILE_NAME, TEST_STATISTICS_FILE_NAME
from ludwig.models.base import BaseModel
from ludwig.models.gbm import GBM
from ludwig.progress_bar import LudwigProgressBar
from ludwig.utils.data_utils import save_csv, save_json
from ludwig.utils.dataframe_utils import from_numpy_dataset
from ludwig.utils.print_utils import repr_ordered_dict
from ludwig.utils.strings_utils import make_safe_filename
from ludwig.utils.torch_utils import get_torch_device

EXCLUDE_PRED_SET = {LOGITS, LAST_HIDDEN}
SKIP_EVAL_METRICS = {"confusion_matrix", "roc_curve"}
STATS_SAMPLE_SIZE = 10000

logger = logging.getLogger(__name__)


class BasePredictor(ABC):
    @abstractmethod
    def batch_predict(self, dataset, dataset_name=None):
        raise NotImplementedError()

    @abstractmethod
    def predict_single(self, batch):
        raise NotImplementedError()

    @abstractmethod
    def batch_evaluation(self, dataset, collect_predictions=False, collect_logits=False, dataset_name=None):
        raise NotImplementedError()

    @abstractmethod
    def batch_collect_activations(self, layer_names, dataset, bucketing_field=None):
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
    """Predictor is a class that uses a model to predict and evaluate."""

    def __init__(
        self,
        model: BaseModel,
        batch_size: int = 128,
        distributed: DistributedStrategy = None,
        report_tqdm_to_ray: bool = False,
        **kwargs,
    ):
        self._batch_size = batch_size
        self._distributed = distributed if distributed is not None else LocalStrategy()
        self.report_tqdm_to_ray = report_tqdm_to_ray

        # TODO (jeffkinnison): revert to using the requested device for GBMs when device usage is fixed
        self.device = get_torch_device() if not isinstance(model, GBM) else "cpu"
        self.model = model.to(self.device)

    def batch_predict(self, dataset: Dataset, dataset_name: str = None, collect_logits: bool = False):
        prev_model_training_mode = self.model.training  # store previous model training mode
        self.model.eval()  # set model to eval mode

        with torch.no_grad():
            with dataset.initialize_batcher(self._batch_size, should_shuffle=False) as batcher:
                progress_bar_config = {
                    "desc": "Prediction" if dataset_name is None else f"Prediction {dataset_name: <5.5}",
                    "total": batcher.steps_per_epoch,
                    "file": sys.stdout,
                    "disable": is_progressbar_disabled(),
                }
                progress_bar = LudwigProgressBar(self.report_tqdm_to_ray, progress_bar_config, self.is_coordinator())
                predictions = defaultdict(list)
                while not batcher.last_batch():
                    batch = batcher.next_batch()
                    preds = self._predict(self.model, batch)
                    self._accumulate_preds(
                        preds, predictions, exclude_pred_set={LAST_HIDDEN} if collect_logits else EXCLUDE_PRED_SET
                    )
                    progress_bar.update(1)

                progress_bar.close()

        # consolidate predictions from each batch to a single tensor
        self._concat_preds(predictions)

        self.model.train(prev_model_training_mode)

        return from_numpy_dataset(predictions)

    def predict_single(self, batch, collect_logits: bool = False):
        prev_model_training_mode = self.model.training  # store previous model training mode
        self.model.eval()  # set model to eval mode

        with torch.no_grad():
            predictions = defaultdict(list)
            preds = self._predict(self.model, batch)
            self._accumulate_preds(
                preds, predictions, exclude_pred_set={LAST_HIDDEN} if collect_logits else EXCLUDE_PRED_SET
            )
            self._concat_preds(predictions)

        # reset model to its original training mode
        self.model.train(prev_model_training_mode)
        return from_numpy_dataset(predictions)

    def _predict(self, model: BaseModel, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Predict a batch of data.

        Params:
            model: BaseModel model
            batch: batch of data

        Returns:
            predictions: dictionary of predictions
        """
        inputs = {
            i_feat.feature_name: torch.from_numpy(np.array(batch[i_feat.proc_column], copy=True)).to(self.device)
            for i_feat in model.input_features.values()
        }

        return model.predict_step(inputs)

    def _accumulate_preds(self, preds, predictions, exclude_pred_set=EXCLUDE_PRED_SET):
        # accumulate predictions from batch for each output feature
        for of_name, of_preds in preds.items():
            for pred_name, pred_values in of_preds.items():
                if pred_name not in exclude_pred_set:
                    key = f"{of_name}_{pred_name}"
                    predictions[key].append(pred_values)

    def _concat_preds(self, predictions):
        for key, pred_value_list in predictions.items():
            # Without cloning and detaching, a runtime error is raised since pred_value_list
            # is a tensor that requires grad.
            predictions[key] = torch.cat(pred_value_list, dim=0).clone().detach().cpu().numpy()

    def batch_evaluation(self, dataset, collect_predictions=False, collect_logits=False, dataset_name=None):
        """Batch evaluate model on dataset.

        Params:
            dataset (Union[str, dict, pandas.DataFrame]): source containing the entire dataset to be evaluated.
            collect_predictions: Return model predictions.
            collect_logits: Return model logits and final layer activations.

        Returns:
            Tuple of dictionaries of (metrics, predictions). The keys of metrics are determined by the metrics in the
            model config. The keys of the predictions dictionary depend on which values are requested by the caller:
            collect_predictions, collect_logits.
        """
        prev_model_training_mode = self.model.training  # store previous model training mode
        self.model.eval()  # set model to eval mode

        with torch.no_grad():
            with dataset.initialize_batcher(
                self._batch_size, should_shuffle=False, distributed=self._distributed
            ) as batcher:
                progress_bar_config = {
                    "desc": "Evaluation" if dataset_name is None else f"Evaluation {dataset_name: <5.5}",
                    "total": batcher.steps_per_epoch,
                    "file": sys.stdout,
                    "disable": is_progressbar_disabled(),
                    "position": 0,  # Necessary to disable extra new line artifacts in training logs.
                }
                progress_bar = LudwigProgressBar(self.report_tqdm_to_ray, progress_bar_config, self.is_coordinator())

                predictions = defaultdict(list)
                while not batcher.last_batch():
                    batch = batcher.next_batch()
                    logger.debug(
                        f"evaluation for {dataset_name}: obtained next batch "
                        f"memory used: {psutil.Process(os.getpid()).memory_info()[0] / 1e6:0.2f}MB"
                    )
                    inputs = {
                        i_feat.feature_name: torch.from_numpy(np.array(batch[i_feat.proc_column], copy=True)).to(
                            self.device
                        )
                        for i_feat in self.model.input_features.values()
                    }
                    targets = {
                        o_feat.feature_name: torch.from_numpy(np.array(batch[o_feat.proc_column], copy=True)).to(
                            self.device
                        )
                        for o_feat in self.model.output_features.values()
                    }

                    preds = self.model.evaluation_step(inputs, targets)

                    # accumulate predictions from batch for each output feature
                    if collect_predictions:
                        self._accumulate_preds(
                            preds, predictions, exclude_pred_set={LAST_HIDDEN} if collect_logits else EXCLUDE_PRED_SET
                        )

                    progress_bar.update(1)
                    if self.is_coordinator():
                        logger.debug(
                            f"evaluation for {dataset_name}: completed batch {progress_bar.total_steps} "
                            f"memory used: {psutil.Process(os.getpid()).memory_info()[0] / 1e6:0.2f}MB"
                        )

                progress_bar.close()

            # consolidate predictions from each batch to a single tensor
            if collect_predictions:
                for key, pred_value_list in predictions.items():
                    predictions[key] = torch.cat(pred_value_list, dim=0).clone().detach().cpu().numpy()

            metrics = self.model.get_metrics()
            self.model.reset_metrics()

            self.model.train(prev_model_training_mode)  # Restores previous model training mode.

            return metrics, from_numpy_dataset(predictions)

    def batch_collect_activations(self, layer_names, dataset, bucketing_field=None):
        if bucketing_field:
            raise ValueError("BucketedBatcher is not supported yet")

        prev_model_training_mode = self.model.training  # store previous model training mode
        self.model.eval()  # set model to eval mode

        with torch.no_grad():
            with dataset.initialize_batcher(
                self._batch_size, should_shuffle=False, distributed=self._distributed
            ) as batcher:
                progress_bar_config = {
                    "desc": "Collecting Tensors",
                    "total": batcher.steps_per_epoch,
                    "file": sys.stdout,
                    "disable": is_progressbar_disabled(),
                }
                progress_bar = LudwigProgressBar(self.report_tqdm_to_ray, progress_bar_config, self.is_coordinator())

                collected_tensors = []
                while not batcher.last_batch():
                    batch = batcher.next_batch()

                    inputs = {
                        i_feat.feature_name: torch.from_numpy(np.array(batch[i_feat.proc_column], copy=True)).to(
                            self.device
                        )
                        for i_feat in self.model.input_features.values()
                    }
                    outputs = self.model(inputs)
                    collected_tensors = [(concat_name, tensor) for concat_name, tensor in outputs.items()]
                    progress_bar.update(1)

                progress_bar.close()

        self.model.train(prev_model_training_mode)  # Restores previous model training mode.

        return collected_tensors

    def is_coordinator(self):
        return self._distributed.rank() == 0


class RemotePredictor(Predictor):
    def __init__(self, model: BaseModel, gpus=None, gpu_memory_limit=None, allow_parallel_threads=True, **kwargs):
        super().__init__(model, **kwargs)

        # Only return results from rank 0 to reduce network overhead
        self.batch_predict = self._distributed.return_first(self.batch_predict)
        self.batch_evaluation = self._distributed.return_first(self.batch_evaluation)


def calculate_overall_stats(output_features, predictions, dataset, training_set_metadata):
    overall_stats = {}
    for of_name, output_feature in output_features.items():
        feature_metadata = training_set_metadata[output_feature.feature_name]
        feature_metadata.update(training_set_metadata[output_feature.feature_name])

        feature_df = predictions.loc[:, predictions.columns.str.startswith(of_name)]
        feature_df = feature_df.rename(columns=lambda c: c[len(of_name) + 1 :])

        target = dataset.loc[:, output_feature.proc_column]

        if not isinstance(feature_df, pd.DataFrame):
            logger.warning(
                "Full computation of stats only supported for pandas dataframes. "
                "Sampling the first 10000 rows of the feature and target dataframes for computing overall stats."
            )
            feature_df = feature_df.head(n=STATS_SAMPLE_SIZE, npartitions=-1, compute=True)
            target = target.head(n=STATS_SAMPLE_SIZE, npartitions=-1, compute=True)

        overall_stats[of_name] = output_feature.calculate_overall_stats(
            feature_df,  # predictions
            target,
            feature_metadata,  # output feature metadata
        )
    return overall_stats


def save_prediction_outputs(
    postprocessed_output,
    output_features,
    output_directory,
    backend,
):
    backend.df_engine.write_predictions(
        postprocessed_output, os.path.join(output_directory, PREDICTIONS_PARQUET_FILE_NAME)
    )
    if not backend.df_engine.partitioned:
        # csv can only be written out for unpartitioned df format (i.e., pandas)
        postprocessed_dict = convert_to_dict(postprocessed_output, output_features)
        csv_filename = os.path.join(output_directory, "{}_{}.csv")
        for output_field, outputs in postprocessed_dict.items():
            for output_name, values in outputs.items():
                save_csv(csv_filename.format(output_field, make_safe_filename(output_name)), values)


def save_evaluation_stats(test_stats, output_directory):
    test_stats_fn = os.path.join(output_directory, TEST_STATISTICS_FILE_NAME)
    save_json(test_stats_fn, test_stats)


def print_evaluation_stats(test_stats):
    for output_field, result in test_stats.items():
        if output_field != COMBINED or (output_field == COMBINED and len(test_stats) > 2):
            logger.info(f"\n===== {output_field} =====")
            for metric in sorted(list(result)):
                if metric not in SKIP_EVAL_METRICS:
                    value = result[metric]
                    if isinstance(value, OrderedDict):
                        value_repr = repr_ordered_dict(value)
                    else:
                        value_repr = pformat(result[metric], indent=2)
                    logger.info(f"{metric}: {value_repr}")


def get_output_columns(output_features, include_logits: bool = False):
    output_columns = []
    for of_name, feature in output_features.items():
        for pred in feature.get_prediction_set():
            if pred not in EXCLUDE_PRED_SET or (pred == LOGITS and include_logits):
                output_columns.append(f"{of_name}_{pred}")
    return output_columns
