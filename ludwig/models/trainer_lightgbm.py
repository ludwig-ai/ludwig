import os
from typing import Dict, Iterable, List, Optional, Tuple

import lightgbm as lgb
import lightgbm_ray as lgb_ray
import torch
from hummingbird.ml import convert
from lightgbm_ray.tune import _TuneLGBMRank0Mixin

from ludwig.constants import BINARY, CATEGORY, COMBINED, LOSS, NUMBER
from ludwig.data.dataset.ray import RayDataset
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.globals import MODEL_WEIGHTS_FILE_NAME, TRAINING_CHECKPOINTS_DIR_PATH
from ludwig.models.gbm import GBM
from ludwig.models.trainer import BaseTrainer, TrainerConfig
from ludwig.modules.metric_modules import get_initial_validation_value
from ludwig.utils.checkpoint_utils import Checkpoint, CheckpointManager
from ludwig.utils.metric_utils import TrainerMetric
from ludwig.utils.trainer_utils import get_new_progress_tracker


def convert_to_pytorch(gbm: lgb.Booster, tree_module: GBM):
    """Convert a LightGBM model to a PyTorch model."""
    hb_model = convert(gbm, "torch")
    model = hb_model.model
    tree_module.set_compiled_model(model)
    return tree_module


def iter_feature_metrics(output_features: LudwigFeatureDict) -> Iterable[Tuple[str, str]]:
    for output_feature_name, output_feature in output_features.items():
        for metric in output_feature.metric_functions:
            yield output_feature_name, metric


class LightGBMTrainer(BaseTrainer):
    TRAIN_KEY = "training"
    VALID_KEY = "validation"
    TEST_KEY = "test"

    def __init__(
        self,
        config: TrainerConfig,
        model: GBM,
        resume: float = False,
        skip_save_model: bool = False,
        skip_save_progress: bool = False,
        skip_save_log: bool = False,
        callbacks: List = None,
        random_seed: float = ...,
        horovod: Optional[Dict] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.model = model

        self._validation_field = config.validation_field
        self._validation_metric = config.validation_metric
        self.reduce_learning_rate_eval_metric = config.reduce_learning_rate_eval_metric
        self.increase_batch_size_eval_metric = config.increase_batch_size_eval_metric
        try:
            base_learning_rate = float(config.learning_rate)
        except ValueError:
            # TODO (ASN): Circle back on how we want to set default placeholder value
            base_learning_rate = 0.001  # Default initial learning rate for autoML.
        self.base_learning_rate = base_learning_rate
        self.early_stop = config.early_stop

        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.progress_tracker = get_new_progress_tracker(
            batch_size=-1,
            learning_rate=self.base_learning_rate,
            best_eval_metric=get_initial_validation_value(self.validation_metric),
            best_reduce_learning_rate_eval_metric=get_initial_validation_value(self.reduce_learning_rate_eval_metric),
            best_increase_batch_size_eval_metric=get_initial_validation_value(self.increase_batch_size_eval_metric),
            output_features=self.model.output_features,
        )

    def train_online(
        self,
        dataset,
    ):
        raise NotImplementedError()

    @property
    def validation_field(self):
        return self._validation_field

    @property
    def validation_metric(self):
        return self._validation_metric

    def train(self, training_set, validation_set=None, test_set=None, save_path="model", **kwargs):
        # TODO: construct new datasets by running encoders (for text, image)

        # TODO: only single task currently
        if len(self.model.output_features) > 1:
            raise ValueError("Only single task currently supported")

        output_params, params = self._construct_lgb_params()

        lgb_train, eval_sets, eval_names = self._construct_lgb_datasets(training_set, validation_set, test_set)

        # categorical_features = [
        #     i
        #     for i, feature in enumerate(self.model.input_features.values())
        #     if feature.type() == CATEGORY or feature.type() == BINARY
        # ]

        eval_results = dict()
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=100,
            valid_sets=eval_sets,
            valid_names=eval_names,
            feature_name=list(self.model.input_features.keys()),
            evals_result=eval_results,
            # NOTE: hummingbird does not support categorical features
            # categorical_feature=categorical_features,
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stop),
                lgb.log_evaluation(),
            ],
        )

        self._update_progress(
            output_params,
            eval_results,
            gbm,
            update_valid=(validation_set is not None),
            update_test=(test_set is not None),
        )

        # convert to pytorch for inference, fine tuning
        self.model = convert_to_pytorch(gbm, self.model)
        self.model = self.model.to(self.device)

        self._save(save_path)

        return (
            self.model,
            self.progress_tracker.train_metrics,
            self.progress_tracker.validation_metrics,
            self.progress_tracker.test_metrics,
        )

    def _construct_lgb_params(self) -> Tuple[dict, dict]:
        output_params = {}
        for feature in self.model.output_features.values():
            if feature.type() == CATEGORY:
                output_params = {
                    "objective": "multiclass",
                    "metric": ["multi_logloss", "auc_mu"],
                    "num_class": feature.num_classes,
                }
            elif feature.type() == BINARY:
                output_params = {
                    "objective": "binary",
                    "metric": ["binary_logloss", "auc"],
                }
            elif feature.type() == NUMBER:
                output_params = {
                    "objective": "regression",
                    "metric": ["l2", "l1"],
                }
            else:
                raise ValueError(f"Output feature must be numerical, categorical, or binary, found: {feature.type}")

        # from: https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py
        params = {
            "boosting_type": "gbdt",
            "num_leaves": 255,
            "learning_rate": self.base_learning_rate,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": 0,
            "max_bin": 255,
            "tree_learner": "serial",
            "min_data_in_leaf": 1,
            "min_sum_hessian_in_leaf": 100,
            **output_params,
        }

        return output_params, params

    def _construct_lgb_datasets(
        self, training_set, validation_set, test_set
    ) -> Tuple[lgb.Dataset, List[lgb.Dataset], List[str]]:
        X_train = training_set.to_df(self.model.input_features.values())
        y_train = training_set.to_df(self.model.output_features.values())

        # create dataset for lightgbm
        # if you want to re-use data, remember to set free_raw_data=False
        lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

        eval_sets = [lgb_train]
        eval_names = [LightGBMTrainer.TRAIN_KEY]
        if validation_set is not None:
            X_val = validation_set.to_df(self.model.input_features.values())
            y_val = validation_set.to_df(self.model.output_features.values())
            lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, free_raw_data=False)
            eval_sets.append(lgb_val)
            eval_names.append(LightGBMTrainer.VALID_KEY)

        if test_set is not None:
            X_test = test_set.to_df(self.model.input_features.values())
            y_test = test_set.to_df(self.model.output_features.values())
            lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train, free_raw_data=False)
            eval_sets.append(lgb_test)
            eval_names.append(LightGBMTrainer.TEST_KEY)

        return lgb_train, eval_sets, eval_names

    def _update_progress(
        self,
        output_params: dict,
        eval_results: dict,
        gbm: lgb.Booster,
        update_valid: Optional[bool] = False,
        update_test: Optional[bool] = False,
    ):
        for output_feature_name, metric in iter_feature_metrics(self.model.output_features):
            if metric == LOSS:
                self.progress_tracker.train_metrics[output_feature_name][metric] = [
                    TrainerMetric(epoch=0, step=i, value=v)
                    for i, v in enumerate(eval_results[LightGBMTrainer.TRAIN_KEY][output_params["metric"][0]])
                ]
                if update_valid:
                    self.progress_tracker.validation_metrics[output_feature_name][metric] = [
                        TrainerMetric(epoch=0, step=i, value=v)
                        for i, v in enumerate(eval_results[LightGBMTrainer.VALID_KEY][output_params["metric"][0]])
                    ]
                if update_test:
                    self.progress_tracker.test_metrics[output_feature_name][metric] = [
                        TrainerMetric(epoch=0, step=i, value=v)
                        for i, v in enumerate(eval_results[LightGBMTrainer.TEST_KEY][output_params["metric"][0]])
                    ]

        for metrics in [
            self.progress_tracker.train_metrics,
            self.progress_tracker.validation_metrics,
            self.progress_tracker.test_metrics,
        ]:
            metrics[COMBINED][LOSS].append(TrainerMetric(epoch=0, step=0, value=0.0))

        self.progress_tracker.steps = gbm.current_iteration()
        self.progress_tracker.last_improvement_steps = gbm.best_iteration
        if update_valid:
            self.progress_tracker.best_eval_metric = gbm.best_score[LightGBMTrainer.VALID_KEY][
                output_params["metric"][0]
            ]
        elif update_test:
            self.progress_tracker.best_eval_metric = gbm.best_score[LightGBMTrainer.TEST_KEY][
                output_params["metric"][0]
            ]

    def _save(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        training_checkpoints_path = os.path.join(save_path, TRAINING_CHECKPOINTS_DIR_PATH)
        checkpoint = Checkpoint(model=self.model)
        checkpoint_manager = CheckpointManager(checkpoint, training_checkpoints_path, device=self.device, max_to_keep=1)
        checkpoint_manager.save(1)

        model_weights_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
        torch.save(self.model.state_dict(), model_weights_path)


class LogEvalDistributed(_TuneLGBMRank0Mixin):
    def __init__(self, period: int, show_stdv: bool = True):
        self.period = period
        self.show_stdv = show_stdv

    def __call__(self, env: lgb.callback.CallbackEnv):
        if not self.is_rank_0:
            return
        if self.period > 0 and env.evaluation_result_list and (env.iteration + 1) % self.period == 0:
            result = "\t".join(
                [lgb.callback._format_eval_result(x, self.show_stdv) for x in env.evaluation_result_list]
            )
            lgb.callback._log_info(f"[{env.iteration + 1}]\t{result}")


class LightGBMRayTrainer(LightGBMTrainer):
    def __init__(
        self,
        config: TrainerConfig,
        model: GBM,
        resume: float = False,
        skip_save_model: bool = False,
        skip_save_progress: bool = False,
        skip_save_log: bool = False,
        callbacks: List = None,
        random_seed: float = ...,
        horovod: Optional[Dict] = None,
        device: Optional[str] = None,
        ray_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(
            config,
            model,
            resume,
            skip_save_model,
            skip_save_progress,
            skip_save_log,
            callbacks,
            random_seed,
            horovod,
            device,
            **kwargs,
        )

        self.ray_kwargs = ray_kwargs or {}

    def train(
        self,
        training_set: RayDataset,
        validation_set: Optional[RayDataset] = None,
        test_set: Optional[RayDataset] = None,
        save_path="model",
        **kwargs,
    ):
        # TODO: construct new datasets by running encoders (for text, image)

        # TODO: only single task currently
        if len(self.model.output_features) > 1:
            raise ValueError("Only single task currently supported")

        output_params, params = self._construct_lgb_params()

        lgb_train, eval_sets, eval_names = self._construct_lgb_datasets(training_set, validation_set, test_set)

        # categorical_features = [
        #     i
        #     for i, feature in enumerate(self.model.input_features.values())
        #     if feature.type() == CATEGORY or feature.type() == BINARY
        # ]

        eval_results = dict()

        gbm = lgb_ray.train(
            params,
            lgb_train,
            num_boost_round=500,  # TODO: add as config param
            valid_sets=eval_sets,
            valid_names=eval_names,
            feature_name=["index"] + list(self.model.input_features.keys()) + ["split"],
            evals_result=eval_results,
            # NOTE: hummingbird does not support categorical features
            # categorical_feature=categorical_features,
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stop),
                LogEvalDistributed(10),
            ],
            ray_params=lgb_ray.RayParams(**self.ray_kwargs),
        )

        self._update_progress(
            output_params,
            eval_results,
            gbm.booster_,
            update_valid=(validation_set is not None),
            update_test=(test_set is not None),
        )

        # convert to pytorch for inference, fine tuning
        self.model = convert_to_pytorch(gbm.booster_, self.model)

        self._save(save_path)

        return (
            self.model,
            self.progress_tracker.train_metrics,
            self.progress_tracker.validation_metrics,
            self.progress_tracker.test_metrics,
        )

    def _construct_lgb_datasets(
        self,
        training_set: RayDataset,
        validation_set: Optional[RayDataset] = None,
        test_set: Optional[RayDataset] = None,
    ) -> Tuple[lgb_ray.RayDMatrix, List[lgb_ray.RayDMatrix], List[str]]:
        label_col = self.model.output_features.values()[0].proc_column
        # feat_names = list(self.model.input_features.keys()) + [label_col]

        lgb_train = lgb_ray.RayDMatrix(training_set.ds, label=label_col, distributed=False)

        eval_sets = [lgb_train]
        eval_names = [LightGBMTrainer.TRAIN_KEY]
        if validation_set is not None:
            lgb_val = lgb_ray.RayDMatrix(validation_set.ds, label=label_col, distributed=False)
            eval_sets.append(lgb_val)
            eval_names.append(LightGBMTrainer.VALID_KEY)

        if test_set is not None:
            lgb_test = lgb_ray.RayDMatrix(test_set.ds, label=label_col, distributed=False)
            eval_sets.append(lgb_test)
            eval_names.append(LightGBMTrainer.TEST_KEY)

        return lgb_train, eval_sets, eval_names
