import os
from typing import Dict, Iterable, List, Optional, Tuple

import lightgbm as lgb
import lightgbm_ray as lgb_ray
import numpy as np
import torch
from hummingbird.ml import convert
from lightgbm_ray.tune import _TuneLGBMRank0Mixin

from ludwig.constants import BINARY, CATEGORY, COMBINED, LOSS, MODEL_GBM, NUMBER
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.globals import MODEL_WEIGHTS_FILE_NAME, TRAINING_CHECKPOINTS_DIR_PATH
from ludwig.models.gbm import GBM
from ludwig.modules.metric_modules import get_initial_validation_value
from ludwig.schema.trainer import TrainerConfig
from ludwig.trainers.base import BaseTrainer
from ludwig.trainers.registry import register_ray_trainer, register_trainer
from ludwig.utils.checkpoint_utils import Checkpoint, CheckpointManager
from ludwig.utils.metric_utils import TrainerMetric
from ludwig.utils.trainer_utils import get_new_progress_tracker


def convert_to_pytorch(gbm: lgb.Booster, tree_module: GBM) -> GBM:
    """Convert a LightGBM model to a PyTorch model and return correspondig GBM."""
    hb_model = convert(gbm, "torch")
    model = hb_model.model
    tree_module.set_compiled_model(model)
    return tree_module


def iter_feature_metrics(features: LudwigFeatureDict) -> Iterable[Tuple[str, str]]:
    """Helper for iterating feature names and metric names."""
    for feature_name, feature in features.items():
        for metric in feature.metric_functions:
            yield feature_name, metric


def lgb_accuracy(preds: np.array, train_data: lgb.Dataset) -> Tuple[str, float, bool]:
    """Calculate accuracy for LightGBM predictions.

    Args:
        preds: LightGBM predictions.
        train_data: LightGBM dataset.

    Returns:
        Tuple of (metric name, metric value, is_higher_better).
    """
    labels = train_data.get_label()
    return "accuracy", np.mean(labels == (preds > 0.5)), True


@register_trainer("lightgbm_trainer", MODEL_GBM, default=True)
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

        self.random_seed = random_seed

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
        self.boosting_type = config.boosting_type
        self.tree_learner = config.tree_learner
        self.num_boost_round = config.num_boost_round
        self.max_depth = config.max_depth
        self.num_leaves = config.num_leaves
        self.min_data_in_leaf = config.min_data_in_leaf
        self.min_sum_hessian_in_leaf = config.min_sum_hessian_in_leaf
        self.feature_fraction = config.feature_fraction
        self.bagging_fraction = config.bagging_fraction
        self.pos_bagging_fraction = config.pos_bagging_fraction
        self.neg_bagging_fraction = config.neg_bagging_fraction
        self.bagging_seed = config.bagging_seed
        self.bagging_freq = config.bagging_freq
        self.feature_fraction_bynode = config.feature_fraction_bynode
        self.feature_fraction_seed = config.feature_fraction_seed
        self.extra_trees = config.extra_trees
        self.extra_seed = config.extra_seed
        self.max_delta_step = config.max_delta_step
        self.lambda_l1 = config.lambda_l1
        self.lambda_l2 = config.lambda_l2
        self.linear_lambda = config.linear_lambda
        self.min_gain_to_split = config.min_gain_to_split
        self.drop_rate = config.drop_rate
        self.max_drop = config.max_drop
        self.skip_drop = config.skip_drop
        self.xgboost_dart_mode = config.xgboost_dart_mode
        self.uniform_drop = config.uniform_drop
        self.drop_seed = config.drop_seed
        self.top_rate = config.top_rate
        self.other_rate = config.other_rate
        self.min_data_per_group = config.min_data_per_group
        self.max_cat_threshold = config.max_cat_threshold
        self.cat_l2 = config.cat_l2
        self.cat_smooth = config.cat_smooth
        self.max_cat_to_onehot = config.max_cat_to_onehot
        self.cegb_tradeoff = config.cegb_tradeoff
        self.cegb_penalty_split = config.cegb_penalty_split
        self.path_smooth = config.path_smooth
        self.verbose = config.verbose
        self.max_bin = config.max_bin

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
            num_boost_round=self.num_boost_round,
            valid_sets=eval_sets,
            valid_names=eval_names,
            feature_name=list(self.model.input_features.keys()),
            evals_result=eval_results,
            feval=[lgb_accuracy],
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
        gbm_sklearn_cls = lgb.LGBMRegressor if output_params["objective"] == "regression" else lgb.LGBMClassifier
        gbm_sklearn = gbm_sklearn_cls(feature_name=list(self.model.input_features.keys()), **params)
        gbm_sklearn._Booster = gbm
        gbm_sklearn.fitted_ = True
        gbm_sklearn._n_features = len(self.model.input_features)
        if isinstance(gbm_sklearn, lgb.LGBMClassifier):
            gbm_sklearn._n_classes = output_params["num_class"] if output_params["objective"] == "multiclass" else 2
        self.model = convert_to_pytorch(gbm_sklearn, self.model)
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
                    "metric": ["binary_logloss", "accuracy", "auc"],
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
            "boosting_type": self.boosting_type,
            "num_leaves": self.num_leaves,
            "learning_rate": self.base_learning_rate,
            "max_depth": self.max_depth,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "pos_bagging_fraction": self.pos_bagging_fraction,
            "neg_bagging_fraction": self.neg_bagging_fraction,
            "bagging_seed": self.bagging_seed,
            "bagging_freq": self.bagging_freq,
            "feature_fraction_bynode": self.feature_fraction_bynode,
            "feature_fraction_seed": self.feature_fraction_seed,
            "extra_trees": self.extra_trees,
            "extra_seed": self.extra_seed,
            "max_delta_step": self.max_delta_step,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "linear_lambda": self.linear_lambda,
            "min_gain_to_split": self.min_gain_to_split,
            "drop_rate": self.drop_rate,
            "max_drop": self.max_drop,
            "skip_drop": self.skip_drop,
            "xgboost_dart_mode": self.xgboost_dart_mode,
            "uniform_drop": self.uniform_drop,
            "drop_seed": self.drop_seed,
            "top_rate": self.top_rate,
            "other_rate": self.other_rate,
            "min_data_per_group": self.min_data_per_group,
            "max_cat_threshold": self.max_cat_threshold,
            "cat_l2": self.cat_l2,
            "cat_smooth": self.cat_smooth,
            "max_cat_to_onehot": self.max_cat_to_onehot,
            "cegb_tradeoff": self.cegb_tradeoff,
            "cegb_penalty_split": self.cegb_penalty_split,
            "path_smooth": self.path_smooth,
            "verbose": self.verbose,
            "max_bin": self.max_bin,
            "tree_learner": self.tree_learner,
            "min_data_in_leaf": self.min_data_in_leaf,
            "min_sum_hessian_in_leaf": self.min_sum_hessian_in_leaf,
            "seed": self.random_seed,
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


@register_ray_trainer("lightgbm_ray_trainer", MODEL_GBM, default=True)
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
        trainer_kwargs: Optional[Dict] = None,
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

        self.ray_kwargs = trainer_kwargs or {}

    def train(
        self,
        training_set: "RayDataset",  # noqa: F821
        validation_set: Optional["RayDataset"] = None,  # noqa: F821
        test_set: Optional["RayDataset"] = None,  # noqa: F821
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
        training_set: "RayDataset",  # noqa: F821
        validation_set: Optional["RayDataset"] = None,  # noqa: F821
        test_set: Optional["RayDataset"] = None,  # noqa: F821
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
