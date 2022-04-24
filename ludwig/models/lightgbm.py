import os
from typing import Iterable, Tuple

import lightgbm as lgb
import torch

from ludwig.constants import BINARY, CATEGORY, COMBINED, LOSS, NUMBER
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.globals import MODEL_WEIGHTS_FILE_NAME, TRAINING_CHECKPOINTS_DIR_PATH
from ludwig.models.hummingbird import convert_to_pytorch
from ludwig.models.trainer import Trainer
from ludwig.modules.metric_modules import get_initial_validation_value
from ludwig.utils.checkpoint_utils import Checkpoint, CheckpointManager
from ludwig.utils.metric_utils import TrainerMetric
from ludwig.utils.trainer_utils import get_new_progress_tracker


def iter_feature_metrics(output_features: LudwigFeatureDict) -> Iterable[Tuple[str, str]]:
    for output_feature_name, output_feature in output_features.items():
        for metric in output_feature.metric_functions:
            yield output_feature_name, metric


class LightGBMTrainer(Trainer):
    def train(self, training_set, validation_set=None, test_set=None, save_path="model", **kwargs):
        # TODO: construct new datasets by running encoders (for text, image)

        # TODO: only single task currently
        if len(self.model.output_features) > 1:
            raise ValueError("Only single task currently supported")

        output_params = {}
        for feature in self.model.output_features.values():
            if feature.type() == CATEGORY:
                output_params = {
                    "objective": "multiclass",
                    "metric": "multi_logloss",
                    "num_class": feature.num_classes,
                }
            elif feature.type() == BINARY:
                output_params = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                }
            elif feature.type() == NUMBER:
                output_params = {
                    "objective": "regression",
                    "metric": {"l2", "l1"},
                }
            else:
                raise ValueError(f"Output feature must be numerical, categorical, or binary, found: {feature.type}")

        # from: https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py
        params = {
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": self.base_learning_rate,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": 0,
            **output_params,
        }

        X_train = training_set.to_df(self.model.input_features.values())
        y_train = training_set.to_df(self.model.output_features.values())

        # create dataset for lightgbm
        # if you want to re-use data, remember to set free_raw_data=False
        lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

        TRAIN_KEY = "training"
        VALID_KEY = "validation"
        TEST_KEY = "test"
        eval_sets = [lgb_train]
        eval_names = [TRAIN_KEY]
        if validation_set is not None:
            X_val = validation_set.to_df(self.model.input_features.values())
            y_val = validation_set.to_df(self.model.output_features.values())
            lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train, free_raw_data=False)
            eval_sets.append(lgb_val)
            eval_names.append(VALID_KEY)

        if test_set is not None:
            X_test = test_set.to_df(self.model.input_features.values())
            y_test = test_set.to_df(self.model.output_features.values())
            lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train, free_raw_data=False)
            eval_sets.append(lgb_test)
            eval_names.append(TEST_KEY)

        progress_tracker = get_new_progress_tracker(
            batch_size=-1,
            learning_rate=self.base_learning_rate,
            best_eval_metric=get_initial_validation_value(self.validation_metric),
            best_reduce_learning_rate_eval_metric=get_initial_validation_value(self.reduce_learning_rate_eval_metric),
            best_increase_batch_size_eval_metric=get_initial_validation_value(self.increase_batch_size_eval_metric),
            output_features=self.model.output_features,
        )

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
            # NOTE: hummingbird does not support categorical features
            # categorical_feature=categorical_features,
            callbacks=[
                lgb.record_evaluation(eval_results),
                lgb.early_stopping(stopping_rounds=self.early_stop),
                lgb.log_evaluation(),
            ],
        )

        for output_feature_name, metric in iter_feature_metrics(self.model.output_features):
            if metric == LOSS:
                progress_tracker.train_metrics[output_feature_name][metric] = [
                    TrainerMetric(epoch=0, step=i, value=v)
                    for i, v in enumerate(eval_results[TRAIN_KEY][output_params["metric"]])
                ]
                if validation_set is not None:
                    progress_tracker.validation_metrics[output_feature_name][metric] = [
                        TrainerMetric(epoch=0, step=i, value=v)
                        for i, v in enumerate(eval_results[VALID_KEY][output_params["metric"]])
                    ]
                if test_set is not None:
                    progress_tracker.test_metrics[output_feature_name][metric] = [
                        TrainerMetric(epoch=0, step=i, value=v)
                        for i, v in enumerate(eval_results[TEST_KEY][output_params["metric"]])
                    ]

        for metrics in [
            progress_tracker.train_metrics,
            progress_tracker.validation_metrics,
            progress_tracker.test_metrics,
        ]:
            metrics[COMBINED][LOSS].append(TrainerMetric(epoch=0, step=0, value=0.0))

        progress_tracker.steps = gbm.current_iteration()
        progress_tracker.last_improvement_steps = gbm.best_iteration
        if validation_set is not None:
            progress_tracker.best_eval_metric = gbm.best_score[VALID_KEY][output_params["metric"]]
        elif test_set is not None:
            progress_tracker.best_eval_metric = gbm.best_score[TEST_KEY][output_params["metric"]]

        # convert to pytorch for inference, fine tuning
        self.model = convert_to_pytorch(gbm, self.model)

        os.makedirs(save_path, exist_ok=True)
        training_checkpoints_path = os.path.join(save_path, TRAINING_CHECKPOINTS_DIR_PATH)
        checkpoint = Checkpoint(model=self.model)
        checkpoint_manager = CheckpointManager(checkpoint, training_checkpoints_path, device=self.device, max_to_keep=1)
        checkpoint_manager.save(1)

        model_weights_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
        torch.save(self.model.state_dict(), model_weights_path)

        return (
            self.model,
            progress_tracker.train_metrics,
            progress_tracker.validation_metrics,
            progress_tracker.test_metrics,
        )
