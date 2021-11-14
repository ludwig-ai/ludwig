import os

import lightgbm as lgb
import torch
from hummingbird.ml import convert

from ludwig.constants import CATEGORY, COMBINED, LOSS, BINARY, NUMERICAL
from ludwig.globals import TRAINING_CHECKPOINTS_DIR_PATH, MODEL_WEIGHTS_FILE_NAME
from ludwig.models.trainer import Trainer
from ludwig.utils.checkpoint_utils import Checkpoint, CheckpointManager


class LightGBMTrainer(Trainer):
    def train(
            self,
            model,
            training_set,
            validation_set=None,
            test_set=None,
            save_path='model',
            **kwargs
    ):
        # TODO: construct new datasets by running encoders (for text, image)

        # TODO: only single task currently
        if len(model.output_features) > 1:
            raise ValueError("Only single task currently supported")

        output_params = {}
        for feature in model.output_features.values():
            if feature.type == CATEGORY:
                output_params = {
                    'objective': 'multiclass',
                    'metric': 'multi_logloss',
                    'num_class': feature.num_classes,
                }
            elif feature.type == BINARY:
                output_params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                }
            elif feature.type == NUMERICAL:
                # TODO: regression
                output_params = {

                }
            else:
                raise ValueError(
                    f"Output feature must be numerical, categorical, or binary, found: {feature.type}"
                )

        # from: https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py
        params = {
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            **output_params
        }

        X_train = training_set.to_df(model.input_features.values())
        y_train = training_set.to_df(model.output_features.values())

        # create dataset for lightgbm
        # if you want to re-use data, remember to set free_raw_data=False
        lgb_train = lgb.Dataset(X_train, y_train,
                                free_raw_data=False)

        eval_sets = []
        if validation_set is not None:
            X_val = validation_set.to_df(model.input_features.values())
            y_val = validation_set.to_df(model.output_features.values())
            lgb_val = lgb.Dataset(X_val, y_val,
                                  reference=lgb_train,
                                  free_raw_data=False)
            eval_sets.append(lgb_val)

        if test_set is not None:
            X_test = test_set.to_df(model.input_features.values())
            y_test = test_set.to_df(model.output_features.values())
            lgb_test = lgb.Dataset(X_test, y_test,
                                   reference=lgb_train,
                                   free_raw_data=False)
            eval_sets.append(lgb_test)

        features_names = list(model.input_features.keys())

        categorical_features = [
            i for i, feature in enumerate(model.input_features.values())
            if feature.type == CATEGORY or feature.type == BINARY
        ]

        # TODO: update training metrics
        (
            train_metrics,
            vali_metrics,
            test_metrics
        ) = self.initialize_training_metrics(model.output_features)

        for output_feature_name, output_feature in model.output_features.items():
            for metric in output_feature.metric_functions:
                train_metrics[output_feature_name][metric].append(0.0)
                vali_metrics[output_feature_name][metric].append(0.0)
                test_metrics[output_feature_name][metric].append(0.0)

        for metrics in [train_metrics, vali_metrics, test_metrics]:
            metrics[COMBINED][LOSS].append(0.0)

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=100,
                        valid_sets=eval_sets,
                        feature_name=features_names,
                        categorical_feature=categorical_features)

        # convert to pytorch for inference, fine tuning
        hb_model = convert(gbm, 'torch')
        model = hb_model.model

        os.makedirs(save_path, exist_ok=True)
        training_checkpoints_path = os.path.join(
            save_path, TRAINING_CHECKPOINTS_DIR_PATH
        )
        checkpoint = Checkpoint(model=model)
        checkpoint_manager = CheckpointManager(
            checkpoint, training_checkpoints_path, device=self.device,
            max_to_keep=1)
        checkpoint_manager.save(1)

        model_weights_path = os.path.join(save_path,
                                          MODEL_WEIGHTS_FILE_NAME)
        torch.save(model.state_dict(), model_weights_path)

        return model, train_metrics, vali_metrics, test_metrics
