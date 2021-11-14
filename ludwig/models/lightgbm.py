from ludwig.constants import CATEGORY
from ludwig.models.trainer import Trainer

import lightgbm as lgb


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
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
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
            if feature.type == CATEGORY
        ]

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=10,
                        valid_sets=eval_sets,
                        feature_name=features_names,
                        categorical_feature=categorical_features)

        # use https://github.com/microsoft/hummingbird to convert to pytorch for inference, fine tuning
        # https://towardsdatascience.com/transform-your-ml-model-to-pytorch-with-hummingbird-da49665497e7
