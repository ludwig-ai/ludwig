import logging
import os

import numpy as np
import ray

from ludwig.api import LudwigModel
from ludwig.datasets import adult_census_income, mnist

# Loads the dataset as a pandas.DataFrame
train_df, test_df, _ = adult_census_income.load(split=True)

# define model configuration
config = {
    "combiner": {"dropout": 0.2, "num_fc_layers": 3, "output_size": 128, "type": "concat"},
    "input_features": [
        {"name": "age", "type": "number"},
        {"name": "workclass", "type": "category"},
        {"name": "fnlwgt", "type": "number"},
        {"name": "education", "type": "category"},
        {"name": "education-num", "type": "number"},
        {"name": "marital-status", "type": "category"},
        {"name": "occupation", "type": "category"},
        {"name": "relationship", "type": "category"},
        {"name": "race", "type": "category"},
        {"name": "sex", "type": "category"},
        {"name": "capital-gain", "type": "number"},
        {"name": "capital-loss", "type": "number"},
        {"name": "hours-per-week", "type": "number"},
        {"name": "native-country", "type": "category"},
    ],
    "output_features": [
        {
            "name": "income",
            "num_fc_layers": 4,
            "output_size": 32,
            "preprocessing": {"fallback_true_label": " >50K"},
            "loss": {"type": "binary_weighted_cross_entropy"},
            "type": "binary",
        }
    ],
    "preprocessing": {"number": {"missing_value_strategy": "fill_with_mean", "normalization": "zscore"}},
    "trainer": {"epochs": 3, "optimizer": {"type": "adam"}},
}

RAY_BACKEND_CONFIG = {
    "type": "ray",
    "processor": {
        "parallelism": 2,
    },
    "trainer": {
        "use_gpu": False,
        "num_workers": 2,
        "resources_per_worker": {
            "CPU": 0.1,
            "GPU": 0,
        },
    },
}

MODEL_PATH = "~/progress_bar_model.ludwig"
FILE_PATH = "~/progress_bar_model/model_hyperparameters.json"

ray.init(address="auto")
# model = LudwigModel(config=config, backend="local", logging_level=logging.INFO)
model = LudwigModel(config=config, backend=RAY_BACKEND_CONFIG, logging_level=logging.INFO)
train_stats, preprocessed_data, output_directory = model.train(training_set=train_df, test_set=test_df)
# Extract subset of test data for evaluation due to limitations in amount of data displayable in colab notebook.
np.random.seed(13)
eval_df = test_df.sample(n=1000)
# Generates predictions and performance statistics for the test set.
test_stats, predictions, output_directory = model.evaluate(
    eval_df,
    collect_predictions=True,
    skip_save_eval_stats=False,
    skip_save_predictions=False,
    calculate_overall_stats=False,
)
ray.shutdown()
