"""
automl.py

Driver script which:

(0) Pre-processing
(1) Builds a base config by performing type inference and populating config
    w/default combiner parameters, training paramers, and hyperopt search space
(2) Tunes config based on resource constraints
(3) Runs hyperparameter optimization experiment
"""
from typing import Dict, Union

import pandas as pd
from ludwig.hyperopt.run import hyperopt

from automl.tune_config import tune_batch_size, tune_learning_rate
from base_config import create_default_config

OUTPUT_DIR = "."


def auto_train(dataset: str, target: str, time_limit_s: Union[int, float]):
    # (1) get a config for each model type concat, tabnet, transformer
    # (2) tunes batch size and learning rate
    # (3) call train
    default_configs = create_default_config(dataset, target, time_limit_s)
    for model_name, model_config in default_configs.items():
        tune_batch_size(model_config)
        tune_learning_rate(model_config)
        train(model_config, dataset, OUTPUT_DIR, model_name=model_name)
    # TODO (ASN) : add logic for choosing which models to run (i.e. just concat on small datasets)


def train(
    config: Dict, dataset: Union[str, Dict, pd.Dataframe], output_dir: str, model_name: str
):
    hyperopt_results = hyperopt(
        config,
        dataset=dataset,
        output_directory=output_dir,
        model_name=model_name
        # gpus=gpu_list,
    )
    return hyperopt_results
