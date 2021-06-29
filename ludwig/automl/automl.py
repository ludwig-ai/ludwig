"""
automl.py

Driver script which:

(1) Builds a base config by performing type inference and populating config
    w/default combiner parameters, training paramers, and hyperopt search space
(2) Tunes config based on resource constraints
(3) Runs hyperparameter optimization experiment
"""
from typing import Dict, Union

import pandas as pd
from ludwig.hyperopt.run import hyperopt

from ludwig.automl.base_config import create_default_config

OUTPUT_DIR = "."


def model_select(default_configs):
    """
    Performs model selection based on dataset.
    Note: Current implementation returns tabnet by default. This will be 
        improved in subsequent iterations
    """
    return default_configs['tabnet'], 'tabnet'


def auto_train(dataset: str, target: str, time_limit_s: Union[int, float]):
    """
    Main auto train API that first builds configs for each model type 
    (e.g. concat, tabnet, transformer). Then selects model based on dataset 
    attributes. And finally runs a hyperparameter optimization experiment.

    All batch and learning rate tuning is done @ training time.

    # Inputs
    :param dataset: (str) filepath to dataset.
    :param target_name: (str) name of target feature
    :param time_limit_s: (int, float) total time allocated to auto_train. acts
                                    as the stopping parameter

    """

    default_configs = create_default_config(dataset, target, time_limit_s)
    model_config, model_name = model_select(default_configs)
    train(model_config, dataset, OUTPUT_DIR, model_name=model_name)


def train(
    config: Dict, dataset: Union[str, Dict, pd.DataFrame], output_dir: str, model_name: str
):
    hyperopt_results = hyperopt(
        config,
        dataset=dataset,
        output_directory=output_dir,
        model_name=model_name
        # gpus=gpu_list,
    )
    return hyperopt_results
