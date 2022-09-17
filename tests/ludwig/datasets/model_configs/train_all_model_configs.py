#! /usr/bin/env python
#
# Trains a ludwig model for every dataset which has a builtin model_config.

# You must have valid kaggle credentials in your environment, a few GB of disk space, and good internet bandwidth.
# Also, for each dataset associated with a Kaggle competition you'll need to sign in to Kaggle and accept the terms of
# the competition.
#
import pandas as pd

from ludwig import datasets
from ludwig.api import LudwigModel


def train_all_datasets():
    dataset_names = []
    has_config = []
    output_directories = []
    model_metrics = []
    model_performance = []

    # Download All Datasets
    for dataset_name in datasets.list_datasets():
        dataset = datasets.get_dataset(dataset_name)
        config = dataset.default_model_config
        if config:
            # Train model on config
            df = dataset.load()
            model = LudwigModel(config)
            train_stats, _, output_directory = model.train(dataset=df, model_name=dataset_name)

            # Get metric for first output feature (assuming

            dataset_names.append(dataset_name)
            has_config.append(True)
            output_directories.append(output_directory)
            model_metrics.append(None)
            model_performance.append(None)
        else:
            dataset_names.append(dataset_name)
            has_config.append(False)
            model_metrics.append(None)
            model_performance.append(None)
    results = pd.DataFrame(
        {
            "dataset": dataset_names,
            "has_config": has_config,
            "output_directory": output_directories,
            "metric": model_metrics,
            "performance": model_performance,
        }
    )
    print(results)


if __name__ == "__main__":
    train_all_datasets()
