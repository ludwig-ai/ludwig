#! /usr/bin/env python
#
# Trains a ludwig model for every dataset which has a builtin model_config.

# You must have valid kaggle credentials in your environment, a few GB of disk space, and good internet bandwidth.
# Also, for each dataset associated with a Kaggle competition you'll need to sign in to Kaggle and accept the terms of
# the competition.
#
import pandas as pd

from ludwig import datasets


def train_all_datasets():
    dataset_names = []
    has_config = []
    model_metrics = []
    model_performance = []

    # Download All Datasets
    for dataset_name in datasets.list_datasets():
        dataset = datasets.get_dataset(dataset_name)
        config = dataset.default_model_config
        dataset_names.append(dataset_name)
        if config:
            has_config.append(True)
            # Train model on config
        else:
            has_config.append(False)
            model_performance.append(None)
    results = pd.DataFrame(
        {
            "dataset": dataset_names,
            "has_config": has_config,
            "metric": model_metrics,
            "performance": model_performance,
        }
    )
    print(results)


if __name__ == "__main__":
    train_all_datasets()
