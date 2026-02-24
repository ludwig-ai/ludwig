#! /usr/bin/env python
#
# Trains a ludwig model for every dataset which has a default_model_config.
# You must have valid kaggle credentials in your environment, a few GB of disk space, and good internet bandwidth.
# Also, for each dataset associated with a Kaggle competition you'll need to sign in to Kaggle and accept the terms of
# the competition.
#
import multiprocessing
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ludwig import datasets, visualize
from ludwig.api import LudwigModel
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.misc_utils import get_commit_hash


@dataclass
class TrainingResults:
    """Results of a training run for a dataset."""

    ludwig_version: str
    ludwig_commit: str | None
    dataset_version: str
    dataset_name: str
    has_config: bool
    output_directory: str | None = None
    splits: str | None = None
    metric: str | None = None
    performance: float | None = None
    load_time: float | None = None
    train_time: float | None = None
    eval_time: float | None = None


def _train_dataset_process(dataset_name, results_queue):
    """Runs each train job in a new process."""
    load_start_time = time.time()
    dataset = datasets.get_dataset(dataset_name)
    config = dataset.default_model_config
    df = dataset.load()
    load_end_time = time.time()
    if "split" not in df:
        df["split"] = 0
    available_splits = sorted(df.split.unique())
    results = TrainingResults(
        LUDWIG_VERSION,
        get_commit_hash(),
        dataset.version,
        dataset.name,
        config is not None,
        splits=" ".join([str(s) for s in available_splits]),
        load_time=load_end_time - load_start_time,
    )
    if config:
        dataset.export(".")
        print(f"Training {dataset_name}")

        # Train model on config
        train_start_time = time.time()
        model = LudwigModel(config)
        train_stats, _, output_directory = model.train(dataset=df, model_name=dataset_name)

        # If dataset has a test split with labels, evaluate on test set.  If not, evaluate on training set.
        evaluate_start_time = time.time()
        eval_stats, _, _ = model.evaluate(
            df,
            split=2 if 2 in available_splits else 0,
            collect_predictions=False,
            collect_overall_stats=True,
        )
        evaluate_end_time = time.time()

        # Visualize learning curve
        visualize.learning_curves([train_stats], model_names=[dataset_name], output_directory=output_directory)

        results.output_directory = output_directory

        # Get metric for first output feature
        first_of_name = config["output_features"][0]["name"]
        stats = eval_stats[first_of_name]
        if "accuracy" in stats:
            results.metric = "accuracy"
            results.performance = stats["accuracy"]
        elif "root_mean_squared_error" in stats:
            results.metric = "root_mean_squared_error"
            results.performance = stats["root_mean_squared_error"]
        elif "mean_squared_error" in stats:
            results.metric = "mean_squared_error"
            results.performance = stats["mean_squared_error"]
        elif "mean_absolute_error" in stats:
            results.metric = "mean_absolute_error"
            results.performance = stats["mean_absolute_error"]
        elif "loss" in stats:
            results.metric = "loss"
            results.performance = stats["loss"]
        results.train_time = evaluate_start_time - train_start_time
        results.eval_time = evaluate_end_time - evaluate_start_time
        print(f"Trained {dataset_name} in {evaluate_end_time - load_start_time:.2f} seconds")
    results_queue.put(results)


def train_all_datasets():
    # Maps dataset name to current running process.
    max_processes = 4
    running_processes = {}
    accumulated_results = []
    # As each process completes it pushes its results onto the results_queue.
    results_queue = multiprocessing.Queue()
    for dataset_name in datasets.list_datasets():
        if len(running_processes) >= max_processes:
            # Block until a subprocess completes
            next_results = results_queue.get()
            accumulated_results.append(next_results)
            process = running_processes[next_results.dataset_name]
            process.join()
            del running_processes[next_results.dataset_name]
        process = multiprocessing.Process(target=_train_dataset_process, args=[dataset_name, results_queue])
        running_processes[dataset_name] = process
        process.start()
    while len(running_processes) > 0:
        if len(running_processes) < 4:
            remaining_datasets = ", ".join(sorted(running_processes.keys()))
            print(f"Finishing up, waiting for {len(running_processes)} to complete ({remaining_datasets})")
        else:
            print(f"Finishing up, waiting for {len(running_processes)} to complete")
        # Block until a subprocess completes, clear it out,
        next_results = results_queue.get()
        accumulated_results.append(next_results)
        process = running_processes[next_results.dataset_name]
        process.join()
        del running_processes[next_results.dataset_name]
    results_df = pd.DataFrame(accumulated_results)
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.precision", 3, "display.width", 120
    ):
        results_to_display = results_df[results_df["has_config"]].copy()
        results_to_display = results_to_display.drop(
            columns=["dataset_version", "output_directory", "ludwig_version", "ludwig_commit", "has_config"]
        )
        print(results_to_display)
    results_df.to_csv("train_all_model_configs_results.csv", index=False)


if __name__ == "__main__":
    train_all_datasets()
