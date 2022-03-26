import logging
import os
import random
import tempfile

import numpy as np
import pytest
import torch

from ludwig.api import LudwigModel
from ludwig.data.dataset_synthesizer import cli_synthesize_dataset

RANDOM_SEED = 1919

INPUT_FEATURES = [
    {"name": "num_1", "type": "number"},
    {"name": "num_2", "type": "number"},
]

OUTPUT_FEATURES = [{"name": "y", "type": "number"}]

CONFIG = {
    "input_features": INPUT_FEATURES,
    "output_features": OUTPUT_FEATURES,
    "trainer": {"epochs": 2, "batch_size": 8},
}


@pytest.fixture(scope="module")
def raw_dataset_fp():
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_fp = os.path.join(tmpdir, "raw_data.csv")
        random.seed(RANDOM_SEED)
        cli_synthesize_dataset(20, INPUT_FEATURES + OUTPUT_FEATURES, raw_fp)
        yield raw_fp


def test_preprocess(raw_dataset_fp):
    # define Ludwig model
    model1 = LudwigModel(config=CONFIG)

    # preprocess the raw data set, specify seed
    preprocessed_data1 = model1.preprocess(raw_dataset_fp, random_seed=RANDOM_SEED)

    # invoke torch random functions
    torch.manual_seed(RANDOM_SEED + 1)
    torch.rand((5,))

    # define Ludwig model
    model2 = LudwigModel(config=CONFIG)
    # preprocess same raw data set with same seed
    preprocessed_data2 = model2.preprocess(raw_dataset_fp, random_seed=RANDOM_SEED)

    # confirm data splits are reproducible
    for i in range(3):
        for k in preprocessed_data1[i].dataset:
            assert np.all(preprocessed_data1[i].dataset[k] == preprocessed_data2[i].dataset[k])


def test_train(raw_dataset_fp):
    # define Ludwig model
    model1 = LudwigModel(config=CONFIG, logging_level=logging.WARN)
    training_statistics1, preprocessed_data1, _ = model1.train(
        dataset=raw_dataset_fp, random_seed=RANDOM_SEED, skip_save_progress=True, skip_save_processed_input=True
    )

    # invoke torch random functions
    torch.manual_seed(RANDOM_SEED + 1)
    torch.rand((5,))

    model2 = LudwigModel(config=CONFIG, logging_level=logging.WARN)
    training_statistics2, preprocessed_data2, _ = model2.train(
        dataset=raw_dataset_fp, random_seed=RANDOM_SEED, skip_save_progress=True, skip_save_processed_input=True
    )

    # confirm data splits are reproducible
    for i in range(3):
        for k in preprocessed_data1[i].dataset:
            assert np.all(preprocessed_data1[i].dataset[k] == preprocessed_data2[i].dataset[k])

    # check for equality of training statistics
    assert training_statistics1 == training_statistics2


def test_experiment(raw_dataset_fp):
    # define Ludwig model
    model1 = LudwigModel(config=CONFIG, logging_level=logging.WARN)

    evaluation_statistics1, training_statistics1, preprocessed_data1, _ = model1.experiment(
        dataset=raw_dataset_fp, random_seed=RANDOM_SEED, skip_save_processed_input=True
    )

    # invoke torch random functions
    torch.manual_seed(RANDOM_SEED + 1)
    torch.rand((5,))

    model2 = LudwigModel(config=CONFIG, logging_level=logging.WARN)
    evaluation_statistics2, training_statistics2, preprocessed_data2, _ = model2.experiment(
        dataset=raw_dataset_fp, random_seed=RANDOM_SEED, skip_save_processed_input=True
    )

    # confirm data splits are reproducible
    for i in range(3):
        for k in preprocessed_data1[i].dataset:
            assert np.all(preprocessed_data1[i].dataset[k] == preprocessed_data2[i].dataset[k])

    # check for equality of training statistics
    assert training_statistics1 == training_statistics2
    assert evaluation_statistics1 == evaluation_statistics2
