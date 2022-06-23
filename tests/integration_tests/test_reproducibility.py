import logging
import os
import pathlib
import random

import numpy as np
import pytest
import torch

from ludwig.api import LudwigModel
from ludwig.data.dataset_synthesizer import cli_synthesize_dataset

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


@pytest.fixture(scope="function")
def raw_dataset_fp(tmpdir: pathlib.Path) -> str:
    """Generates dataset to be used in this test.

    Returns (str):  file path string for dataset to use in this tests
    """
    raw_fp = os.path.join(tmpdir, "raw_data.csv")
    random.seed(42)
    cli_synthesize_dataset(64, INPUT_FEATURES + OUTPUT_FEATURES, raw_fp)
    yield raw_fp


@pytest.mark.parametrize("second_seed_offset", [0, 1])
@pytest.mark.parametrize("random_seed", [1919, 31])
def test_preprocess(raw_dataset_fp: str, random_seed: int, second_seed_offset: int) -> None:
    """Test reproducibility of train/validation/test splits.

    Args:
        raw_dataset_fp (str): file path for data to be used as part of this test
        random_seed(int): random seed integer to use for test
        second_seed_offset(int): zero to use same random seed for second test, non-zero to use a different
            seed for the second run.

    Returns: None
    """
    # define Ludwig model
    model1 = LudwigModel(config=CONFIG)

    # preprocess the raw data set, specify seed
    preprocessed_data1 = model1.preprocess(raw_dataset_fp, random_seed=random_seed)

    # perform second preprocess operation
    model2 = LudwigModel(config=CONFIG)
    # preprocess same raw data set with same seed
    preprocessed_data2 = model2.preprocess(raw_dataset_fp, random_seed=random_seed + second_seed_offset)

    # confirm data splits are reproducible
    for i in range(3):
        for k in preprocessed_data1[i].dataset:
            if second_seed_offset == 0:
                # same seeds should result in same output
                assert np.all(preprocessed_data1[i].dataset[k] == preprocessed_data2[i].dataset[k])
            else:
                # non-zero second_seed_offset uses different seeds and should result in different output
                assert not np.all(preprocessed_data1[i].dataset[k] == preprocessed_data2[i].dataset[k])


@pytest.mark.parametrize("random_seed", [1919, 31])
def test_preprocess_ignore_torch_seed(raw_dataset_fp: str, random_seed: int) -> None:
    """Test reproducibility of train/validation/test splits when an unrelated torch random operation is performed
    between the Ludwig operations.

    Args:
        raw_dataset_fp (str): file path for data to be used as part of this test
        random_seed(int): random seed integer to use for test

    Returns: None
    """
    # define Ludwig model
    model1 = LudwigModel(config=CONFIG)

    # preprocess the raw data set, specify seed
    preprocessed_data1 = model1.preprocess(raw_dataset_fp, random_seed=random_seed)

    # invoke torch random functions with unrelated seed to
    # see if it affects Ludwig reproducibility
    torch.manual_seed(random_seed + 5)
    torch.rand((5,))

    # define Ludwig model
    model2 = LudwigModel(config=CONFIG)
    # preprocess same raw data set with same seed
    preprocessed_data2 = model2.preprocess(raw_dataset_fp, random_seed=random_seed)

    # confirm data splits are reproducible
    for i in range(3):
        for k in preprocessed_data1[i].dataset:
            # same seeds should result in same output
            assert np.all(preprocessed_data1[i].dataset[k] == preprocessed_data2[i].dataset[k])


@pytest.mark.parametrize("second_seed_offset", [0, 1])
@pytest.mark.parametrize("random_seed", [1919, 31])
def test_train(raw_dataset_fp: str, random_seed: int, second_seed_offset: int) -> None:
    """Test reproducibility of training API.

    Args:
        raw_dataset_fp (str): file path for data to be used as part of this test
        random_seed(int): random seed integer to use for test
        second_seed_offset(int): zero to use same random seed for second test, non-zero to use a different
            seed for the second run.

    Returns: None
    """
    # perform first model training run
    model1 = LudwigModel(config=CONFIG, logging_level=logging.WARN)
    training_statistics1, preprocessed_data1, _ = model1.train(
        dataset=raw_dataset_fp, random_seed=random_seed, skip_save_progress=True, skip_save_processed_input=True
    )

    # perform second model training run
    model2 = LudwigModel(config=CONFIG, logging_level=logging.WARN)
    training_statistics2, preprocessed_data2, _ = model2.train(
        dataset=raw_dataset_fp,
        random_seed=random_seed + second_seed_offset,
        skip_save_progress=True,
        skip_save_processed_input=True,
    )

    # confirm data splits are reproducible
    for i in range(3):
        for k in preprocessed_data1[i].dataset:
            if second_seed_offset == 0:
                # same seeds should result in same output
                assert np.all(preprocessed_data1[i].dataset[k] == preprocessed_data2[i].dataset[k])
            else:
                # non-zero second_seed_offset uses different seeds and should result in different output
                assert not np.all(preprocessed_data1[i].dataset[k] == preprocessed_data2[i].dataset[k])

    # confirm reproducibility/non-reproducibility of results
    if second_seed_offset == 0:
        # same seeds should result in same output
        assert training_statistics1 == training_statistics2
    else:
        # non-zero second_seed_offset uses different seeds and should result in different output
        assert training_statistics1 != training_statistics2


@pytest.mark.parametrize("random_seed", [1919, 31])
def test_train_ignore_torch_seed(raw_dataset_fp: str, random_seed: int) -> None:
    """Test reproducibility of training API when an unrelated torch random operation is performed between the
    Ludwig operations.

    Args:
        raw_dataset_fp (str): file path for data to be used as part of this test
        random_seed(int): random seed integer to use for test

    Returns: None
    """
    # define Ludwig model
    model1 = LudwigModel(config=CONFIG, logging_level=logging.WARN)
    training_statistics1, preprocessed_data1, _ = model1.train(
        dataset=raw_dataset_fp, random_seed=random_seed, skip_save_progress=True, skip_save_processed_input=True
    )

    # invoke torch random functions with unrelated seed to
    # see if it affects Ludwig reproducibility
    torch.manual_seed(random_seed + 5)
    torch.rand((5,))

    model2 = LudwigModel(config=CONFIG, logging_level=logging.WARN)
    training_statistics2, preprocessed_data2, _ = model2.train(
        dataset=raw_dataset_fp,
        random_seed=random_seed,
        skip_save_progress=True,
        skip_save_processed_input=True,
    )

    # confirm data splits are reproducible
    for i in range(3):
        for k in preprocessed_data1[i].dataset:
            # same seeds should result in same output
            assert np.all(preprocessed_data1[i].dataset[k] == preprocessed_data2[i].dataset[k])

    # confirm reproducibility/non-reproducibility of results
    assert training_statistics1 == training_statistics2


@pytest.mark.parametrize("second_seed_offset", [0, 1])
@pytest.mark.parametrize("random_seed", [1919, 31])
def test_experiment(raw_dataset_fp: str, random_seed: int, second_seed_offset: int) -> None:
    """Test reproducibility of experiment API.

    Args:
        raw_dataset_fp (str): file path for data to be used as part of this test
        random_seed(int): random seed integer to use for test
        second_seed_offset(int): zero to use same random seed for second test, non-zero to use a different
            seed for the second run.

    Returns: None
    """
    # perform first model experiment
    model1 = LudwigModel(config=CONFIG, logging_level=logging.WARN)
    evaluation_statistics1, training_statistics1, preprocessed_data1, _ = model1.experiment(
        dataset=raw_dataset_fp, random_seed=random_seed, skip_save_processed_input=True
    )

    # perform second model experiment
    model2 = LudwigModel(config=CONFIG, logging_level=logging.WARN)
    evaluation_statistics2, training_statistics2, preprocessed_data2, _ = model2.experiment(
        dataset=raw_dataset_fp, random_seed=random_seed + second_seed_offset, skip_save_processed_input=True
    )

    # confirm data splits are reproducible
    for i in range(3):
        for k in preprocessed_data1[i].dataset:
            if second_seed_offset == 0:
                # same seeds should result in same output
                assert np.all(preprocessed_data1[i].dataset[k] == preprocessed_data2[i].dataset[k])
            else:
                # non-zero second_seed_offset uses different seeds and should result in different output
                assert not np.all(preprocessed_data1[i].dataset[k] == preprocessed_data2[i].dataset[k])

    # confirm results reproducibility/non-reproducibility of results
    if second_seed_offset == 0:
        # same seeds should result in same output
        assert training_statistics1 == training_statistics2
        assert evaluation_statistics1 == evaluation_statistics2
    else:
        # non-zero second_seed_offset uses different seeds and should result in different output
        assert training_statistics1 != training_statistics2
        assert evaluation_statistics1 != evaluation_statistics2


@pytest.mark.parametrize("random_seed", [1919, 31])
def test_experiment_ignore_torch_seed(raw_dataset_fp: str, random_seed: int) -> None:
    """Test reproducibility of experiment API when an unrelated torch random operation is performed between the
    Ludwig operations.

    Args:
        raw_dataset_fp (str): file path for data to be used as part of this test
        random_seed(int): random seed integer to use for test

    Returns: None
    """
    # define Ludwig model
    model1 = LudwigModel(config=CONFIG, logging_level=logging.WARN)

    evaluation_statistics1, training_statistics1, preprocessed_data1, _ = model1.experiment(
        dataset=raw_dataset_fp, random_seed=random_seed, skip_save_processed_input=True
    )

    # invoke torch random functions with unrelated seed to
    # see if it affects Ludwig reproducibility
    torch.manual_seed(random_seed + 5)
    torch.rand((5,))

    model2 = LudwigModel(config=CONFIG, logging_level=logging.WARN)
    evaluation_statistics2, training_statistics2, preprocessed_data2, _ = model2.experiment(
        dataset=raw_dataset_fp, random_seed=random_seed, skip_save_processed_input=True
    )

    # confirm data splits are reproducible
    for i in range(3):
        for k in preprocessed_data1[i].dataset:
            # same seeds should result in same output
            assert np.all(preprocessed_data1[i].dataset[k] == preprocessed_data2[i].dataset[k])

    # confirm results reproducibility/non-reproducibility of results
    # same seeds should result in same output
    assert training_statistics1 == training_statistics2
    assert evaluation_statistics1 == evaluation_statistics2
