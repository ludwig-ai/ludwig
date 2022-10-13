import os
from pathlib import Path

import pandas as pd
import pytest

from ludwig.constants import CHECKSUM, META, TEST, TRAINING, VALIDATION
from ludwig.data.cache.manager import alphanum, CacheManager
from ludwig.data.cache.types import CacheableDataframe, wrap
from ludwig.data.dataset.pandas import PandasDatasetManager
from ludwig.globals import TRAINING_PREPROC_FILE_NAME
from tests.integration_tests.utils import category_feature, LocalTestBackend, sequence_feature


@pytest.fixture
def change_test_dir(tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)


@pytest.mark.parametrize("use_df", [True, False], ids=["df", "filename"])
@pytest.mark.parametrize("use_split", [True, False], ids=["split", "no_split"])
@pytest.mark.parametrize("use_cache_dir", [True, False], ids=["cache_dir", "no_cache_dir"])
def test_cache_dataset(use_cache_dir, use_split, use_df, tmpdir, change_test_dir):
    dataset_manager = PandasDatasetManager(backend=LocalTestBackend())
    cache_dir = os.path.join(tmpdir, "cache") if use_cache_dir else None
    manager = CacheManager(dataset_manager, cache_dir=cache_dir)

    config = {
        "input_features": [sequence_feature(encoder={"reduce_output": "sum"})],
        "output_features": [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")],
        "combiner": {"type": "concat", "output_size": 14},
        "preprocessing": {},
    }

    def touch(basename):
        path = os.path.join(tmpdir, f"{basename}.csv")
        Path(path).touch()
        return path

    def create_dataset(name):
        if use_df:
            return CacheableDataframe(df=pd.DataFrame(), name=name, checksum=name)
        else:
            return wrap(touch(name))

    dataset = training_set = test_set = validation_set = None
    if not use_split:
        dataset = create_dataset("dataset")
        cache_key = manager.get_cache_key(dataset, config)
    else:
        training_set = create_dataset("train")
        test_set = create_dataset("test")
        validation_set = create_dataset("validation")
        cache_key = manager.get_cache_key(training_set, config)

    training_set_metadata = {
        CHECKSUM: cache_key,
    }

    cache = manager.get_dataset_cache(config, dataset, training_set, test_set, validation_set)
    cache_map = cache.cache_map
    assert len(cache_map) == 4

    train_path = os.path.join(cache_dir, alphanum(cache_key)) if use_cache_dir else os.path.join(tmpdir, "dataset")
    test_path = val_path = train_path

    if use_split and not use_cache_dir:
        train_path = os.path.join(tmpdir, "train")
        test_path = os.path.join(tmpdir, "test")
        val_path = os.path.join(tmpdir, "validation")

    assert cache_map[META] == f"{train_path}.meta.json"
    assert cache_map[TRAINING] == f"{train_path}.{TRAINING_PREPROC_FILE_NAME}"
    assert cache_map[TEST] == f"{test_path}.test.hdf5"
    assert cache_map[VALIDATION] == f"{val_path}.validation.hdf5"

    for cache_path in cache_map.values():
        assert not os.path.exists(cache_path)

    training_set = pd.DataFrame()
    test_set = pd.DataFrame()
    validation_set = pd.DataFrame()

    if use_cache_dir:
        os.makedirs(cache_dir)
    cache.put(training_set, test_set, validation_set, training_set_metadata)

    for cache_path in cache_map.values():
        assert os.path.exists(cache_path)

    cache.delete()

    for cache_path in cache_map.values():
        assert not os.path.exists(cache_path)
