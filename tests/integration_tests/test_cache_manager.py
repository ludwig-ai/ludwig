import os
from pathlib import Path

import pandas as pd
import pytest

from ludwig.constants import META, TRAINING, VALIDATION, TEST, CHECKSUM
from ludwig.data.cache.manager import CacheManager, alphanum
from ludwig.data.dataset import PandasDatasetManager

from tests.integration_tests.utils import sequence_feature, category_feature, LocalTestBackend


@pytest.mark.parametrize('use_cache_dir', [True, False], ids=['cache_dir', 'no_cache_dir'])
def test_cache_dataset(use_cache_dir, tmpdir):
    dataset_manager = PandasDatasetManager(backend=LocalTestBackend())
    cache_dir = os.path.join(tmpdir, 'cache') if use_cache_dir else None
    manager = CacheManager(dataset_manager, cache_dir=cache_dir)

    config = {
        'input_features': [sequence_feature(reduce_output='sum')],
        'output_features': [category_feature(vocab_size=2, reduce_input='sum')],
        'combiner': {'type': 'concat', 'fc_size': 14},
        'preprocessing': {},
    }

    dataset = os.path.join(tmpdir, 'dataset.csv')
    Path(dataset).touch()

    cache_key = manager.get_cache_key(dataset, config)
    training_set_metadata = {
        CHECKSUM: cache_key,
    }

    cache = manager.get_dataset_cache(config, dataset)
    cache_map = cache.cache_map
    assert len(cache_map) == 4

    base_path = os.path.join(cache_dir, alphanum(cache_key)) if \
        use_cache_dir else \
        os.path.join(tmpdir, 'dataset')

    assert cache_map[META] == f'{base_path}.meta.json'
    assert cache_map[TRAINING] == f'{base_path}.training.hdf5'
    assert cache_map[TEST] == f'{base_path}.test.hdf5'
    assert cache_map[VALIDATION] == f'{base_path}.validation.hdf5'

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
