import logging
import os
import re
import uuid
from pathlib import Path

from ludwig.constants import CHECKSUM, META, TEST, TRAINING, VALIDATION
from ludwig.data.cache.util import calculate_checksum
from ludwig.utils import data_utils
from ludwig.utils.fs_utils import delete, path_exists

logger = logging.getLogger(__name__)


def alphanum(v):
    """Filters a string to only its alphanumeric characters."""
    return re.sub(r"\W+", "", v)


class DatasetCache:
    def __init__(self, config, checksum, cache_map, dataset_manager):
        self.config = config
        self.checksum = checksum
        self.cache_map = cache_map
        self.dataset_manager = dataset_manager

    def get(self):
        training_set_metadata_fp = self.cache_map[META]
        if not path_exists(training_set_metadata_fp):
            return None

        cache_training_set_metadata = data_utils.load_json(training_set_metadata_fp)

        cached_training_set = self.cache_map[TRAINING] if path_exists(self.cache_map[TRAINING]) else None

        cached_test_set = self.cache_map[TEST] if path_exists(self.cache_map[TEST]) else None

        cached_validation_set = self.cache_map[VALIDATION] if path_exists(self.cache_map[VALIDATION]) else None

        valid = self.checksum == cache_training_set_metadata.get(CHECKSUM) and cached_training_set is not None

        return valid, cache_training_set_metadata, cached_training_set, cached_test_set, cached_validation_set

    def put(self, training_set, test_set, validation_set, training_set_metadata):
        logger.info("Writing preprocessed training set cache")
        training_set = self.dataset_manager.save(
            self.cache_map[TRAINING],
            training_set,
            self.config,
            training_set_metadata,
            TRAINING,
        )

        if test_set is not None:
            logger.info("Writing preprocessed test set cache")
            test_set = self.dataset_manager.save(
                self.cache_map[TEST],
                test_set,
                self.config,
                training_set_metadata,
                TEST,
            )

        if validation_set is not None:
            logger.info("Writing preprocessed validation set cache")
            validation_set = self.dataset_manager.save(
                self.cache_map[VALIDATION],
                validation_set,
                self.config,
                training_set_metadata,
                VALIDATION,
            )

        logger.info("Writing train set metadata")
        data_utils.save_json(self.cache_map[META], training_set_metadata)

        return training_set, test_set, validation_set, training_set_metadata

    def delete(self):
        for fname in self.cache_map.values():
            if path_exists(fname):
                delete(fname)


class CacheManager:
    def __init__(self, dataset_manager, cache_dir=None):
        self._dataset_manager = dataset_manager
        self._cache_dir = cache_dir

    def get_dataset_cache(self, config, dataset=None, training_set=None, test_set=None, validation_set=None):
        if dataset is not None:
            key = self.get_cache_key(dataset, config)
            cache_map = {
                META: self.get_cache_path(dataset, key, META, "json"),
                TRAINING: self.get_cache_path(dataset, key, TRAINING),
                TEST: self.get_cache_path(dataset, key, TEST),
                VALIDATION: self.get_cache_path(dataset, key, VALIDATION),
            }
            return DatasetCache(config, key, cache_map, self._dataset_manager)
        else:
            key = self.get_cache_key(training_set, config)
            cache_map = {
                META: self.get_cache_path(training_set, key, META, "json"),
                TRAINING: self.get_cache_path(training_set, key, TRAINING),
                TEST: self.get_cache_path(test_set, key, TEST),
                VALIDATION: self.get_cache_path(validation_set, key, VALIDATION),
            }
            return DatasetCache(config, key, cache_map, self._dataset_manager)

    def get_cache_key(self, dataset, config):
        if not isinstance(dataset, str):
            # TODO(travis): could try hashing the in-memory dataset, but this is tricky for Dask
            return str(uuid.uuid1())
        return calculate_checksum(dataset, config)

    def get_cache_path(self, dataset, key, tag, ext=None):
        if not isinstance(dataset, str):
            dataset = None

        if self._cache_dir is None and dataset is not None:
            # Use the input dataset filename (minus the extension) as the cache path
            stem = Path(dataset).stem
        else:
            # To avoid collisions across different directories, we use the unique checksum
            # as the cache path
            stem = alphanum(key)

        ext = ext or self.data_format
        cache_fname = f"{stem}.{tag}.{ext}"
        return os.path.join(self.get_cache_directory(dataset), cache_fname)

    def get_cache_directory(self, input_fname):
        if self._cache_dir is None:
            if input_fname is not None:
                return os.path.dirname(input_fname)
            return "."
        return self._cache_dir

    def can_cache(self, skip_save_processed_input):
        return self._dataset_manager.can_cache(skip_save_processed_input)

    @property
    def data_format(self):
        return self._dataset_manager.data_format
