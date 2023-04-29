import logging
import os
from typing import Optional

from ludwig.constants import CHECKSUM, META, TEST, TRAINING, VALIDATION
from ludwig.data.cache.types import alphanum, CacheableDataset
from ludwig.data.cache.util import calculate_checksum
from ludwig.data.dataset.base import DatasetManager
from ludwig.utils import data_utils
from ludwig.utils.fs_utils import delete, path_exists

logger = logging.getLogger(__name__)


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

        try:
            cached_training_set_metadata = data_utils.load_json(training_set_metadata_fp)
        except Exception:
            logger.exception(f"Failed to load cached training set metadata at {training_set_metadata_fp}")
            return None

        cached_training_set = self.cache_map[TRAINING] if path_exists(self.cache_map[TRAINING]) else None
        if not cached_training_set:
            logger.warning(f"Failed to load cached training set at {self.cache_map[TRAINING]}")

        cached_validation_set = self.cache_map[VALIDATION] if path_exists(self.cache_map[VALIDATION]) else None
        if not cached_validation_set:
            logger.warning(f"Failed to load cached validation set at {self.cache_map[VALIDATION]}")

        cached_test_set = self.cache_map[TEST] if path_exists(self.cache_map[TEST]) else None
        if not cached_test_set:
            logger.warning(f"Failed to load cached test set at {self.cache_map[TEST]}")

        valid = self.checksum == cached_training_set_metadata.get(CHECKSUM) and cached_training_set is not None

        return valid, cached_training_set_metadata, cached_training_set, cached_test_set, cached_validation_set

    def put(self, training_set, test_set, validation_set, training_set_metadata):
        logger.info(f"Writing preprocessed training set cache to {self.cache_map[TRAINING]}")
        training_set = self.dataset_manager.save(
            self.cache_map[TRAINING],
            training_set,
            self.config,
            training_set_metadata,
            TRAINING,
        )

        if validation_set is not None:
            logger.info(f"Writing preprocessed validation set cache to {self.cache_map[VALIDATION]}")
            validation_set = self.dataset_manager.save(
                self.cache_map[VALIDATION],
                validation_set,
                self.config,
                training_set_metadata,
                VALIDATION,
            )

        if test_set is not None:
            logger.info(f"Writing preprocessed test set cache to {self.cache_map[TEST]}")
            test_set = self.dataset_manager.save(
                self.cache_map[TEST],
                test_set,
                self.config,
                training_set_metadata,
                TEST,
            )

        logger.info(f"Writing train set metadata to {self.cache_map[META]}")
        data_utils.save_json(self.cache_map[META], training_set_metadata)

        return training_set, test_set, validation_set, training_set_metadata

    def delete(self):
        for fname in self.cache_map.values():
            if path_exists(fname):
                # Parquet entries in the cache_ma can be pointers to directories.
                delete(fname, recursive=True)

    def get_cached_obj_path(self, cached_obj_name: str) -> str:
        return self.cache_map.get(cached_obj_name)


class CacheManager:
    def __init__(
        self,
        dataset_manager: DatasetManager,
        cache_dir: Optional[str] = None,
    ):
        self._dataset_manager = dataset_manager
        self._cache_dir = cache_dir

    def get_dataset_cache(
        self,
        config: dict,
        dataset: Optional[CacheableDataset] = None,
        training_set: Optional[CacheableDataset] = None,
        test_set: Optional[CacheableDataset] = None,
        validation_set: Optional[CacheableDataset] = None,
    ) -> DatasetCache:
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

    def get_cache_key(self, dataset: CacheableDataset, config: dict) -> str:
        return calculate_checksum(dataset, config)

    def get_cache_path(self, dataset: Optional[CacheableDataset], key: str, tag: str, ext: Optional[str] = None) -> str:
        if self._cache_dir is None and dataset is not None:
            # Use the input dataset filename (minus the extension) as the cache path
            stem = dataset.get_cache_path()
        else:
            # To avoid collisions across different directories, we use the unique checksum
            # as the cache path
            stem = alphanum(key)

        ext = ext or self.data_format
        cache_fname = f"{stem}.{tag}.{ext}"
        return os.path.join(self.get_cache_directory(dataset), cache_fname)

    def get_cache_directory(self, dataset: Optional[CacheableDataset]) -> str:
        if self._cache_dir is None:
            if dataset is None:
                return os.getcwd()
            return dataset.get_cache_directory()
        return self._cache_dir

    def can_cache(self, skip_save_processed_input: bool) -> bool:
        return self._dataset_manager.can_cache(skip_save_processed_input)

    @property
    def data_format(self) -> str:
        return self._dataset_manager.data_format
