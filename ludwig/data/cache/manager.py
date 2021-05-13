import logging
import os
import re
import uuid
from pathlib import Path

from ludwig.constants import CHECKSUM, TRAINING, TEST, VALIDATION
from ludwig.data.cache.util import calculate_checksum
from ludwig.utils import data_utils
from ludwig.utils.fs_utils import path_exists, delete

logger = logging.getLogger(__name__)


def alphanum(v):
    """Filters a string to only its alphanumeric characters."""
    return re.sub(r'\W+', '', v)


class CacheManager(object):
    def __init__(self, dataset_manager, cache_dir=None):
        self._dataset_manager = dataset_manager
        self._cache_dir = cache_dir

    def put_dataset(self, dataset, training_set, validation_set, test_set,
                    config, processed, skip_save_processed_input):
        input_fname = dataset or training_set
        if not self.can_cache(input_fname, config, skip_save_processed_input):
            return processed

        training_df, test_df, validation_df, training_set_metadata = processed
        key = training_set_metadata.get(CHECKSUM)
        if not key:
            key = self.get_cache_key(input_fname, config)
            training_set_metadata[CHECKSUM] = key

        logger.info('Writing preprocessed training set cache')
        training_df = self.save(
            self.get_cache_path(input_fname, key, TRAINING),
            training_df,
            config,
            training_set_metadata,
            TRAINING,
        )

        if test_df is not None:
            logger.info('Writing preprocessed test set cache')
            test_fname = test_set or dataset
            test_df = self.save(
                self.get_cache_path(test_fname, key, TEST),
                test_df,
                config,
                training_set_metadata,
                TEST,
            )

        if validation_df is not None:
            logger.info('Writing preprocessed validation set cache')
            vali_fname = validation_set or dataset
            validation_df = self.save(
                self.get_cache_path(vali_fname, key, VALIDATION),
                validation_df,
                config,
                training_set_metadata,
                VALIDATION,
            )

        logger.info('Writing train set metadata')
        data_utils.save_json(
            self.get_cache_path(input_fname, key, 'meta', 'json'),
            training_set_metadata
        )

        return training_df, test_df, validation_df, training_set_metadata

    def get_dataset_path(self, dataset, training_set, validation_set, test_set,
                         config):
        input_fname = dataset or training_set
        key = self.get_cache_key(input_fname, config)
        training_set_metadata_fp = self.get_cache_path(
            input_fname, key, 'meta', 'json'
        )

        if path_exists(training_set_metadata_fp):
            cache_training_set_metadata = data_utils.load_json(
                training_set_metadata_fp
            )

            dataset_fp = self.get_cache_path(input_fname, key, TRAINING)
            valid = (key == cache_training_set_metadata.get(CHECKSUM)
                     and path_exists(dataset_fp))

            val_fp = None
            if dataset:
                val_fp = self.get_cache_path(input_fname, key, VALIDATION)
            elif validation_set:
                val_fp = self.get_cache_path(validation_set, key, VALIDATION)
            if val_fp and not path_exists(val_fp):
                val_fp = None
                valid = False

            test_fp = None
            if dataset:
                test_fp = self.get_cache_path(input_fname, key, TEST)
            elif test_set:
                test_fp = self.get_cache_path(test_set, key, TEST)
            if test_fp and not path_exists(test_fp):
                test_fp = None
                valid = False

            return valid, cache_training_set_metadata, dataset_fp, test_fp, val_fp

        return None

    def delete_dataset(self, dataset, training_set, validation_set, test_set,
                       config):
        input_fname = dataset or training_set
        key = self.get_cache_key(input_fname, config)
        fnames = [
            self.get_cache_path(input_fname, key, 'meta', 'json'),
            self.get_cache_path(input_fname, key, TRAINING),
            self.get_cache_path(input_fname, key, TEST),
            self.get_cache_path(input_fname, key, VALIDATION),
            self.get_cache_path(test_set, key, TEST),
            self.get_cache_path(validation_set, key, VALIDATION),
        ]

        for fname in fnames:
            if path_exists(fname):
                delete(fname)

    def get_cache_key(self, input_fname, config):
        if input_fname is None:
            # TODO(travis): could try hashing the in-memory dataset, but this is tricky for Dask
            return str(uuid.uuid1())
        return calculate_checksum(input_fname, config)

    def get_cache_path(self, input_fname, key, tag, ext=None):
        stem = alphanum(key) \
            if self._cache_dir is not None or input_fname is None \
            else Path(input_fname).stem
        ext = ext or self.data_format
        cache_fname = f'{stem}.{tag}.{ext}'
        return os.path.join(self.get_cache_directory(input_fname), cache_fname)

    def get_cache_directory(self, input_fname):
        if self._cache_dir is None:
            if input_fname is not None:
                return os.path.dirname(input_fname)
            return '.'
        return self._cache_dir

    def save(self, cache_path, dataset, config, training_set_metadata, tag):
        return self._dataset_manager.save(cache_path, dataset, config, training_set_metadata, tag)

    def can_cache(self, input_fname, config, skip_save_processed_input):
        return self._dataset_manager.can_cache(input_fname, config, skip_save_processed_input)

    @property
    def data_format(self):
        return self._dataset_manager.data_format
