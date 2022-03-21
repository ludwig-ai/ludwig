import uuid

import ludwig
from ludwig.constants import NAME, PREPROCESSING, TYPE
from ludwig.data.cache.types import CacheableDataset
from ludwig.utils.fs_utils import checksum
from ludwig.utils.misc_utils import hash_dict


def get_dataset_checksum(original_dataset):
    if isinstance(original_dataset, str):
        return checksum(original_dataset)
    if isinstance(original_dataset, CacheableDataset):
        return original_dataset.checksum
    else:
        # TODO(travis): could try hashing the in-memory dataset, but this is tricky for Dask
        return str(uuid.uuid1())


def calculate_checksum(original_dataset, config):
    features = config.get("input_features", []) + config.get("output_features", []) + config.get("features", [])
    info = {
        "ludwig_version": ludwig.globals.LUDWIG_VERSION,
        "dataset_checksum": get_dataset_checksum(original_dataset),
        "global_preprocessing": config["preprocessing"],
        "feature_names": [feature[NAME] for feature in features],
        "feature_types": [feature[TYPE] for feature in features],
        "feature_preprocessing": [feature.get(PREPROCESSING, {}) for feature in features],
    }
    return hash_dict(info, max_length=None).decode("ascii")
