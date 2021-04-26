import ludwig
from ludwig.constants import NAME, TYPE, PREPROCESSING
from ludwig.utils.fs_utils import get_modified_timestamp
from ludwig.utils.misc_utils import hash_dict


def calculate_checksum(original_dataset, config):
    info = {}
    info['ludwig_version'] = ludwig.globals.LUDWIG_VERSION
    info['dataset_modification_date'] = get_modified_timestamp(original_dataset)
    info['global_preprocessing'] = config['preprocessing']
    features = config.get('input_features', []) + \
               config.get('output_features', []) + \
               config.get('features', [])
    info['feature_names'] = [feature[NAME] for feature in features]
    info['feature_types'] = [feature[TYPE] for feature in features]
    info['feature_preprocessing'] = [feature.get(PREPROCESSING, {})
                                     for feature in features]
    hash = hash_dict(info, max_length=None)
    return hash.decode('ascii')