from typing import Any, Dict
import ludwig
from ludwig.constants import DEFAULTS, ENCODER, INPUT_FEATURES, NAME, OUTPUT_FEATURES, PREPROCESSING, TYPE
from ludwig.data.cache.types import CacheableDataset
from ludwig.types import ModelConfigDict
from ludwig.utils.config_utils import merge_fixed_preprocessing_params
from ludwig.utils.data_utils import hash_dict


def calculate_checksum(original_dataset: CacheableDataset, config: ModelConfigDict):
    features = config.get(INPUT_FEATURES, []) + config.get(OUTPUT_FEATURES, []) + config.get("features", [])
    info = {
        "ludwig_version": ludwig.globals.LUDWIG_VERSION,
        "dataset_checksum": original_dataset.checksum,
        "global_preprocessing": config.get(PREPROCESSING, {}),
        "global_defaults": config.get(DEFAULTS, {}),
        "feature_names": [feature[NAME] for feature in features],
        "feature_types": [feature[TYPE] for feature in features],
        "feature_preprocessing": [
            _merge_encoder_cache_params(
                merge_fixed_preprocessing_params(
                    feature[TYPE], feature.get(PREPROCESSING, {}), feature.get(ENCODER, {})
                ),
                feature.get(ENCODER, {}),
            )
            for feature in features
        ],
    }
    return hash_dict(info, max_length=None).decode("ascii")


def _merge_encoder_cache_params(preprocessing_params: Dict[str, Any], encoder_params: Dict[str, Any]) -> Dict[str, Any]:
    if preprocessing_params.get("cache_encoder_embeddings"):
        preprocessing_params[ENCODER] = encoder_params
    return preprocessing_params
