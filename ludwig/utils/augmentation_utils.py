from typing import Dict, List, Union

from ludwig.api_annotations import DeveloperAPI
from ludwig.utils.registry import Registry

###
# Registry for augmentation operations
# Each augmentation operation is registered with the feature type it is applicable to
# and the name of the operation.
###
_augmentation_op_registry = Registry()


@DeveloperAPI
def get_augmentation_op_registry() -> Registry:
    return _augmentation_op_registry


@DeveloperAPI
def register_augmentation_op(name: str, features: Union[str, List[str]]):
    if isinstance(features, str):
        features = [features]

    def wrap(cls):
        for feature in features:
            augmentation_op_registry = get_augmentation_op_registry().get(feature, {})
            augmentation_op_registry[name] = cls
            get_augmentation_op_registry()[feature] = augmentation_op_registry
        return cls

    return wrap


@DeveloperAPI
def get_augmentation_op(feature_type: str, op_name: str):
    return get_augmentation_op_registry()[feature_type][op_name]


class AugmentationPipelines:
    """Container holding augmentation pipelines defined in the model."""

    def __init__(self, augmentation_pipelines: Dict):
        self.augmentation_pipelines = augmentation_pipelines

    def __getitem__(self, key):
        return self.augmentation_pipelines[key]

    def __contains__(self, key):
        return key in self.augmentation_pipelines

    def __len__(self):
        return len(self.augmentation_pipelines)

    def __iter__(self):
        return self.augmentation_pipelines.__iter__()

    def items(self):
        return self.augmentation_pipelines.items()
