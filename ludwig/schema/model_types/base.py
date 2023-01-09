from abc import ABC
import copy
from typing import Any, Dict, Optional
from marshmallow import ValidationError

from marshmallow_dataclass import dataclass
import marshmallow_dataclass

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ENCODER, INPUT_FEATURES, MODEL_ECD, PREPROCESSING, TYPE
from ludwig.schema import utils as schema_utils
from ludwig.schema.defaults.defaults import DefaultsConfig
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig, FeatureCollection
from ludwig.schema.hyperopt import HyperoptConfig
from ludwig.schema.model_types.utils import (
    merge_fixed_preprocessing_params,
    merge_with_defaults,
    set_validation_parameters,
)
from ludwig.schema.preprocessing import PreprocessingConfig
from ludwig.schema.trainer import BaseTrainerConfig
from ludwig.utils.backward_compatibility import upgrade_config_dict_to_latest_version
from ludwig.utils.registry import Registry

model_type_schema_registry = Registry()


@DeveloperAPI
@dataclass(repr=False)
class ModelConfig(schema_utils.BaseMarshmallowConfig, ABC):
    input_features: FeatureCollection[BaseInputFeatureConfig]
    output_features: FeatureCollection[BaseOutputFeatureConfig]

    model_type: str

    trainer: BaseTrainerConfig
    preprocessing: PreprocessingConfig
    defaults: DefaultsConfig
    hyperopt: Optional[HyperoptConfig] = None

    @staticmethod
    def from_dict(config: Dict[str, Any]) -> "ModelConfig":
        config = copy.deepcopy(config)
        config = upgrade_config_dict_to_latest_version(config)
        config = merge_with_defaults(config)

        model_type = config.get("model_type", MODEL_ECD)

        # TODO(travis): move this into helper function
        # Update preprocessing parameters if encoders require fixed preprocessing parameters
        for feature_config in config.get(INPUT_FEATURES, []):
            if TYPE not in feature_config:
                continue

            preprocessing_parameters = feature_config.get(PREPROCESSING, {})
            preprocessing_parameters = merge_fixed_preprocessing_params(
                model_type, feature_config[TYPE], preprocessing_parameters, feature_config.get(ENCODER, {})
            )
            feature_config[PREPROCESSING] = preprocessing_parameters

        if model_type not in model_type_schema_registry:
            raise ValidationError(
                f"Invalid model type: '{model_type}', expected one of: {list(model_type_schema_registry.keys())}"
            )
        cls = model_type_schema_registry[model_type]
        schema = marshmallow_dataclass.class_schema(cls)()

        config_obj = schema.load(config)
        set_validation_parameters(config_obj)

        return config_obj


@DeveloperAPI
def register_model_type(name: str):
    def wrap(model_type_config: ModelConfig) -> ModelConfig:
        model_type_schema_registry[name] = model_type_config
        return model_type_config

    return wrap
