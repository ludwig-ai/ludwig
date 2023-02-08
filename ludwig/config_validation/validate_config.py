from ludwig.config_validation.checks import get_config_check_registry
from ludwig.schema.model_types.base import ModelConfig


def validate_config(config):
    from ludwig.utils.backward_compatibility import upgrade_config_dict_to_latest_version

    # Update config from previous versions to check that backwards compatibility will enable a valid config
    updated_config = upgrade_config_dict_to_latest_version(config)

    print(updated_config)

    model_config = ModelConfig.from_dict(updated_config)

    for config_check_cls in get_config_check_registry().values():
        config_check_cls.check(model_config.to_dict())

    # print(model_config.to_dict())
