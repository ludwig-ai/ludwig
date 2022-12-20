from ludwig.types import ModelConfigDict
from ludwig.utils.config_utils import has_pretrained_encoder, has_trainable_encoder, has_unstructured_input_feature


def get_auto_learning_rate(config_dict: ModelConfigDict) -> float:
    """Uses config heuristics to determine an appropriate learning rate.

    The main idea behind the following heuristics is that smaller learning rates are more
    suitable for features with larger encoders, which are typically used with unstructured features.
    Note that these are meant to be rough heuristics that are solely based on feature types and the
    type of the corresponding encoder. More factors could be taken into consideration such as model
    size, dataset size, batch size, number of features, etc.

    param:
        config: Ludwig config used to train the model.
    """
    if not has_unstructured_input_feature(config_dict):
        return 0.001

    if not has_pretrained_encoder(config_dict):
        return 0.0001

    if has_trainable_encoder(config_dict):
        return 0.00001
    return 0.00002
