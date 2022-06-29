import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch

from ludwig.features.date_feature import create_vector_from_datetime_obj
from ludwig.utils.audio_utils import read_audio_from_path
from ludwig.utils.image_utils import read_image_from_path
from ludwig.utils.torch_utils import place_on_device
from ludwig.utils.types import TorchDevice, TorchscriptPreprocessingInput
from ludwig.utils.output_feature_utils import get_feature_name_from_concat_name, get_tensor_name_from_concat_name


# Duplicated from ludwig.constants to minimize dependencies.
PREPROCESSOR = "preprocessor"
PREDICTOR = "predictor"
POSTPROCESSOR = "postprocessor"
BINARY = "binary"
CATEGORY = "category"
BAG = "bag"
SET = "set"
TEXT = "text"
SEQUENCE = "sequence"
TIMESERIES = "timeseries"
VECTOR = "vector"
COLUMN = "column"
TYPE = "type"
NAME = "name"

# Duplicated from ludwig.utils.types to minimize dependencies.
TorchAudioTuple = Tuple[torch.Tensor, int]
TorchscriptPreprocessingInput = Union[List[str], List[torch.Tensor], List[TorchAudioTuple], torch.Tensor]
TorchDevice = Union[str, torch.device]

FEATURES_TO_CAST_AS_STRINGS = {BINARY, CATEGORY, BAG, SET, TEXT, SEQUENCE, TIMESERIES, VECTOR}


def unflatten_dict_by_feature_name(flattened_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Convert a flattened dictionary of objects to a nested dictionary of outputs per feature name."""
    outputs: Dict[str, Dict[str, Any]] = {}
    for concat_key, tensor_values in flattened_dict.items():
        feature_name = get_feature_name_from_concat_name(concat_key)
        tensor_name = get_tensor_name_from_concat_name(concat_key)
        feature_outputs: Dict[str, Any] = {}
        if feature_name not in outputs:
            outputs[feature_name] = feature_outputs
        else:
            feature_outputs = outputs[feature_name]
        feature_outputs[tensor_name] = tensor_values
    return outputs


def get_filename_from_stage(stage: str, device: TorchDevice) -> str:
    """Returns the filename for a stage of inference."""
    if stage not in [PREPROCESSOR, PREDICTOR, POSTPROCESSOR]:
        raise ValueError(f"Invalid stage: {stage}.")
    # device is only tracked for predictor stage
    if stage == PREDICTOR:
        return f"inference_{stage}-{device}.pt"
    else:
        return f"inference_{stage}.pt"


def to_inference_module_input_from_dataframe(
    dataset: pd.DataFrame, config: Dict[str, Any], load_paths: bool = False, device: Optional[torch.device] = None
) -> TorchscriptPreprocessingInput:
    """Converts a pandas DataFrame to be compatible with a torchscripted InferenceModule forward pass."""
    inputs = {}
    for if_config in config["input_features"]:
        feature_inputs = to_inference_model_input_from_series(
            dataset[if_config[COLUMN]],
            if_config[TYPE],
            load_paths=load_paths,
            feature_config=if_config,
        )
        feature_inputs = place_on_device(feature_inputs, device)
        inputs[if_config[NAME]] = feature_inputs
    return inputs


def to_inference_model_input_from_series(
    s: pd.Series, feature_type: str, load_paths: bool = False, feature_config: Optional[Dict[str, Any]] = None
) -> Union[List[str], torch.Tensor]:
    """Converts a pandas Series to be compatible with a torchscripted InferenceModule forward pass."""
    if feature_type == "image":
        if load_paths:
            return [read_image_from_path(v) if isinstance(v, str) else v for v in s]
    elif feature_type == "audio":
        if load_paths:
            return [read_audio_from_path(v) if isinstance(v, str) else v for v in s]
    elif feature_type == "date":
        if feature_config is None:
            raise ValueError('"date" feature type requires the associated feature config to be provided.')
        datetime_format = feature_config["preprocessing"]["datetime_format"]
        return [torch.tensor(create_vector_from_datetime_obj(datetime.strptime(v, datetime_format))) for v in s]
    elif feature_type in FEATURES_TO_CAST_AS_STRINGS:
        return s.astype(str).to_list()
    return torch.from_numpy(s.to_numpy())
