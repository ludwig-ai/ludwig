from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import torch

from ludwig.constants import (
    AUDIO,
    BAG,
    BINARY,
    CATEGORY,
    COLUMN,
    DATE,
    IMAGE,
    NAME,
    POSTPROCESSOR,
    PREDICTOR,
    PREPROCESSOR,
    SEQUENCE,
    SET,
    TEXT,
    TIMESERIES,
    TYPE,
    VECTOR,
)
from ludwig.types import FeatureConfigDict, ModelConfigDict
from ludwig.utils.audio_utils import read_audio_from_path
from ludwig.utils.date_utils import create_vector_from_datetime_obj
from ludwig.utils.image_utils import read_image_from_path
from ludwig.utils.torch_utils import place_on_device
from ludwig.utils.types import TorchDevice, TorchscriptPreprocessingInput

FEATURES_TO_CAST_AS_STRINGS = {BINARY, CATEGORY, BAG, SET, TEXT, SEQUENCE, TIMESERIES, VECTOR}


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
    dataset: pd.DataFrame, config: ModelConfigDict, load_paths: bool = False, device: Optional[torch.device] = None
) -> Dict[str, TorchscriptPreprocessingInput]:
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
    s: pd.Series, feature_type: str, load_paths: bool = False, feature_config: Optional[FeatureConfigDict] = None
) -> TorchscriptPreprocessingInput:
    """Converts a pandas Series to be compatible with a torchscripted InferenceModule forward pass."""
    if feature_type == IMAGE:
        if load_paths:
            return [read_image_from_path(v) if isinstance(v, str) else v for v in s]
    elif feature_type == AUDIO:
        if load_paths:
            return [read_audio_from_path(v) if isinstance(v, str) else v for v in s]
    elif feature_type == DATE:
        if feature_config is None:
            raise ValueError('"date" feature type requires the associated feature config to be provided.')
        datetime_format = feature_config["preprocessing"]["datetime_format"]
        return [torch.tensor(create_vector_from_datetime_obj(datetime.strptime(v, datetime_format))) for v in s]
    elif feature_type in FEATURES_TO_CAST_AS_STRINGS:
        return s.astype(str).to_list()
    return torch.from_numpy(s.to_numpy())
