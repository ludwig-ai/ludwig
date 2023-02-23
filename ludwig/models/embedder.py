from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BINARY, CATEGORY, NAME, NUMBER, PROC_COLUMN, TYPE
from ludwig.features.feature_registries import get_input_type_registry
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.models.base import BaseModel
from ludwig.schema.features.base import BaseInputFeatureConfig, FeatureCollection
from ludwig.types import FeatureConfigDict, TrainingSetMetadataDict
from ludwig.utils.batch_size_tuner import BatchSizeEvaluator
from ludwig.utils.dataframe_utils import from_numpy_dataset
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.torch_utils import get_torch_device, LudwigModule

_SCALAR_TYPES = {BINARY, CATEGORY, NUMBER}


@DeveloperAPI
class Embedder(LudwigModule):
    def __init__(self, feature_configs: List[FeatureConfigDict], metadata: TrainingSetMetadataDict):
        super().__init__()

        self.input_features = LudwigFeatureDict()

        input_feature_configs = []
        for feature in feature_configs:
            feature_cls = get_from_registry(feature[TYPE], get_input_type_registry())
            feature_obj = feature_cls.get_schema_cls().from_dict(feature)
            feature_cls.update_config_with_metadata(feature_obj, metadata[feature[NAME]])

            # When running prediction or eval, we need the preprocessing to use the original pretrained
            # weights, which requires unsetting this field. In the future, we could avoid this by plumbing
            # through the saved weights and loading them dynamically after building the model.
            feature_obj.encoder.saved_weights_in_checkpoint = False

            input_feature_configs.append(feature_obj)

        feature_collection = FeatureCollection[BaseInputFeatureConfig](input_feature_configs)
        try:
            self.input_features.update(BaseModel.build_inputs(input_feature_configs=feature_collection))
        except KeyError as e:
            raise KeyError(
                f"An input feature has a name that conflicts with a class attribute of torch's ModuleDict: {e}"
            )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        encoder_outputs = {}
        for input_feature_name, input_values in inputs.items():
            encoder = self.input_features[input_feature_name]
            encoder_output = encoder(input_values)
            encoder_outputs[input_feature_name] = encoder_output["encoder_output"]
        return encoder_outputs


@DeveloperAPI
def create_embed_batch_size_evaluator(
    features_to_encode: List[FeatureConfigDict], metadata: TrainingSetMetadataDict
) -> BatchSizeEvaluator:
    class _EmbedBatchSizeEvaluator(BatchSizeEvaluator):
        def __init__(self):
            embedder = Embedder(features_to_encode, metadata)
            self.device = get_torch_device()
            self.embedder = embedder.to(self.device)
            self.embedder.eval()

        def step(self, batch_size: int):
            inputs = {
                input_feature_name: input_feature.create_sample_input(batch_size=batch_size).to(self.device)
                for input_feature_name, input_feature in self.embedder.input_features.items()
            }
            with torch.no_grad():
                self.embedder(inputs)

    return _EmbedBatchSizeEvaluator


@DeveloperAPI
def create_embed_transform_fn(
    features_to_encode: List[FeatureConfigDict], metadata: TrainingSetMetadataDict
) -> Callable:
    class EmbedTransformFn:
        def __init__(self):
            embedder = Embedder(features_to_encode, metadata)
            self.device = get_torch_device()
            self.embedder = embedder.to(self.device)
            self.embedder.eval()

        def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
            batch = _prepare_batch(df, features_to_encode, metadata)
            name_to_proc = {i_feat.feature_name: i_feat.proc_column for i_feat in self.embedder.input_features.values()}
            inputs = {
                i_feat.feature_name: torch.from_numpy(np.array(batch[i_feat.proc_column], copy=True)).to(self.device)
                for i_feat in self.embedder.input_features.values()
            }
            with torch.no_grad():
                encoder_outputs = self.embedder(inputs)

            encoded = {name_to_proc[k]: v.detach().cpu().numpy() for k, v in encoder_outputs.items()}
            output_df = from_numpy_dataset(encoded)

            for c in output_df.columns:
                df[c] = output_df[c]

            return df

    return EmbedTransformFn


def _prepare_batch(
    df: pd.DataFrame, features: List[FeatureConfigDict], metadata: TrainingSetMetadataDict
) -> Dict[str, np.ndarray]:
    batch = {}
    for feature in features:
        c = feature[PROC_COLUMN]
        if feature[TYPE] not in _SCALAR_TYPES:
            # Ensure columns stacked instead of turned into np.array([np.array, ...], dtype=object) objects
            batch[c] = np.stack(df[c].values)
        else:
            batch[c] = df[c].to_numpy()

    for feature in features:
        c = feature[PROC_COLUMN]
        reshape = metadata.get(feature[NAME], {}).get("reshape")
        if reshape is not None:
            batch[c] = batch[c].reshape((-1, *reshape))

    return batch
