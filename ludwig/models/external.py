from abc import abstractmethod
from typing import Dict

import torch

from ludwig.constants import MODEL_EXTERNAL
from ludwig.features.text_feature import TextOutputFeature
from ludwig.models.base import BaseModel
from ludwig.schema.features.base import BaseOutputFeatureConfig, FeatureCollection
from ludwig.schema.model_types.external import ExternalModelConfig
from ludwig.utils.misc_utils import set_random_seed


class External(BaseModel):
    """Base model for use with external models.

    Implementations of this class should implement the following methods:
    - evaluate()
    """

    @staticmethod
    def type() -> str:
        return MODEL_EXTERNAL

    def __init__(self, random_seed: int = None):
        self.config_obj: ExternalModelConfig = dict()
        self._random_seed = random_seed

        if random_seed is not None:
            set_random_seed(random_seed)

        super().__init__()

        self.input_features = self.create_feature_dict()
        self.output_features = self.create_feature_dict()

    @abstractmethod
    def evaluate(self, inputs: Dict) -> Dict:
        """Evaluates inputs using the model."""
        raise NotImplementedError("evaluate() is not implemented")

    @classmethod
    def build_outputs(
        cls, output_feature_configs: FeatureCollection[BaseOutputFeatureConfig]
    ) -> Dict[str, TextOutputFeature]:
        """Builds and returns output feature."""
        # TODO: only single task currently
        if len(output_feature_configs) > 1:
            raise ValueError("The LLM model type only supports a single output feature.")

        output_feature_config = output_feature_configs[0]
        output_feature_config.input_size = torch.Size([1, 1])[-1]

        output_features = {}
        output_feature = cls.build_single_output(output_feature_config, output_features)
        output_features[output_feature_config.name] = output_feature

        return output_features

    def to_device(self, device):
        raise TypeError("to_device() is not supported for External")

    def metrics_to_device(self, device: str):
        raise TypeError("metrics_to_device() is not supported for External")

    def get_model_size(self) -> int:
        raise TypeError("get_model_size() is not supported for External")

    def to_torchscript(self, *args):
        raise TypeError("to_torchscript() is not supported for External")

    def save_torchscript(self, *args):
        raise TypeError("save_torchscript() is not supported for External")

    @property
    def input_shape(self):
        raise TypeError("input_shape() is not supported for External")

    def forward(self, *args):
        raise TypeError("forward() is not supported for External")

    def predictions(self, inputs):
        raise TypeError("predictions() is not supported for External")

    def evaluation_step(self, inputs, targets):
        raise TypeError("evaluation_step() is not supported for External")

    def predict_step(self, inputs):
        raise TypeError("predict_step() is not supported for External")

    def train_loss(self, *args):
        raise TypeError("train_loss() is not supported for External")

    def collect_weights(self, tensor_names=None, **kwargs):
        raise TypeError("collect_weights() is not supported for External")

    def unskip(self):
        raise TypeError("unskip() is not supported for External")

    def save(self, save_path: str):
        raise TypeError("save() is not supported for External")

    def load(self, save_path: str):
        raise TypeError("load() is not supported for External")

    def get_args(self):
        raise TypeError("get_args() is not supported for External")

    @property
    def device(self):
        raise TypeError("device() is not supported for External")

    def prepare_for_training(self):
        raise TypeError("prepare_for_training() is not supported for External")

    def update_loss(self, *args):
        raise TypeError("update_loss() is not supported for External")

    @property
    def input_dtype(self):
        raise TypeError("input_dtype() is not supported for External")

    @property
    def output_shape(self):
        raise TypeError("output_shape() is not supported for External")

    def _computed_output_shape(self):
        raise TypeError("_computed_output_shape() is not supported for External")
