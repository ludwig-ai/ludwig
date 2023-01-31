import os
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import torch
import torchmetrics
from hummingbird.ml import convert

from ludwig.constants import BINARY, LOGITS, MODEL_GBM, NUMBER
from ludwig.features.base_feature import OutputFeature
from ludwig.globals import MODEL_WEIGHTS_FILE_NAME
from ludwig.models.base import BaseModel
from ludwig.schema.features.base import BaseOutputFeatureConfig, FeatureCollection
from ludwig.schema.model_config import ModelConfig
from ludwig.utils import output_feature_utils
from ludwig.utils.fs_utils import path_exists
from ludwig.utils.gbm_utils import reshape_logits
from ludwig.utils.torch_utils import get_torch_device
from ludwig.utils.types import TorchDevice


class GBM(BaseModel):
    @staticmethod
    def type() -> str:
        return MODEL_GBM

    def __init__(
        self,
        config_obj: ModelConfig,
        random_seed: int = None,
        **_kwargs,
    ):
        self.config_obj = config_obj
        self._random_seed = random_seed

        super().__init__(random_seed=self._random_seed)

        # ================ Inputs ================
        try:
            self.input_features.update(self.build_inputs(input_feature_configs=self.config_obj.input_features))
        except KeyError as e:
            raise KeyError(
                f"An input feature has a name that conflicts with a class attribute of torch's ModuleDict: {e}"
            )

        # ================ Outputs ================
        self.output_features.update(
            self.build_outputs(output_feature_configs=self.config_obj.output_features, input_size=self.input_shape[-1])
        )

        # ================ Combined loss metric ================
        self.eval_loss_metric = torchmetrics.MeanMetric()
        self.eval_additional_losses_metrics = torchmetrics.MeanMetric()

        self.lgbm_model: lgb.LGBMModel = None
        self.compiled_model: torch.nn.Module = None

    @classmethod
    def build_outputs(
        cls, output_feature_configs: FeatureCollection[BaseOutputFeatureConfig], input_size: int
    ) -> Dict[str, OutputFeature]:
        """Builds and returns output feature."""
        # TODO: only single task currently
        if len(output_feature_configs) > 1:
            raise ValueError("Only single task currently supported")

        output_feature_config = output_feature_configs[0]
        output_feature_config.input_size = input_size

        output_features = {}
        output_feature = cls.build_single_output(output_feature_config, output_features)
        output_features[output_feature_config.name] = output_feature

        return output_features

    @contextmanager
    def compile(self):
        """Convert the LightGBM model to a PyTorch model and store internally."""
        if self.lgbm_model is None:
            raise ValueError("Model has not been trained yet.")

        try:
            self.compiled_model = convert(
                self.lgbm_model,
                "torch",
                extra_config={
                    # explicitly disable post-transform, so we get logits at inference time
                    "post_transform": None,
                    # return pytorch module only
                    "container": False,
                },
            )
            yield
        finally:
            self.compiled_model = None

    def forward(
        self,
        inputs: Union[
            Dict[str, torch.Tensor], Dict[str, np.ndarray], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ],
        mask=None,
    ) -> Dict[str, torch.Tensor]:
        # Invoke output features.
        output_logits = {}
        output_feature_name = self.output_features.keys()[0]
        output_feature = self.output_features[output_feature_name]

        # If `inputs` is a tuple, it should contain `(inputs, targets)`.
        if isinstance(inputs, tuple):
            inputs, _ = inputs

        assert list(inputs.keys()) == self.input_features.keys()

        # If the model has not been compiled, predict using the LightGBM sklearn iterface. Otherwise, use torch with
        # the Hummingbird compiled model. Notably, when compiling the model to torchscript, compiling with Hummingbird
        # first should preserve the torch predictions code path.
        if self.compiled_model is None:
            # The LGBM sklearn interface works with array-likes, so we place the inputs into a 2D numpy array.
            in_array = np.stack(list(inputs.values()), axis=0).T

            # Predict on the input batch and convert the predictions to torch tensors so that they are compatible with
            # the existing metrics modules.
            # Input: 2D eval_batch_size x n_features array
            # Output: 1D eval_batch_size array if regression, else 2D eval_batch_size x n_classes array
            logits = torch.from_numpy(self.lgbm_model.predict(in_array, raw_score=True))
            logits = reshape_logits(output_feature, logits)
        else:
            # Convert inputs to tensors of type float as expected by hummingbird GEMMTreeImpl.
            for input_feature_name, input_values in inputs.items():
                if not isinstance(input_values, torch.Tensor):
                    inputs[input_feature_name] = torch.from_numpy(input_values).float()
                else:
                    inputs[input_feature_name] = input_values.view(-1, 1).float()

            # TODO(travis): include encoder and decoder steps during inference
            # encoder_outputs = {}
            # for input_feature_name, input_values in inputs.items():
            #     encoder = self.input_features[input_feature_name]
            #     encoder_output = encoder(input_values)
            #     encoder_outputs[input_feature_name] = encoder_output

            # concatenate inputs
            inputs = torch.cat(list(inputs.values()), dim=1)

            assert (
                type(inputs) is torch.Tensor
                and inputs.dtype == torch.float32
                and inputs.ndim == 2
                and inputs.shape[1] == len(self.input_features)
            ), (
                f"Expected inputs to be a 2D tensor of shape (batch_size, {len(self.input_features)}) of type float32, "
                f"but got {inputs.shape} of type {inputs.dtype}"
            )
            # Predict using PyTorch module, so it is included when converting to TorchScript.
            preds = self.compiled_model(inputs)

            if output_feature.type() == NUMBER:
                # regression
                logits = preds.view(-1)
            else:
                # classification
                _, logits = preds
                logits = reshape_logits(output_feature, logits)

                if output_feature.type() == BINARY:
                    logits = logits.view(-1)

        output_feature_utils.set_output_feature_tensor(output_logits, output_feature_name, LOGITS, logits)

        return output_logits

    def save(self, save_path):
        """Saves the model to the given path."""
        if self.lgbm_model is None:
            raise ValueError("Model has not been trained yet.")

        import joblib

        weights_save_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
        joblib.dump(self.lgbm_model, weights_save_path)

    def load(self, save_path):
        """Loads the model from the given path."""
        import joblib

        weights_save_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
        self.lgbm_model = joblib.load(weights_save_path)

    def to_torchscript(self, device: Optional[TorchDevice] = None):
        """Converts the ECD model as a TorchScript model."""
        with self.compile():
            # Disable gradient calculation for hummingbird Parameter nodes.
            device = torch.device(get_torch_device())
            self.compiled_model.to(device)
            self.compiled_model.requires_grad_(False)
            trace = super().to_torchscript(device)
        return trace

    def has_saved(self, save_path):
        return path_exists(os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME))

    def get_args(self):
        """Returns init arguments for constructing this model."""
        return self.config_obj.input_features.to_list(), self.config_obj.output_features.to_list(), self._random_seed
