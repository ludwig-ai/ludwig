import copy
import os
from typing import Any, Dict, List, Tuple, Union

import lightgbm as lgb
import numpy as np
import torch
import torchmetrics
from hummingbird.ml import convert

from ludwig.constants import BINARY, CATEGORY, LOGITS, MODEL_GBM, NAME, NUMBER
from ludwig.features.base_feature import OutputFeature
from ludwig.globals import MODEL_WEIGHTS_FILE_NAME
from ludwig.models.base import BaseModel
from ludwig.utils import output_feature_utils
from ludwig.utils.torch_utils import get_torch_device


class GBM(BaseModel):
    @staticmethod
    def type() -> str:
        return MODEL_GBM

    def __init__(
        self,
        input_features: List[Dict[str, Any]],
        output_features: List[Dict[str, Any]],
        random_seed: int = None,
        **_kwargs,
    ):
        super().__init__(random_seed=random_seed)

        self._input_features_def = copy.deepcopy(input_features)
        self._output_features_def = copy.deepcopy(output_features)

        # ================ Inputs ================
        try:
            self.input_features.update(self.build_inputs(self._input_features_def))
        except KeyError as e:
            raise KeyError(
                f"An input feature has a name that conflicts with a class attribute of torch's ModuleDict: {e}"
            )

        # ================ Outputs ================
        self.output_features.update(self.build_outputs(self._output_features_def, input_size=self.input_shape[-1]))

        # ================ Combined loss metric ================
        self.eval_loss_metric = torchmetrics.MeanMetric()
        self.eval_additional_losses_metrics = torchmetrics.MeanMetric()

        self.lgb_booster: lgb.Booster = None
        self.compiled_model: torch.nn.Module = None

    @classmethod
    def build_outputs(cls, output_features_def: List[Dict[str, Any]], input_size: int) -> Dict[str, OutputFeature]:
        """Builds and returns output feature."""
        # TODO: only single task currently
        if len(output_features_def) > 1:
            raise ValueError("Only single task currently supported")

        output_feature_def = output_features_def[0]
        output_features = {}

        output_feature_def["input_size"] = input_size
        output_feature = cls.build_single_output(output_feature_def, output_features)
        output_features[output_feature_def[NAME]] = output_feature

        return output_features

    def compile(self):
        """Convert the LightGBM model to a PyTorch model and store internally."""
        if self.lgb_booster is None:
            raise ValueError("Model has not been trained yet.")

        output_feature_name = self.output_features.keys()[0]
        output_feature = self.output_features[output_feature_name]

        # https://github.com/microsoft/LightGBM/issues/1942#issuecomment-453975607
        gbm_sklearn_cls = lgb.LGBMRegressor if output_feature.type() == NUMBER else lgb.LGBMClassifier
        gbm_sklearn = gbm_sklearn_cls(feature_name=list(self.input_features.keys()))  # , **params)
        gbm_sklearn._Booster = self.lgb_booster
        gbm_sklearn.fitted_ = True
        gbm_sklearn._n_features = len(self.input_features)
        if isinstance(gbm_sklearn, lgb.LGBMClassifier):
            gbm_sklearn._n_classes = output_feature.num_classes if output_feature.type() == CATEGORY else 2

        hb_model = convert(gbm_sklearn, "torch", extra_config={"tree_implementation": "gemm"})

        self.compiled_model = hb_model.model

    def forward(
        self,
        inputs: Union[
            Dict[str, torch.Tensor], Dict[str, np.ndarray], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ],
        mask=None,
    ) -> Dict[str, torch.Tensor]:
        if self.compiled_model is None:
            raise ValueError("Model has not been trained yet.")

        if isinstance(inputs, tuple):
            inputs, targets = inputs
            # Convert targets to tensors.
            for target_feature_name, target_value in targets.items():
                if not isinstance(target_value, torch.Tensor):
                    targets[target_feature_name] = torch.from_numpy(target_value)
                else:
                    targets[target_feature_name] = target_value
        else:
            targets = None

        assert list(inputs.keys()) == self.input_features.keys()

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

        # Invoke output features.
        output_logits = {}
        output_feature_name = self.output_features.keys()[0]
        output_feature = self.output_features[output_feature_name]

        preds = self.compiled_model(inputs)

        if output_feature.type() == NUMBER:
            # regression
            if len(preds.shape) == 2:
                preds = preds.squeeze(1)
            logits = preds
        else:
            # classification
            _, probs = preds
            # keep positive class only for binary feature
            probs = probs[:, 1] if output_feature.type() == BINARY else probs
            logits = torch.logit(probs)
        output_feature_utils.set_output_feature_tensor(output_logits, output_feature_name, LOGITS, logits)

        return output_logits

    def save(self, save_path):
        """Saves the model to the given path."""
        if self.lgb_booster is None:
            raise ValueError("Model has not been trained yet.")

        weights_save_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
        self.lgb_booster.save_model(weights_save_path, num_iteration=self.lgb_booster.best_iteration)

    def load(self, save_path):
        """Loads the model from the given path."""
        weights_save_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
        self.lgb_booster = lgb.Booster(model_file=weights_save_path)
        self.compile()

        device = torch.device(get_torch_device())
        self.compiled_model.to(device)

    def get_args(self):
        """Returns init arguments for constructing this model."""
        return (self._input_features_df, self._output_features_df, self._random_seed)
