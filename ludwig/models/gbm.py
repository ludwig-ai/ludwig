import copy
from typing import Any, Dict, List, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from hummingbird.ml import convert

from ludwig.constants import BINARY, CATEGORY, LOGITS, MODEL_GBM, NAME, NUMBER
from ludwig.features.base_feature import OutputFeature
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.models.base import BaseModel
from ludwig.utils import output_feature_utils


class GBM(BaseModel):
    @staticmethod
    def type() -> str:
        return MODEL_GBM

    def __init__(self, input_features, output_features, random_seed=None, **_kwargs):
        super().__init__(random_seed=random_seed)

        self._input_features_def = copy.deepcopy(input_features)
        self._output_features_def = copy.deepcopy(output_features)

        # ================ Inputs ================
        self.input_features = LudwigFeatureDict()
        try:
            self.input_features.update(self.build_inputs(self._input_features_def))
        except KeyError as e:
            raise KeyError(
                f"An input feature has a name that conflicts with a class attribute of torch's ModuleDict: {e}"
            )

        # ================ Outputs ================
        self.output_features = LudwigFeatureDict()
        self.output_features.update(self.build_outputs(self._output_features_def, input_size=self.input_shape[-1]))

        # ================ Combined loss metric ================
        self.eval_loss_metric = torchmetrics.MeanMetric()
        self.eval_additional_losses_metrics = torchmetrics.MeanMetric()

        self._init_state_dict()

    def _init_state_dict(self):
        """Creates a dummy model to initialize this module's state dict."""
        output_feature_name = self.output_features.keys()[0]
        output_feature = self.output_features[output_feature_name]
        if output_feature.type() == CATEGORY:
            output_params = {"objective": "multiclass", "num_class": output_feature.num_classes}
        elif output_feature.type() == BINARY:
            output_params = {"objective": "binary"}
        elif output_feature.type() == NUMBER:
            output_params = {"objective": "regression"}
        else:
            raise ValueError(
                "Model type GBM only supports numerical, categorical, or binary output features,"
                f" found: {output_feature.type}"
            )

        df = pd.DataFrame({"label": 0.0, **{str(i): [0.0] for i in range(len(self.input_features))}})
        gbm = lgb.train(params={"verbosity": -1, **output_params}, train_set=lgb.Dataset(df), num_boost_round=1)
        self.compile(gbm)

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
        # HACK: avoid using non-tree decoder part of the model for inference
        output_feature.decoder_obj = nn.Identity()
        output_features[output_feature_def[NAME]] = output_feature

        return output_features

    def compile(self, gbm: lgb.Booster):
        """Convert the LightGBM model to a PyTorch model and store internally."""
        output_feature_name = self.output_features.keys()[0]
        output_feature = self.output_features[output_feature_name]

        # https://github.com/microsoft/LightGBM/issues/1942#issuecomment-453975607
        gbm_sklearn_cls = lgb.LGBMRegressor if output_feature.type() == NUMBER else lgb.LGBMClassifier
        gbm_sklearn = gbm_sklearn_cls(feature_name=list(self.input_features.keys()))  # , **params)
        gbm_sklearn._Booster = gbm
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

        # Convert inputs to tensors.
        for input_feature_name, input_values in inputs.items():
            if not isinstance(input_values, torch.Tensor):
                inputs[input_feature_name] = torch.from_numpy(input_values)
            else:
                inputs[input_feature_name] = input_values.view(-1, 1)

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
            logits = preds.squeeze()
        else:
            # classification
            _, probs = preds
            # keep positive class only for binary feature
            probs = probs[:, 1] if output_feature.type() == BINARY else probs
            logits = torch.logit(probs)
        output_feature_utils.set_output_feature_tensor(output_logits, output_feature_name, LOGITS, logits)

        return output_logits

    def get_args(self):
        """Returns init arguments for constructing this model."""
        return (self._input_features_df, self._output_features_df, self._random_seed)
