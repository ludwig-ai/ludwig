from typing import Union, Dict, Tuple, Optional

import numpy as np
import torch
from hummingbird.ml import convert

from ludwig.constants import COMBINED, LOSS
from ludwig.utils.torch_utils import LudwigModule, reg_loss


class TreeModule(LudwigModule):
    def __init__(
            self,
            compiled_model,
            input_features,
            output_features
    ):
        super().__init__()
        self.compiled_model = compiled_model
        self.input_features = input_features
        self.output_features = output_features

    def get_model_inputs(self, training=True):
        inputs = {
            input_feature_name: input_feature.create_input()
            for input_feature_name, input_feature in
            self.input_features.items()
        }

        if not training:
            return inputs

        targets = {
            output_feature_name: output_feature.create_input()
            for output_feature_name, output_feature in
            self.output_features.items()
        }
        return inputs, targets

    def forward(
            self,
            inputs: Union[
                Dict[str, torch.Tensor],
                Dict[str, np.ndarray],
                Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            ],
            mask=None
    ) -> Dict[str, torch.Tensor]:
        if isinstance(inputs, tuple):
            inputs, targets = inputs
            # Convert targets to tensors.
            for target_feature_name, target_value in targets.items():
                if not isinstance(target_value, torch.Tensor):
                    targets[target_feature_name] = torch.from_numpy(
                        target_value)
                else:
                    targets[target_feature_name] = target_value
        else:
            targets = None
        assert inputs.keys() == self.input_features.keys()

        # Convert inputs to tensors.
        for input_feature_name, input_values in inputs.items():
            if not isinstance(input_values, torch.Tensor):
                inputs[input_feature_name] = torch.from_numpy(input_values)
            else:
                inputs[input_feature_name] = input_values

        # TODO(travis): include encoder and decoder steps during inference

        # encoder_outputs = {}
        # for input_feature_name, input_values in inputs.items():
        #     encoder = self.input_features[input_feature_name]
        #     encoder_output = encoder(input_values)
        #     encoder_outputs[input_feature_name] = encoder_output

        # combiner_outputs = self.combiner(encoder_outputs)
        #
        # output_logits = {}
        # output_last_hidden = {}
        # for output_feature_name, decoder in self.output_features.items():
        #     # use presence or absence of targets
        #     # to signal training or prediction
        #     decoder_inputs = (combiner_outputs, copy.copy(output_last_hidden))
        #     if targets is not None:
        #         # targets are only used during training,
        #         # during prediction they are omitted
        #         decoder_inputs = (decoder_inputs, targets[output_feature_name])
        #
        #     decoder_outputs = decoder(decoder_inputs, mask=mask)
        #     output_logits[output_feature_name] = decoder_outputs
        #     output_last_hidden[output_feature_name] = decoder_outputs[
        #         'last_hidden']

        output_logits = self.compiled_model(inputs)

        return output_logits

    def predictions(self, inputs, output_features=None):
        # check validity of output_features
        if output_features is None:
            of_list = self.output_features
        elif isinstance(output_features, str):
            if output_features == 'all':
                of_list = set(self.output_features.keys())
            elif output_features in self.output_features:
                of_list = [output_features]
            else:
                raise ValueError(
                    "'output_features' {} is not a valid for this model. "
                    "Available ones are: {}".format(
                        output_features, set(self.output_features.keys())
                    )
                )
        elif isinstance(output_features, list or set):
            if output_features.issubset(self.output_features):
                of_list = output_features
            else:
                raise ValueError(
                    "'output_features' {} must be a subset of "
                    "available features {}".format(
                        output_features, set(self.output_features.keys())
                    )
                )
        else:
            raise ValueError(
                "'output_features' must be None or a string or a list "
                "of output features"
            )

        outputs = self(inputs)

        predictions = {}
        for of_name in of_list:
            predictions[of_name] = self.output_features[of_name].predictions(
                outputs[of_name]
            )

        return predictions

    def evaluation_step(self, inputs, targets):
        predictions = self.predictions(inputs, output_features=None)
        self.update_metrics(targets, predictions)
        return predictions

    def predict_step(self, inputs):
        return self.predictions(inputs, output_features=None)

    def train_loss(
            self,
            targets,
            predictions,
            regularization_type: Optional[str] = None,
            regularization_lambda: Optional[float] = None
    ):
        train_loss = 0
        of_train_losses = {}
        for of_name, of_obj in self.output_features.items():
            of_train_loss = of_obj.train_loss(targets[of_name],
                                              predictions[of_name])
            train_loss += of_obj.loss['weight'] * of_train_loss
            of_train_losses[of_name] = of_train_loss

        for loss in self.losses():
            train_loss += loss

        # Add regularization loss
        if regularization_type is not None:
            train_loss += reg_loss(
                self,
                regularization_type,
                l1=regularization_lambda,
                l2=regularization_lambda
            )

        return train_loss, of_train_losses

    def eval_loss(self, targets, predictions):
        eval_loss = 0
        of_eval_losses = {}
        for of_name, of_obj in self.output_features.items():
            of_eval_loss = of_obj.eval_loss(
                targets[of_name], predictions[of_name]
            )
            eval_loss += of_obj.loss['weight'] * of_eval_loss
            of_eval_losses[of_name] = of_eval_loss
        eval_loss += sum(self.losses())  # regularization / other losses
        return eval_loss, of_eval_losses

    def update_metrics(self, targets, predictions):
        for of_name, of_obj in self.output_features.items():
            of_obj.update_metrics(targets[of_name], predictions[of_name])

        self.eval_loss_metric.update(self.eval_loss(targets, predictions)[0])

    def get_metrics(self):
        all_of_metrics = {}
        for of_name, of_obj in self.output_features.items():
            all_of_metrics[of_name] = of_obj.get_metrics()
        all_of_metrics[COMBINED] = {
            LOSS: self.eval_loss_metric.compute().detach().numpy().item()
        }
        return all_of_metrics

    def reset_metrics(self):
        for of_obj in self.output_features.values():
            of_obj.reset_metrics()
        self.eval_loss_metric.reset()

    def collect_weights(
            self,
            tensor_names=None,
            **kwargs
    ):
        """Returns named parameters filtered against `tensor_names` if not None."""
        if not tensor_names:
            return self.named_parameters()

        # Check for bad tensor names.
        weight_names = set(name for name, _ in self.named_parameters())
        for name in tensor_names:
            if name not in weight_names:
                raise ValueError(
                    f'Requested tensor name filter "{name}" not present in the model graph')

        # Apply filter.
        tensor_set = set(tensor_names)
        return [named_param for named_param in self.named_parameters() if named_param[0] in tensor_set]

    def get_args(self):
        return self._input_features_df, self._combiner_def, self._output_features_df, self._random_seed


def convert_to_pytorch(tree_model, ecd):
    hb_model = convert(tree_model, 'torch')
    model = hb_model.model
    return TreeModule(model, ecd.input_features, ecd.output_features)
