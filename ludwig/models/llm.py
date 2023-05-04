import copy
import logging
import os
import tempfile
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torchmetrics
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from ludwig.constants import LOGITS, MODEL_LLM
from ludwig.features.base_feature import OutputFeature
from ludwig.features.text_feature import TextOutputFeature
from ludwig.globals import MODEL_WEIGHTS_FILE_NAME
from ludwig.models.base import BaseModel
from ludwig.schema.features.base import BaseOutputFeatureConfig, FeatureCollection
from ludwig.schema.model_types.llm import LLMModelConfig
from ludwig.utils.augmentation_utils import AugmentationPipelines
from ludwig.utils.data_utils import clear_data_cache
from ludwig.utils.fs_utils import open_file
from ludwig.utils.state_dict_backward_compatibility import update_state_dict
from ludwig.utils.torch_utils import get_torch_device

logger = logging.getLogger(__name__)


class LLM(BaseModel):
    @staticmethod
    def type() -> str:
        return MODEL_LLM

    def __init__(
        self,
        config_obj: LLMModelConfig,
        random_seed=None,
        device=None,
        model=None,
        tokenizer=None,
        **_kwargs,
    ):
        super().__init__(random_seed=random_seed)

        self.config_obj = config_obj
        self.random_seed = random_seed

        self.model_name = self.config_obj.model_name

        logger.info("Loading large language model...")
        if not model:
            self.model = AutoModelForCausalLM.from_pretrained(self.config_obj.model_name)
            self.curr_device = torch.device("cpu")  # model initially loaded onto cpu
        else:
            self.model = model
            self.curr_device = self.model.device.type
        logger.info("Done.")

        # Used only for its metadata about the vocabulary
        if not tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config_obj.model_name, use_fast=False)
        else:
            self.tokenizer = tokenizer

        # Determines the maximum length of the context (input + output tokens)
        if hasattr(self.model.config, "max_sequence_length"):
            self.context_len = self.model.config.max_sequence_length
        elif hasattr(self.model.config, "max_position_embeddings"):
            self.context_len = self.model.config.max_position_embeddings
        else:
            self.context_len = 2048

        # max input length value copied from FastChat
        # https://github.com/lm-sys/FastChat/blob/0e958b852a14f4bef5f0e9d7a5e7373477329cf2/fastchat/serve/inference.py#L183  # noqa
        self.max_new_tokens = self.config_obj.generation.max_new_tokens
        self.max_input_length = self.context_len - self.max_new_tokens - 8

        self.generation = GenerationConfig(**self.config_obj.generation.to_dict())

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

        # Extract the decoder object for the forward pass
        _, self.output_feature_decoder = self.output_features.items()[0]

        # ================ Combined loss metric ================
        self.eval_loss_metric = torchmetrics.MeanMetric()
        self.eval_additional_losses_metrics = torchmetrics.MeanMetric()

        clear_data_cache()

    def to_device(self, device):
        device = torch.device(device)
        if device == self.curr_device:
            logger.warning(f"LLM already on device '{device}'. Skipping device placement step.")
            return self
        else:
            logger.info(f"Moving LLM from '{self.curr_device}' to '{device}'.")

        model_kwargs = {}
        if device == torch.device("cuda"):
            num_gpus = torch.cuda.device_count()
            # TODO: make this configurable in the future. These parameters are from FastChat:
            # https://github.com/lm-sys/FastChat/blob/0e958b852a14f4bef5f0e9d7a5e7373477329cf2/fastchat/serve/inference.py#L90  # noqa
            model_kwargs.update(
                dict(
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    max_memory={i: "13GiB" for i in range(num_gpus)},
                )
            )
            # we save and reload the weights to ensure that they can be sharded across the GPUs using `from_pretrained`
            with tempfile.TemporaryDirectory() as tmpdir:
                self.model.save_pretrained(tmpdir)
                self.model = AutoModelForCausalLM.from_pretrained(tmpdir, **model_kwargs)

            self.eval_loss_metric = self.eval_loss_metric.to(device)
            self.eval_additional_losses_metrics = self.eval_additional_losses_metrics.to(device)
            self.output_features.update({k: v.to(device) for k, v in self.output_features.items()})
        else:
            self.model = self.model.to(device)

        self.curr_device = device
        return self

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

    def forward(
        self,
        inputs: Union[
            Dict[str, torch.Tensor], Dict[str, np.ndarray], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ],
        mask=None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            inputs: Inputs to the model. Can be a dictionary of input names to
                input tensors or a tuple of (inputs, targets) where inputs is
                a dictionary of input names to input tensors and targets is a
                dictionary of target names to target tensors.
            mask: A mask for the inputs.

        Returns:
            A dictionary of output {feature name}::{tensor_name} -> output tensor.
        """

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

        with torch.no_grad():
            input_ids = self.get_input_ids(inputs)

            input_lengths = []
            sequences_list = []
            for input_ids_sample in input_ids:
                input_ids_sample_no_padding = self._remove_left_padding(input_ids_sample)

                if input_ids_sample_no_padding.shape[1] > self.max_input_length:
                    logger.warning(
                        f"Input length {input_ids_sample_no_padding.shape[1]} is "
                        f"greater than max input length {self.max_input_length}. Truncating."
                    )
                    input_ids_sample_no_padding = input_ids_sample_no_padding[:, -self.max_input_length :]

                input_lengths.append(input_ids_sample_no_padding.shape[1])

                # Generate text using the model
                model_outputs = self.model.generate(
                    input_ids=input_ids_sample_no_padding,
                    attention_mask=mask,
                    generation_config=self.generation,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                sequences_list.append(model_outputs.sequences[0])

            # Extract the predictions, probabilities and logits from the model outputs
            # through the forward pass of the output feature
            outputs = self.output_feature_decoder.decoder_obj.forward(
                sequences_list,
                llm_model_input_lengths=input_lengths,
            )
        return self.extract(outputs)

    def get_input_ids(self, inputs):
        """Returns the input ids for the text feature input."""
        return inputs[self.config_obj.input_features[0].name].type(torch.int32)

    def extract(
        self,
        outputs,
    ):
        """Extracts predictions and probabilities from the model outputs."""
        return {self.config_obj.output_features[0].name: outputs}

    def update_metrics(self, targets, predictions):
        """Updates the model's metrics given targets and predictions."""
        for of_name, of_obj in self.output_features.items():
            if isinstance(of_obj, TextOutputFeature):
                # Align the target length with the predictions length to enable text metric evaluation.
                _targets = self._realign_target_tensor(targets, predictions, of_name)
                of_obj.update_metrics(_targets[of_name], predictions[of_name])
                continue
            of_obj.update_metrics(targets[of_name], predictions[of_name])

        eval_loss, additional_losses = self.eval_loss(targets, predictions)
        self.eval_loss_metric.update(eval_loss)
        self.eval_additional_losses_metrics.update(additional_losses)

    def eval_loss(self, targets, predictions):
        """Computes all evaluation losses for the model given targets and predictions.

        Args:
            targets: A dictionary of target names to target tensors.
            predictions: A dictionary of output names to output tensors.

        Returns:
            A tuple of loss values for eval losses and additional losses.
        """
        eval_loss = 0
        for of_name, of_obj in self.output_features.items():
            if isinstance(of_obj, TextOutputFeature):
                # Align the target length with the predictions length to enable text metric evaluation.
                _targets = self._realign_target_tensor(targets, predictions, of_name)
                of_eval_loss = of_obj.eval_loss(_targets[of_name], predictions[of_name])
            else:
                # TODO(Arnav): Figure out loss updates.
                # To update eval-loss, we need "logits" but right now we're only producing "predictions"
                # This is required by the SequenceSoftmaxCrossEntropyLoss function
                # of_eval_loss = of_obj.eval_loss(targets[of_name], predictions[of_name])

                # HACK(geoffrey): we need a non-empty loss, so we just fill it with zeros
                of_eval_loss = torch.tensor(0.0).to(predictions[of_name][LOGITS].device)

            eval_loss += of_obj.loss.weight * of_eval_loss

        additional_loss = 0
        additional_losses = self.losses()
        if additional_losses:
            additional_loss = torch.sum(torch.stack(additional_losses))  # other losses

        return eval_loss, additional_loss

    def outputs_to_predictions(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Returns the model's predictions for each output feature."""
        predictions = {}
        for of_name in self.output_features:
            generated_predictions = outputs[of_name]
            predictions[of_name] = generated_predictions
        return predictions

    def save(self, save_path):
        """Saves the model to the given path."""
        # TODO: figure out what we want to do about weight saving for LLMs.
        # weights_save_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
        # torch.save(self.state_dict(), weights_save_path)
        logger.warning("Saving LLMs is not yet supported.")

    def load(self, save_path):
        """Loads the model from the given path."""
        weights_save_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
        device = torch.device(get_torch_device())
        with open_file(weights_save_path, "rb") as f:
            state_dict = torch.load(f, map_location=device)
            self.load_state_dict(update_state_dict(state_dict))

    def get_args(self):
        """Returns init arguments for constructing this model."""
        return (
            self.config_obj.input_features.to_list(),
            self.config_obj.output_features.to_list(),
            self.random_seed,
        )

    def _realign_target_tensor(self, targets, predictions, of_name: str):
        """Realigns the target tensor with the predictions.

        This is necessary for text metrics that require the target and prediction
        to be of the same length.

        Args:
            targets: The target tensor.
            predictions: The prediction tensor.

        Returns:
            The realigned target tensor.
        """
        _targets = copy.deepcopy(targets)
        _targets[of_name] = torch.nn.functional.pad(
            _targets.get(of_name),
            (0, predictions[of_name].get("predictions").size()[1] - _targets.get(of_name).size()[1]),
            "constant",
            0,
        )
        return _targets

    def _remove_left_padding(self, input_ids_sample: torch.Tensor):
        bos_idxs = torch.where(input_ids_sample == self.tokenizer.bos_token_id)[0]  # all BOS token locations
        if len(bos_idxs) != 0:
            bos_idx = bos_idxs[0]  # get first BOS token location
        else:
            bos_idx = 0

        input_ids_sample_no_padding = input_ids_sample[bos_idx:].unsqueeze(0)
        return input_ids_sample_no_padding

    def get_augmentation_pipelines(self) -> AugmentationPipelines:
        """Returns the augmentation pipeline for this model."""
        return AugmentationPipelines({})
