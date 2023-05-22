import contextlib
import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    GPT2Tokenizer,
    GPT2TokenizerFast,
    LlamaConfig,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedTokenizer,
)

from ludwig.constants import LOGITS, MODEL_LLM, PREDICTIONS, PROBABILITIES, TEXT
from ludwig.features.base_feature import ModuleWrapper, OutputFeature
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.features.text_feature import TextOutputFeature
from ludwig.globals import MODEL_WEIGHTS_FILE_NAME
from ludwig.models.base import BaseModel
from ludwig.schema.features.base import BaseOutputFeatureConfig, FeatureCollection
from ludwig.schema.model_types.llm import LLMModelConfig
from ludwig.utils.augmentation_utils import AugmentationPipelines
from ludwig.utils.data_utils import clear_data_cache
from ludwig.utils.logging_utils import log_once
from ludwig.utils.output_feature_utils import set_output_feature_tensor
from ludwig.utils.torch_utils import reg_loss

logger = logging.getLogger(__name__)


class DictWrapper:
    """Wrapper for a LudwigFeatureDict module that allows for iteration over keys.

    The purpose of this class is to avoid exposing input and output features as modules of the LLM. This is because we
    only wish to train the underlying model, and having these additional modules can confuse systems like DeepSpeed.
    """

    def __init__(self, obj: LudwigFeatureDict):
        self.obj = obj

    def get(self, key) -> torch.nn.Module:
        return self.obj.get(key)

    def set(self, key: str, module: torch.nn.Module) -> None:
        self.obj.set(key, module)

    def __len__(self) -> int:
        return len(self.obj)

    def __next__(self) -> None:
        return next(iter(self.obj))

    def __iter__(self) -> None:
        return iter(self.obj.keys())

    def keys(self) -> List[str]:
        return self.obj.keys()

    def values(self) -> List[torch.nn.Module]:
        return self.obj.values()

    def items(self) -> List[Tuple[str, torch.nn.Module]]:
        return self.obj.items()

    def update(self, modules: Dict[str, torch.nn.Module]) -> None:
        self.obj.update(modules)


class LLM(BaseModel):
    @staticmethod
    def type() -> str:
        return MODEL_LLM

    def __init__(
        self,
        config_obj: LLMModelConfig,
        random_seed=None,
        device=None,
        **_kwargs,
    ):
        super().__init__(random_seed=random_seed)

        self.config_obj = config_obj
        self._random_seed = random_seed

        self.model_name = self.config_obj.model_name

        logger.info("Loading large language model...")
        self.model = AutoModelForCausalLM.from_pretrained(self.config_obj.model_name)
        self.curr_device = torch.device("cpu")  # model initially loaded onto cpu
        logger.info("Done.")

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

        # Initialize tokenizer
        use_fast = True
        if isinstance(AutoConfig.from_pretrained(self.config_obj.model_name), LlamaConfig):
            # HACK: Llama fast tokenizer takes about 2-4 minutes to load, so we disable it for now.
            use_fast = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.config_obj.model_name, use_fast=use_fast)
        self._set_pad_token()

        self.generation = GenerationConfig(**self.config_obj.generation.to_dict())

        # ================ Inputs ================
        try:
            self.input_features.update(self.build_inputs(input_feature_configs=self.config_obj.input_features))
        except KeyError as e:
            raise KeyError(
                f"An input feature has a name that conflicts with a class attribute of torch's ModuleDict: {e}"
            )

        # ================ Outputs ================
        self.output_feature_type = self.config_obj.output_features[0].type

        self.output_features.update(
            self.build_outputs(
                output_feature_configs=self.config_obj.output_features,
                # Set the input size to the model vocab size instead of the tokenizer vocab size
                # because the model has additional "head" layers that are used to predict the next
                # token in the sequence. These head layers can add additional dimensions to the
                # logits tensor, beyond the vocab_size dimension.
                input_size=self.input_shape[-1] if self.output_feature_type == TEXT else self.model.config.vocab_size,
            )
        )

        # Extract the decoder object for the forward pass
        self._output_feature_decoder = ModuleWrapper(self.output_features.items()[0][1])

        clear_data_cache()

    def create_feature_dict(self) -> LudwigFeatureDict:
        return DictWrapper(LudwigFeatureDict())

    @property
    def output_feature_decoder(self) -> OutputFeature:
        return self._output_feature_decoder.module

    def initialize_adapter(self):
        """If an adapter config is provided, we want to wrap the model with a PEFT model for fine-tuning."""
        if self.config_obj.adapter:
            from peft import get_peft_model

            peft_config = self.config_obj.adapter.to_config(
                task_type="CAUSAL_LM", tokenizer_name_or_path=self.model_name
            )
            self.model = get_peft_model(self.model, peft_config)

            logger.info("==================================================")
            logger.info("Trainable Parameter Summary For Fine-Tuning:")
            logger.info(f"Fine-tuning with adapter: {self.config_obj.adapter.type}")
            self.model.print_trainable_parameters()
            logger.info("==================================================")

    def to_device(self, device):
        device = torch.device(device)

        if device == self.curr_device:
            self.initialize_adapter()
            return self
        else:
            log_once(f"Moving LLM from '{self.curr_device}' to '{device}'.")

        model_kwargs = {}
        num_gpus = torch.cuda.device_count()
        if device == torch.device("cuda") and num_gpus > 1:
            # TODO: make this configurable in the future. These parameters are from FastChat:
            # https://github.com/lm-sys/FastChat/blob/0e958b852a14f4bef5f0e9d7a5e7373477329cf2/fastchat/serve/inference.py#L90  # noqa
            # TODO: Wrap device_map="auto" in a try-except block since it may not be supported for all models (E.g. BertLMHead)  # noqa
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

            self.initialize_adapter()
        else:
            self.initialize_adapter()
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
        """Produces logits tensor for finetuning the model.

        Args:
            inputs: Inputs to the model. Can be a dictionary of input names to
                input tensors or a tuple of (inputs, targets) where inputs is
                a dictionary of input names to input tensors and targets is a
                dictionary of target names to target tensors.
            mask: A mask for the inputs.

        Returns:
            A dictionary of output {feature name}::{tensor_name} -> output tensor.
        """

        input_ids, target_ids = self._unpack_inputs(inputs)

        # Generate merged input_id, target_id pairs for the model, and create corresponding attention masks
        model_inputs, attention_masks = self._generate_merged_ids(input_ids, target_ids)

        # Wrap with flash attention backend for faster generation
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ) if self.model.cuda() else contextlib.nullcontext():
            # Forward pass using PEFT wrapped model for fine-tuning
            model_outputs = self.model(input_ids=model_inputs, attention_mask=attention_masks).get(LOGITS)

        if self.output_feature_type != TEXT:
            # Pass generated tokens through decoder after averaging the token probabilities
            # This is required for the classification head for the classifier decoder
            model_outputs = torch.mean(model_outputs, dim=1)

        if self.output_feature_type == TEXT:
            decoder_outputs = model_outputs
        else:
            decoder_outputs = self.output_feature_decoder.decoder_obj(model_outputs)

        # Set the output feature tensor to the decoder outputs (logits)
        outputs = {}
        of_name = self.config_obj.output_features[0].name
        set_output_feature_tensor(outputs, of_name, LOGITS, decoder_outputs)

        # Get predictions, probabilities and logits tensor from the output feature's predictions function
        outputs = self.output_features.get(of_name).predictions(outputs, of_name)

        # Cast to float32 for metric computation incase we're using deespeed with
        # reduced precision such as bfloat16.
        for prediction_key, prediction_tensor in outputs.items():
            outputs[prediction_key] = prediction_tensor.type(torch.float32)

        return outputs

    def generate(
        self,
        inputs: Union[
            Dict[str, torch.Tensor], Dict[str, np.ndarray], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ],
        mask=None,
    ) -> Dict[str, torch.Tensor]:
        """Generates tokens using the model."""

        input_ids, _ = self._unpack_inputs(inputs)

        with torch.no_grad():
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

                # Wrap with flash attention backend for faster generation
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True, enable_math=False, enable_mem_efficient=False
                ) if self.model.cuda() else contextlib.nullcontext():
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

        return outputs

    def _unpack_inputs(
        self,
        inputs: Union[
            Dict[str, torch.Tensor], Dict[str, np.ndarray], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Converts input tensors to input ids."""
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

        input_ids = self.get_input_ids(inputs)
        target_ids = self.get_target_ids(targets) if targets else None

        return input_ids, target_ids

    def get_input_ids(
        self,
        inputs: Union[
            Dict[str, torch.Tensor], Dict[str, np.ndarray], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ],
    ) -> torch.Tensor:
        """Returns the input ids for the text feature input."""
        return inputs[self.config_obj.input_features[0].name].type(torch.int32)

    def get_target_ids(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns the output ids for the text feature output."""
        return outputs[self.config_obj.output_features[0].name].type(torch.int32)

    def update_metrics(self, targets, predictions):
        """Updates the model's metrics given targets and predictions."""
        for of_name, of_obj in self.output_features.items():
            if isinstance(of_obj, TextOutputFeature):
                # Align the target length with the predictions length to enable text metric evaluation.
                _targets, _predictions = realign_target_and_prediction_tensors(
                    targets, predictions, of_name, self.tokenizer
                )
                of_obj.update_metrics(_targets[of_name], _predictions[of_name])
                continue
            of_obj.update_metrics(targets[of_name], predictions[of_name])

        eval_loss, additional_losses = self.eval_loss(targets, predictions)
        self.eval_loss_metric.update(eval_loss)
        self.eval_additional_losses_metrics.update(additional_losses)

    def train_loss(
        self,
        targets,
        predictions,
        regularization_type: Optional[str] = None,
        regularization_lambda: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes the training loss for the model.

        Args:
            targets: A dictionary of target names to target tensors.
            predictions: A dictionary of output names to output tensors.
            regularization_type: One of 'l1', 'l2', 'l1_l2', or None.
            regularization_lambda: The regularization lambda.

        Returns:
            A tuple of the loss tensor and a dictionary of loss for every
            output feature.
        """
        train_loss = 0
        of_train_losses = {}
        for of_name, of_obj in self.output_features.items():
            _targets, _predictions = targets, predictions
            if isinstance(of_obj, TextOutputFeature):
                # Align the target length with the predictions length to enable text metric evaluation.
                _predictions = {of_name: _predictions}

                # Remove left padding from target tensors since we also do this for the model's forward pass when we
                # concatenate the input_ids with the target_ids
                targets_without_padding = []
                lengths = []
                for target in _targets[of_name]:
                    target = self._remove_left_padding(target)
                    targets_without_padding.append(target)
                    lengths.append(target.shape[1])

                # We need all target tensors to have the same length for the loss computation. We pad the target
                # tensors with -100 since we want to negate all tokens that are not target_ids during the softmax
                # cross entropy loss computation. This ensures that the loss is computed only for the target tokens.
                max_length = max(lengths)
                for i, target in enumerate(targets_without_padding):
                    targets_without_padding[i] = self._add_left_padding(targets_without_padding[i][0], max_length, -100)
                _targets[of_name] = torch.stack(targets_without_padding).to(
                    dtype=_targets[of_name].dtype, device=_targets[of_name].device
                )

                # Re-align target tensors without padding to have equal length before realigning with the prediction
                # tensors. Padding left with -100 to match the length of the target tensor masks the input ids during
                # softmax cross entropy loss computation. This ensures that the loss is computed only for the target
                # token IDs. Examples:
                # BERTLMHead: https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/bert/modeling_bert.py#L1216-L1219 # noqa
                # GPTNeoForCausalLM: https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L736 # noqa
                _targets, _predictions = realign_target_and_prediction_tensors(
                    _targets, _predictions, of_name, self.tokenizer, "left", -100
                )

            # TODO(Arnav): Seems like doing this again and going between these format types in unnecessary, but
            # refactor so that we don't have to do this at a later point.
            predictions = {}
            for key, _ in _predictions[of_name].items():
                set_output_feature_tensor(predictions, of_name, key, _predictions[of_name][key])
            _predictions = predictions

            # TODO(Arnav): Verify if this works for category output features during fine-tuning if that is something
            # we want to support.
            # Compute output feature train loss
            of_train_loss = of_obj.train_loss(_targets[of_name], _predictions, of_name)
            train_loss += of_obj.loss.weight * of_train_loss
            of_train_losses[of_name] = of_train_loss

        additional_losses = self.losses()
        if additional_losses:
            train_loss += torch.sum(torch.stack(additional_losses))  # other losses

        # Add regularization loss
        if regularization_type is not None and regularization_lambda != 0:
            train_loss += reg_loss(self, regularization_type, l1=regularization_lambda, l2=regularization_lambda)

        return train_loss, of_train_losses

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
                _targets, _predictions = realign_target_and_prediction_tensors(
                    targets, predictions, of_name, self.tokenizer
                )
                of_eval_loss = of_obj.eval_loss(_targets[of_name], _predictions[of_name])
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
            # TODO(travis): this will need to change when we support multiple output features
            predictions[of_name] = outputs
        return predictions

    def save(self, save_path):
        """Saves the model to the given path."""
        if self.config_obj.adapter:
            weights_save_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
            self.model.save_pretrained(weights_save_path)
        else:
            logger.info("Skipped saving LLM without weight adjustments.")

    def load(self, save_path):
        """Loads the model from the given path."""
        if self.config_obj.adapter:
            from peft import PeftConfig, PeftModel  # noqa

            weights_save_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
            config = PeftConfig.from_pretrained(weights_save_path)
            config.inference_mode = False
            self.model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
            self.model = PeftModel.from_pretrained(self.model, weights_save_path)

    def get_args(self):
        """Returns init arguments for constructing this model."""
        return (
            self.config_obj.input_features.to_list(),
            self.config_obj.output_features.to_list(),
            self._random_seed,
        )

    def _set_pad_token(self):
        """Sets the pad token for the tokenizer if it is not already set."""
        # HACK(Arnav): gpt, gpt2 and llama tokenizers had no pad tokens.
        # These recommend using eos tokens instead
        # https://github.com/huggingface/transformers/issues/2648#issuecomment-616177044
        # https://github.com/huggingface/transformers/issues/2630#issuecomment-1290809338
        if any(
            isinstance(self.tokenizer, t)
            for t in [GPT2Tokenizer, GPT2TokenizerFast, LlamaTokenizer, LlamaTokenizerFast]
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _generate_merged_ids(self, input_ids, target_ids):
        """This function merges the input_ids and target_ids together to create a unified tensor to pass into the
        model.

        This is required for PEFT based fine-tuning. It also returns attention masks for the merged tensors.
        """

        # target_ids is None during evaluation of the validation/test sets in the training loop.
        if not torch.is_tensor(target_ids):
            # Create attention masks for the input_ids.
            attention_masks = []
            for input_id_sample in input_ids:
                attention_masks.append(self._create_attention_mask(input_id_sample))
            return input_ids, torch.stack(attention_masks)

        merged_input_and_targets = []
        lengths = []

        pad_tensor = torch.tensor([self.tokenizer.pad_token_id]).to(target_ids[0].device)

        # Merge input_ids and target_ids by concatenating them together.
        # We remove the left padding from both input_ids and target_ids before concatenating them.
        for input_id_sample, target_id_sample in zip(input_ids, target_ids):
            input_id_sample_no_padding = self._remove_left_padding(input_id_sample)[0]
            target_id_sample_no_padding = torch.cat(
                (self._remove_left_padding(target_id_sample)[0], pad_tensor), dim=-1
            )

            merged_sample_ids = torch.cat((input_id_sample_no_padding, target_id_sample_no_padding), dim=-1)

            merged_input_and_targets.append(merged_sample_ids)
            lengths.append(merged_sample_ids.shape[0])

        # Since we remove the left padding from the target_ids, the merged input_ids and target_ids
        # may not have the same lengths. We need to align them to the same length by adding left padding
        # and generate an attention mask for just the part of the input that is not padding.
        max_length = max(lengths)
        attention_masks = []
        for i, merged_sample_ids in enumerate(merged_input_and_targets):
            merged_input_and_targets[i] = self._add_left_padding(merged_sample_ids, max_length)
            attention_masks.append(self._create_attention_mask(merged_input_and_targets[i]))

        return torch.stack(merged_input_and_targets), torch.stack(attention_masks)

    def _remove_left_padding(self, input_ids_sample: torch.Tensor):
        """Removes left padding from the input_ids tensor."""
        # Remove all PAD tokens
        pad_idxs = torch.where(input_ids_sample == self.tokenizer.pad_token_id)[0]  # all PAD token locations
        if len(pad_idxs) != 0:
            pad_idx = pad_idxs[-1]  # get last PAD token location
        else:
            pad_idx = 0
        input_ids_sample_no_padding = input_ids_sample[pad_idx + 1 :]

        # Start from the first BOS token
        bos_idxs = torch.where(input_ids_sample_no_padding == self.tokenizer.bos_token_id)[0]  # all BOS token locations
        if len(bos_idxs) != 0:
            bos_idx = bos_idxs[0]  # get first BOS token location
        else:
            bos_idx = 0

        input_ids_sample_no_bos = input_ids_sample_no_padding[bos_idx:].unsqueeze(0)
        return input_ids_sample_no_bos

    def _add_left_padding(self, input_ids, max_length, pad_value=0):
        """Adds left padding to the input_ids tensor."""
        padding = torch.tensor(
            [pad_value] * (max_length - input_ids.shape[0]), dtype=torch.int32, device=input_ids.device
        )
        return torch.cat((padding, input_ids), dim=-1)

    def _create_attention_mask(self, input_ids):
        """Creates attention mask for the input_ids tensor."""
        return (input_ids != self.tokenizer.pad_token_id).float()

    def get_augmentation_pipelines(self) -> AugmentationPipelines:
        """Returns the augmentation pipeline for this model."""
        return AugmentationPipelines({})


def realign_target_and_prediction_tensors(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    of_name: str,
    tokenizer: PreTrainedTokenizer,
    pad_direction: str = "right",
    pad_value: int = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Realigns the target tensor with the predictions.

    This is necessary for text metrics that require the target and prediction
    to be of the same length.
    Args:
        targets: The target tensor.
        predictions: The prediction tensor.
        of_name: The output feature's name.
        pad_direction: The direction to pad the tensors. Can be 'left' or 'right'.
            Defaults to 'right'.

    Returns:
        The realigned target tensor.
    """
    target_length = targets.get(of_name).size()[1]
    prediction_length = predictions[of_name].get(PREDICTIONS).size()[1]

    if target_length == prediction_length:
        return targets, predictions

    if pad_direction not in {"left", "right"}:
        raise ValueError(f'pad_direction must be either "left" or "right". Got {pad_direction}.')

    if not pad_value:
        pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Align target and prediction tensors for text to text metric computation
    if target_length > prediction_length:
        # Pad the predictions.
        zeros_to_add = target_length - prediction_length

        if pad_direction == "right":
            predictions[of_name][PREDICTIONS] = F.pad(
                predictions[of_name][PREDICTIONS], (0, zeros_to_add), value=pad_value
            )
            predictions[of_name][PROBABILITIES] = F.pad(predictions[of_name][PROBABILITIES], (0, 0, 0, zeros_to_add))
            predictions[of_name][LOGITS] = F.pad(predictions[of_name][LOGITS], (0, 0, 0, zeros_to_add))
        elif pad_direction == "left":
            predictions[of_name][PREDICTIONS] = F.pad(
                predictions[of_name][PREDICTIONS], (zeros_to_add, 0), value=pad_value
            )
            predictions[of_name][PROBABILITIES] = F.pad(predictions[of_name][PROBABILITIES], (0, 0, zeros_to_add, 0))
            predictions[of_name][LOGITS] = F.pad(predictions[of_name][LOGITS], (0, 0, zeros_to_add, 0))

    else:
        if pad_direction == "right":
            targets[of_name] = F.pad(targets[of_name], (0, prediction_length - target_length), value=pad_value)
        elif pad_direction == "left":
            targets[of_name] = F.pad(targets[of_name], (prediction_length - target_length, 0), value=pad_value)

    # This is important since we operate on float16/bfloat16 tensors when using deepspeed or when
    # loading the model to GPU, and metric computation requires float32 tensors.
    predictions[of_name][PREDICTIONS] = predictions[of_name][PREDICTIONS].type(torch.float32)
    predictions[of_name][PROBABILITIES] = predictions[of_name][PROBABILITIES].type(torch.float32)
    predictions[of_name][LOGITS] = predictions[of_name][LOGITS].type(torch.float32)
    targets[of_name] = targets[of_name].type(torch.float32)

    return targets, predictions
