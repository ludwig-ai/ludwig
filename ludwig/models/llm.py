import contextlib
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoConfig, GenerationConfig

from ludwig.accounting.used_tokens import get_used_tokens_for_llm
from ludwig.constants import IGNORE_INDEX_TOKEN_ID, LOGITS, MODEL_LLM, PREDICTIONS, TEXT, USED_TOKENS
from ludwig.features.base_feature import ModuleWrapper, OutputFeature
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.features.text_feature import TextOutputFeature
from ludwig.globals import MODEL_WEIGHTS_FILE_NAME
from ludwig.models.base import BaseModel
from ludwig.modules.training_hooks import NEFTuneHook
from ludwig.schema.features.base import BaseOutputFeatureConfig, FeatureCollection
from ludwig.schema.model_types.llm import LLMModelConfig
from ludwig.utils.augmentation_utils import AugmentationPipelines
from ludwig.utils.data_utils import clear_data_cache
from ludwig.utils.llm_quantization_utils import convert_quantized_linear_to_linear
from ludwig.utils.llm_utils import (
    add_left_padding,
    generate_merged_ids,
    get_context_len,
    get_realigned_target_and_prediction_tensors_for_inference,
    initialize_adapter,
    load_pretrained_from_config,
    pad_target_tensor_for_fine_tuning,
    remove_left_padding,
    to_device,
)
from ludwig.utils.logging_utils import log_once
from ludwig.utils.output_feature_utils import set_output_feature_tensor
from ludwig.utils.tokenizers import HFTokenizer
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
        _device=None,
        **_kwargs,
    ):
        super().__init__(random_seed=random_seed)

        self.config_obj = config_obj
        self._random_seed = random_seed

        self.model_name = self.config_obj.base_model
        self.model_config = AutoConfig.from_pretrained(self.config_obj.base_model)

        self.model = load_pretrained_from_config(self.config_obj, model_config=self.model_config)
        self.curr_device = next(self.model.parameters()).device
        logger.info("Done.")

        self.context_len = get_context_len(self.model_config)

        # TODO(Arnav): This needs be more flexible to account for RoPE Scaling
        # When merging input IDs and target IDs for LLM fine-tuning, we want to make sure that the merged tensor is
        # not longer than the global maximum sequence length. This is provided in the preprocessing config. We never
        # want to exceed the maximum possible context length so we also check for that.
        if self.config_obj.preprocessing.global_max_sequence_length:
            global_max_sequence_length = self.config_obj.preprocessing.global_max_sequence_length
            self.global_max_sequence_length = (
                global_max_sequence_length if global_max_sequence_length <= self.context_len else self.context_len
            )
        else:
            self.global_max_sequence_length = self.context_len

        # Initialize tokenizer
        self.tokenizer = HFTokenizer(self.config_obj.base_model).tokenizer

        self._set_generation_config(self.config_obj.generation.to_dict())

        # ================ Inputs ================
        try:
            self.input_features.update(self.build_inputs(input_feature_configs=self.config_obj.input_features))
        except KeyError as e:
            raise KeyError(
                f"An input feature has a name that conflicts with a class attribute of torch's ModuleDict: {e}"
            ) from e

        # This is used to store the model inputs during the forward pass when fine-tuning LLMs. This allows us to have
        # access to the joint model inputs (input_ids and target_ids) when computing metrics. In particular, the target
        # ids are needed to correctly compute next token softmax cross entropy loss.
        self.model_inputs = None

        # ================ Outputs ================
        self.output_feature_type = self.config_obj.output_features[0].type

        self.output_features.update(
            self.build_outputs(
                output_feature_configs=self.config_obj.output_features,
                # Set the input size to the model vocab size instead of the tokenizer vocab size
                # because the model has additional "head" layers that are used to predict the next
                # token in the sequence. These head layers can add additional dimensions to the
                # logits tensor, beyond the vocab_size dimension.
                input_size=self.input_shape[-1] if self.output_feature_type == TEXT else self.model_config.vocab_size,
            )
        )

        # Extract the decoder object for the forward pass
        self._output_feature_decoder = ModuleWrapper(self.output_features.items()[0][1])

        self.attention_masks = None

        clear_data_cache()

    def create_feature_dict(self) -> DictWrapper:
        return DictWrapper(LudwigFeatureDict())

    @contextlib.contextmanager
    def use_generation_config(self, generation_config_dict: Optional[Dict[str, Any]] = None):
        """Sets the generation config for the model."""
        # Save the original generation config so that we can reset it if/when we change it when self.generation gets is
        # dynamically mutated during 1-off predict calls after fine-tuning.
        original_generation_config_dict = self.generation.to_dict()
        try:
            # no-op if generation_config is None
            if generation_config_dict is not None:
                # unwrap the original generation config, update it with the new generation config
                new_generation_config_dict = {**original_generation_config_dict, **generation_config_dict}
                self._set_generation_config(new_generation_config_dict)
            yield
        finally:
            self._set_generation_config(original_generation_config_dict)

    def _set_generation_config(self, new_generation_config_dict: Dict[str, Any]):
        self.generation = GenerationConfig(**new_generation_config_dict)
        # We need to manually set the pad_token_id to the tokenizer's pad_token_id for certain models like GPT and
        # CodeLlama to avoid getting an error. This workaround can be found here:
        # (https://github.com/huggingface/transformers/issues/25353#issuecomment-1669339754)
        self.generation.pad_token_id = self.tokenizer.pad_token_id
        self.max_new_tokens = self.generation.max_new_tokens
        # max input length value copied from FastChat
        # https://github.com/lm-sys/FastChat/blob/0e958b852a14f4bef5f0e9d7a5e7373477329cf2/fastchat/serve/inference.py#L183  # noqa E501
        self.max_input_length = self.context_len - self.max_new_tokens - 8

    @property
    def output_feature_decoder(self) -> OutputFeature:
        return self._output_feature_decoder.module

    def initialize_adapter(self):
        """If an adapter config is provided, we want to wrap the model with a PEFT model for fine-tuning."""
        if self.config_obj.adapter:
            if self.config_obj.trainer.type != "finetune" and not self.config_obj.adapter.pretrained_adapter_weights:
                raise ValueError(
                    "Adapter config was provided, but trainer type is not set to `finetune`. Either set the trainer to "
                    "`finetune` or remove the adapter config."
                )

            self.model = initialize_adapter(self.model, self.config_obj)

            logger.info("==================================================")
            logger.info("Trainable Parameter Summary For Fine-Tuning")
            logger.info(f"Fine-tuning with adapter: {self.config_obj.adapter.type}")
            self.model.print_trainable_parameters()
            logger.info("==================================================")

    def prepare_for_training(self):
        # TODO: this implementation will not work if resuming from a previous checkpoint. Need to fix this.
        if self.config_obj.quantization:
            self.prepare_for_quantized_training()
        self.initialize_adapter()

    def prepare_for_quantized_training(self):
        from peft import prepare_model_for_kbit_training

        self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)

    def to_device(self, device):
        self.model, device = to_device(self.model, device, self.config_obj, self.curr_device)
        self.curr_device = device
        return self

    @classmethod
    def build_outputs(
        cls, output_feature_configs: FeatureCollection[BaseOutputFeatureConfig], input_size: int
    ) -> Dict[str, OutputFeature]:
        """Builds and returns output feature."""
        # TODO: only single task currently
        if len(output_feature_configs) > 1:
            raise ValueError("The LLM model type only supports a single output feature.")

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
        # We save them as class variables so that we can use them when realigning target and prediction tensors
        self.model_inputs, self.attention_masks = generate_merged_ids(
            input_ids, target_ids, self.tokenizer, self.global_max_sequence_length
        )

        # Wrap with flash attention backend for faster generation
        with (
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
            if (torch.cuda.is_available() and self.curr_device.type == "cuda")
            else contextlib.nullcontext()
        ):
            # TODO (jeffkinnison): Determine why the 8-bit `SCB` and `CB` matrices are deleted in the forward pass
            model_outputs = self.model(input_ids=self.model_inputs, attention_mask=self.attention_masks).get(LOGITS)

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
            if prediction_key != PREDICTIONS:
                # Skipping casting it to float32 since the predictions are tokens and they should be int64
                # (which is already the case)
                outputs[prediction_key] = prediction_tensor.type(torch.float32)

        # Add token usage.
        outputs[USED_TOKENS] = get_used_tokens_for_llm(self.model_inputs, self.tokenizer)
        return outputs

    def generate(
        self,
        inputs: Union[
            Dict[str, torch.Tensor], Dict[str, np.ndarray], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ],
        mask=None,
    ) -> Dict[str, torch.Tensor]:
        """Generates tokens using the model."""
        log_once(f"For generating text, using: {self.generation}")
        input_ids, _ = self._unpack_inputs(inputs)

        with torch.no_grad():
            input_lengths = []
            sequences_list = []
            for input_ids_sample in input_ids:
                input_ids_sample_no_padding = remove_left_padding(input_ids_sample, self.tokenizer)

                if input_ids_sample_no_padding.shape[1] > self.max_input_length:
                    logger.warning(
                        f"Input length {input_ids_sample_no_padding.shape[1]} is "
                        f"greater than max input length {self.max_input_length}. Truncating."
                    )
                    input_ids_sample_no_padding = input_ids_sample_no_padding[:, -self.max_input_length :]  # noqa E203

                input_lengths.append(input_ids_sample_no_padding.shape[1])

                # Wrap with flash attention backend for faster generation
                with (
                    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
                    if (torch.cuda.is_available() and self.curr_device.type == "cuda")
                    else contextlib.nullcontext()
                ):
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
                input_lengths,
                self.max_new_tokens,
            )

        return outputs

    def is_merge_and_unload_set(self) -> bool:
        """Check if the "adapter" configuration section exists and, if affirmative, that it contains the
        "postprocessor" subsection and the "merge_adapter_into_base_model" and "progressbar" directives.

        # Return

            :return (bool): whether merge_and_unload should be done.
        """
        return (
            self.config_obj.adapter is not None
            and self.config_obj.adapter.postprocessor is not None
            and self.config_obj.adapter.postprocessor.merge_adapter_into_base_model
        )

    def merge_and_unload(self, progressbar: bool = False) -> None:
        """This method merges the LoRa layers into the base model.  This is needed if someone wants to use the base
        model as a standalone model.  The implementation calls merge_and_unload() of the underlying LoraModel class
        (in peft).

        Args:
            progressbar (bool): whether to show a progressbar indicating the unload and merge process
        """
        from peft import LoraModel

        if isinstance(self.model.base_model, LoraModel):
            self.model.base_model.merge_and_unload(progressbar=progressbar)
        else:
            raise ValueError("This operation requires an LLM model trained with a LoRA adapter.")

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
        """Updates the model's metrics given targets and predictions for zero-shot/few-shot."""
        for of_name, of_obj in self.output_features.items():
            if isinstance(of_obj, TextOutputFeature):
                # Align the target length with the predictions length to enable text metric evaluation.
                _targets, _predictions = get_realigned_target_and_prediction_tensors_for_inference(
                    targets, predictions, of_name, self.tokenizer
                )
                of_obj.update_metrics(_targets[of_name], _predictions[of_name], self.tokenizer)
            else:
                of_obj.update_metrics(targets[of_name], predictions[of_name])

        # HACK (Tim): get the device of the targets to transfer self.eval_loss_metric to the same device
        target_device = list(targets.values())[0].device

        eval_loss, additional_losses = self.eval_loss(targets, predictions)
        self.eval_loss_metric = self.eval_loss_metric.to(target_device)
        self.eval_loss_metric.update(eval_loss)
        self.eval_additional_losses_metrics.update(additional_losses)

    def update_metrics_finetune_llm(self, targets, predictions):
        """Updates the model's metrics given targets and predictions for fine-tuning."""
        _targets, _predictions = targets, predictions
        for of_name, of_obj in self.output_features.items():
            if isinstance(of_obj, TextOutputFeature):
                # Update the target tensor to enable text metric evaluation. This pads the target tensor with -100s
                # to match the prediction length and depends on how much of the target tensor was included in the
                # forward pass.
                _targets = self._update_target_tensor_for_finetuning(_targets, _predictions, of_name)
                if isinstance(of_obj, TextOutputFeature):
                    of_obj.update_metrics(_targets[of_name], _predictions[of_name], self.tokenizer)
                else:
                    of_obj.update_metrics(_targets[of_name], _predictions[of_name])
                continue

            of_obj.update_metrics(_targets[of_name], _predictions[of_name])

        eval_loss, additional_losses = self.eval_loss(_targets, _predictions)
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
                _predictions = {of_name: _predictions}

                # Update the target tensor to enable text metric evaluation. This pads the target tensor with -100s
                # to match the prediction length and depends on how much of the target tensor was included in the
                # forward pass.
                _targets = self._update_target_tensor_for_finetuning(_targets, _predictions, of_name)

            # TODO(Arnav): Seems like doing this again and going between these format types in unnecessary, but
            # refactor so that we don't have to do this at a later point.
            predictions = {}
            for key, _ in _predictions[of_name].items():
                set_output_feature_tensor(predictions, of_name, key, _predictions[of_name][key])
            _predictions = predictions

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
                _targets, _predictions = get_realigned_target_and_prediction_tensors_for_inference(
                    targets, predictions, of_name, self.tokenizer
                )
                of_eval_loss = of_obj.eval_loss(_targets[of_name], _predictions[of_name])
            else:
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
        # TODO(travis): use the implementation of trainer itself to decide whether to save the model, to
        # avoid this hack
        if self.config_obj.trainer.type != "none":
            weights_save_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
            # We initialize the model's generation configuration; otherwise, we get a validation error.
            self.model.generation_config = self.generation
            self.model.save_pretrained(weights_save_path)
        else:
            logger.info("Skipped saving LLM without weight adjustments.")

    def save_base_model(self, save_path):
        """Saves the base LLM model to the given path."""
        # TODO: see the "TODO" statement from "LLM.save()" in this module.
        if self.config_obj.trainer.type != "none":
            weights_save_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
            self.model.base_model.save_pretrained(weights_save_path)
            # While this class initializes the tokenizer (from the base_model) automatically, and hence does not
            # need to be saved if inference is to be done using LudwigModel.predict(), the rationale for saving the
            # tokenizer to HuggingFace Hub is to provide access to models fine-tuned and persisted to HuggingFace Hub
            # using Ludwig at a later time, with the ability to perform inference, independently of Ludwig itself.
            self.tokenizer.save_pretrained(weights_save_path)
        else:
            logger.info("Skipped saving LLM without weight adjustments.")

    def save_dequantized_base_model(self, save_path: str) -> None:
        """Upscales quantized weights of a model to fp16 and saves the result in a folder specified by save_path.

        Args:
            save_path (str): The path to the folder where the upscaled model weights will be saved.

        Returns:
            None
        """
        from peft import PeftModel

        if isinstance(self.model, PeftModel):
            # Get the base model back by removing all the adapter modules without merging.
            logger.warning(
                "LLM model is currently wrapped in a PeftModel. Removing the adapter layers and saving the base model."
                "Reload the model via LudwigModel.load() to use your trained adapter layers for inference."
            )
            self.model = self.model.unload()

        # Dequantize the model weights and cast them to fp16 - replace quantized layers with appropriate
        # linear layers in-place.
        logger.info("Upscaling quantized weights to fp16...")
        convert_quantized_linear_to_linear(self.model)
        logger.info("Done.")

        # Remove the quantization configuration from the model
        # The reason we can't delete the quantization config is because it is a property of the model and
        # HF does some weird serialization of the config that causes an error when trying to access `self.model.config`
        # after you try and delete a key from the config: TypeError: Object of type dtype is not JSON serializable.
        self.model.config.quantization_config = {}

        # Override properties of the model to indicate that it is no longer quantized.
        # This is also necessary to ensure that the model can be saved, otherwise it will raise an error like
        # "You are calling `save_pretrained` on a 4-bit converted model. This is currently not supported"
        # See: https://github.com/huggingface/transformers/blob/0ad4e7e6dad670a7151aaceb1af3c272a3bf73a8/src/transformers/modeling_utils.py#L2054 # noqa
        self.model.is_loaded_in_4bit = False
        self.model.is_loaded_in_8bit = False

        # Save the model
        logger.info(f"Saving upscaled model to {save_path}")
        self.model.save_pretrained(save_path)
        logger.info("Done.")

        # Save the tokenizer
        logger.info(f"Saving tokenizer to {save_path}")
        self.tokenizer.save_pretrained(save_path)
        logger.info("Done.")

    def load(self, save_path):
        """Loads the model from the given path."""
        weights_save_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
        if self.config_obj.adapter:
            from peft import PeftModel  # noqa

            if isinstance(self.model, PeftModel):
                # Unwrap and reload PeftModel
                self.model = self.model.base_model

            self.model = PeftModel.from_pretrained(self.model, weights_save_path)
        elif self.config_obj.trainer.type != "none":
            self.model = load_pretrained_from_config(
                self.config_obj, model_config=self.model_config, weights_save_path=weights_save_path
            )
        else:
            logger.info("Skipped loading LLM without weight adjustments.")

    def get_args(self):
        """Returns init arguments for constructing this model."""
        return (
            self.config_obj.input_features.to_list(),
            self.config_obj.output_features.to_list(),
            self._random_seed,
        )

    def _update_target_tensor_for_finetuning(
        self, targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], of_name: str
    ) -> Dict[str, torch.Tensor]:
        """Update target tensor for fine-tuning.

        This method removes left padding from target tensors, adds a eos token to the end of the target tensors,
        and pads the target tensors with -100 to ensure equal length for loss computation. It then realigns the
        target tensors with the prediction tensors.

        Args:
            targets (Dict[str, torch.Tensor]): A dictionary containing the target tensors.
            predictions (Dict[str, torch.Tensor]): A dictionary containing the predicted tensors.
            of_name (str): The name of the target tensor.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the updated target tensors aligned with predictions.
        """
        # Remove left padding from target tensors since we also do this for the model's forward pass when we
        # concatenate the input_ids with the target_ids. We also need to add the pad token to the end of the
        # target tensors.
        targets_without_padding = []
        lengths = []

        eos_token_tensor = torch.tensor([self.tokenizer.eos_token_id])
        for target in targets[of_name]:
            target = remove_left_padding(target, self.tokenizer)[0]
            target = torch.cat([target, eos_token_tensor.to(device=target.device)], dim=-1).unsqueeze(0)
            targets_without_padding.append(target)
            lengths.append(target.shape[1])

        # We need all target tensors to have the same length for the loss computation. We pad the target
        # tensors with -100 since we want to negate all tokens that are not target_ids during the softmax
        # cross entropy loss computation. This ensures that the loss is computed only for the target tokens.
        max_length = max(lengths)
        for i, target in enumerate(targets_without_padding):
            targets_without_padding[i] = add_left_padding(
                targets_without_padding[i][0],
                max_length,
                IGNORE_INDEX_TOKEN_ID,
            )

        targets[of_name] = torch.stack(targets_without_padding, dim=0).to(
            dtype=targets[of_name].dtype,
            device=targets[of_name].device,
        )

        # Re-align target tensors without padding to have equal length before realigning with the prediction
        # tensors. Padding left with -100 to match the length of the target tensor masks the input ids during
        # softmax cross entropy loss computation. This ensures that the loss is computed only for the target
        # token IDs. Examples:
        # BERTLMHead: https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/bert/modeling_bert.py#L1216-L1219 # noqa
        # GPTNeoForCausalLM: https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L736 # noqa
        _targets = pad_target_tensor_for_fine_tuning(targets, predictions, self.model_inputs, of_name)

        return _targets

    def _activate_forward_hooks(self):
        """Activates/registers forward hooks for the model."""
        if not self.config_obj.model_parameters:
            return

        # Initialize forward hook handles
        if self.config_obj.model_parameters.neftune_noise_alpha:
            self._forward_hook_handles.append(
                NEFTuneHook(neftune_noise_alpha=self.config_obj.model_parameters.neftune_noise_alpha)
            )

        # Activate forward hooks iteratively
        for hook in self._forward_hook_handles:
            # Update the model with the forward hooks in place
            self.model = hook.activate_hook(self.model)

    @staticmethod
    def get_augmentation_pipelines() -> AugmentationPipelines:
        """Returns the augmentation pipeline for this model."""
        return AugmentationPipelines({})
