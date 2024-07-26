import copy
import logging
import tempfile
from typing import Dict, Optional, Tuple, TYPE_CHECKING, Union

import torch
import torch.nn.functional as F
import transformers
from bitsandbytes.nn.modules import Embedding
from packaging import version
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, TextStreamer

from ludwig.constants import IGNORE_INDEX_TOKEN_ID, LOGITS, PREDICTIONS, PROBABILITIES
from ludwig.schema.trainer import LLMTrainerConfig
from ludwig.utils.error_handling_utils import default_retry
from ludwig.utils.logging_utils import log_once
from ludwig.utils.model_utils import find_embedding_layer_with_path

if TYPE_CHECKING:
    from ludwig.schema.encoders.text_encoders import LLMEncoderConfig
    from ludwig.schema.model_types.llm import LLMModelConfig


logger = logging.getLogger(__name__)

transformers_436 = version.parse(transformers.__version__) >= version.parse("4.36.0")

FALLBACK_CONTEXT_LEN = 2048

_MODELS_WITH_DEVICE_MAP_AUTO_EXCLUSION = set()


@default_retry(tries=8)
def load_pretrained_from_config(
    config_obj: Union["LLMModelConfig", "LLMEncoderConfig"],
    model_config: Optional[AutoConfig] = None,
    weights_save_path: Optional[str] = None,
) -> PreTrainedModel:
    load_kwargs = {}
    if config_obj.quantization:
        # Apply quantization configuration at model load time
        load_kwargs["torch_dtype"] = getattr(torch, config_obj.quantization.bnb_4bit_compute_dtype)
        load_kwargs["quantization_config"] = config_obj.quantization.to_bitsandbytes()
        load_kwargs["device_map"] = "auto"

        if transformers_436:
            load_kwargs["attn_implementation"] = "eager"

    if config_obj.model_parameters:
        # Add any model specific parameters to the load kwargs
        for param_name, param_value in config_obj.model_parameters.to_dict().items():
            # Not all parameters are supported by all models, so we only add the parameter to the load kwargs
            # if it is supported by the model.
            if param_value is None:
                continue

            if hasattr(model_config, param_name):
                load_kwargs[param_name] = param_value
            else:
                logger.warning(f"Parameter {param_name} is not supported by {config_obj.base_model}. Skipping.")

    logger.info("Loading large language model...")
    pretrained_model_name_or_path = weights_save_path or config_obj.base_model
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **load_kwargs)
    return model


def to_device(
    model: PreTrainedModel,
    device: Union[str, torch.DeviceObjType],
    config_obj: "LLMModelConfig",  # noqa F821
    curr_device: torch.DeviceObjType,
) -> Tuple[PreTrainedModel, torch.DeviceObjType]:
    """Move an LLM to the requested device, accounting for sharding and adapters.

    Args:
        model: Pretrained model to put on device
        config_obj: LLM config
        curr_device: The current device that the model is on

    Returns:
        `model` moved to `device`
    """
    device = torch.device(device)

    if device.type == curr_device.type:
        log_once(f"Model already on device'{device}'.")
        return model, device
    else:
        log_once(f"Moving LLM from '{curr_device}' to '{device}'.")

    model_kwargs = {}
    num_gpus = torch.cuda.device_count()
    if device == torch.device("cuda") and num_gpus > 1:
        # TODO: make this configurable in the future. These parameters are from FastChat:
        # https://github.com/lm-sys/FastChat/blob/0e958b852a14f4bef5f0e9d7a5e7373477329cf2/fastchat/serve/inference.py#L90  # noqa
        # TODO: Wrap device_map="auto" in a try-except block since it may not be supported for all models (E.g. BertLMHead)  # noqa
        # We don't add quantization here (float16 or bfloat16) since we may not always want to quantize. We should
        # make quantization configurable in the future via the trainer config.
        model_kwargs.update(
            dict(
                low_cpu_mem_usage=True,
                max_memory={i: "13GiB" for i in range(num_gpus)},
            )
        )

        if config_obj.base_model not in _MODELS_WITH_DEVICE_MAP_AUTO_EXCLUSION:
            model_kwargs["device_map"] = "auto"

        if config_obj.quantization:
            model_kwargs["quantization_config"] = config_obj.quantization.to_bitsandbytes()

        # we save and reload the weights to ensure that they can be sharded across the GPUs using `from_pretrained`
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            if config_obj.adapter:
                model = AutoModelForCausalLM.from_pretrained(
                    config_obj.base_model,
                    **model_kwargs,
                )

                # Leave this import inline to support a minimal install of Ludwig
                from peft import PeftModel  # noqa

                model = PeftModel.from_pretrained(
                    model,
                    tmpdir,
                    torch_dtype=torch.float16,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    tmpdir,
                    **model_kwargs,
                )
    else:
        model = model.to(device)

    return model, device


def initialize_adapter(
    model: PreTrainedModel, config_obj: "LLMModelConfig", is_trainable: bool = False  # noqa F821
) -> Union["PeftModel", PreTrainedModel]:  # noqa F821
    """Wrap a pretrained model with a PEFT model for fine-tuning.

    Args:
         model: Pretrained model to fine-tune with an adapter.
         config_obj: LLM config
         is_trainable: bool indicating whether the adapter should be trainable

    Returns:
        `model` wrapped in a PEFT model if an adapter config was provided, otherwise `model`.
    """
    # Only load a PEFT model if the config specifies an adapter, otherwise return the model unaltered.
    if config_obj.adapter:
        if config_obj.adapter.pretrained_adapter_weights:
            # Load pretrained adapter weights if specified.
            logger.info(f"Using pretrained adapter weights: {config_obj.adapter.pretrained_adapter_weights}")

            # Leave this import inline to support a minimal install of Ludwig
            from peft import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PeftConfig  # noqa

            peft_config = PeftConfig.from_pretrained(config_obj.adapter.pretrained_adapter_weights)

            model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type].from_pretrained(
                model, config_obj.adapter.pretrained_adapter_weights, is_trainable=is_trainable
            )
        else:
            # Leave this import inline to support a minimal install of Ludwig
            from peft import get_peft_model, TaskType  # noqa

            # If no pretrained adapter is provided, we want to load untrained weights into the model
            peft_config = config_obj.adapter.to_config(
                task_type=TaskType.CAUSAL_LM, tokenizer_name_or_path=config_obj.base_model
            )

            model = get_peft_model(model, peft_config)

    return model


def get_context_len(model_config: AutoConfig):
    """Determines the maximum length of the context (input + output tokens) based on the provided model
    configuration.

    Args:
        model_config (AutoConfig): The model configuration object containing information about the model's properties.

    Returns:
        int: The maximum context length, which can be derived from the model configuration. If no relevant attribute
             is found, the default value of 2048 is returned.

    This function examines the provided model configuration object to identify the attribute that specifies the maximum
    context length. It checks for attributes in the following order of preference:
    1. 'max_sequence_length': If this attribute is present in the model configuration, its value is returned.
    2. 'max_position_embeddings': If 'max_sequence_length' is not found but 'max_position_embeddings' is present, its
       value is returned.
    3. 'n_positions': If neither 'max_sequence_length' nor 'max_position_embeddings' are found, and 'n_positions' is
       present, its value is returned.
    4. Default: If none of the relevant attributes are present, the function returns a default value of 2048.

    Note:
    - The maximum context length is important for defining the size of input and output sequences in a model.

    Example Usage:
    >>> config = AutoConfig.from_pretrained("bert-base-uncased")
    >>> context_len = get_context_len(config)
    >>> print(context_len)
    512
    """
    if hasattr(model_config, "max_sequence_length"):
        return model_config.max_sequence_length
    elif hasattr(model_config, "max_position_embeddings"):
        return model_config.max_position_embeddings
    elif hasattr(model_config, "n_positions"):
        return model_config.n_positions
    else:
        return FALLBACK_CONTEXT_LEN


def has_padding_token(input_tensor: torch.Tensor, tokenizer: PreTrainedTokenizer):
    """Checks if the input tensor contains any padding tokens.

    Args:
        input_tensor (torch.Tensor): The input tensor.
        tokenizer (PreTrainedTokenizer): The tokenizer used to encode the input.

    Returns:
        bool: True if the input tensor contains any padding tokens, False otherwise.

    Example:
        >>> import torch
        >>> from transformers import PreTrainedTokenizer
        >>> tokenizer = PreTrainedTokenizer.from_pretrained('bert-base-uncased')
        >>> input_sentence = "This is an example sentence."
        >>> input_ids = tokenizer.encode(input_sentence, add_special_tokens=True)
        >>> padded_input_ids = torch.nn.functional.pad(input_ids, (0, 10 - len(input_ids)))
        >>> has_padding = has_padding_token(padded_input_ids, tokenizer)
        >>> has_padding
        True
    """
    if input_tensor.dim() == 1:
        return torch.any(input_tensor == tokenizer.pad_token_id).item()
    elif input_tensor.dim() == 2:
        return torch.any(input_tensor == tokenizer.pad_token_id, dim=-1).item()
    else:
        raise ValueError("Input tensor must be 1D or 2D")


def remove_left_padding(input_ids_sample: torch.Tensor, tokenizer: PreTrainedTokenizer):
    """Removes left padding and other tokens until the first BOS token from the input_ids tensor.

    Args:
        input_ids_sample (torch.Tensor): The input tensor with padding and other tokens.
        tokenizer (PreTrainedTokenizer): The tokenizer used to encode the input.

    Returns:
        torch.Tensor: The input tensor without left padding and other tokens until the first BOS token.

    Example:
        >>> import torch
        >>> from transformers import PreTrainedTokenizer
        >>> tokenizer = PreTrainedTokenizer.from_pretrained('bert-base-uncased')
        >>> input_sentence = "This is an example sentence."
        >>> input_ids = tokenizer.encode(input_sentence, add_special_tokens=True)
        >>> padded_input_ids = torch.nn.functional.pad(input_ids, (10 - len(input_ids), 0))
        >>> input_ids_no_padding = remove_left_padding(padded_input_ids, tokenizer)
        >>> input_ids_no_padding
        tensor([[1, 2, 3]])
    """
    # Remove all PAD tokens
    pad_idxs = torch.where(input_ids_sample == tokenizer.pad_token_id)[0]  # all PAD token locations
    input_ids_no_padding = input_ids_sample
    if len(pad_idxs) != 0:
        pad_idx = pad_idxs[-1]  # get last PAD token location
        input_ids_no_padding = input_ids_sample[pad_idx + 1 :]

    # Start from the first BOS token
    bos_idxs = torch.where(input_ids_no_padding == tokenizer.bos_token_id)[0]  # all BOS token locations
    if len(bos_idxs) != 0:
        bos_idx = bos_idxs[0]  # get first BOS token location
    else:
        bos_idx = 0

    input_ids_no_bos = input_ids_no_padding[bos_idx:].unsqueeze(0)
    return input_ids_no_bos


def add_left_padding(input_ids, max_length, pad_value=0):
    """Adds left padding to the input_ids tensor.

    Args:
        input_ids (torch.Tensor): The input tensor.
        max_length (int): The maximum length of the tensor after padding.
        pad_value (int, optional): The value used for padding. Defaults to 0.

    Returns:
        torch.Tensor: The input_ids tensor with left padding.

    Example:
        >>> input_ids = torch.tensor([1, 2, 3])
        >>> max_length = 5
        >>> padded_tensor = add_left_padding(input_ids, max_length)
        >>> padded_tensor
        tensor([0, 0, 1, 2, 3])
    """
    padding = torch.tensor([pad_value] * (max_length - input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
    return torch.cat((padding, input_ids), dim=-1)


def create_attention_mask(input_ids: torch.Tensor, tokenizer: PreTrainedTokenizer):
    """Creates an attention mask for the input_ids tensor. This also sets the last padding token ID to 1 if it
    exists.

    Args:
        input_ids (torch.Tensor): The input tensor.
        tokenizer (PreTrainedTokenizer): The tokenizer used to encode the input.

    Returns:
        torch.Tensor: The attention mask tensor.

    Example:
        >>> import torch # noqa
        >>> from transformers import PreTrainedTokenizer
        >>> tokenizer = PreTrainedTokenizer.from_pretrained('bert-base-uncased')
        >>> input_sentence = "This is an example sentence."
        >>> input_ids = tokenizer.encode(input_sentence, add_special_tokens=True)
        >>> attention_mask = create_attention_mask(input_ids, tokenizer)
        >>> attention_mask
        tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    """
    attention_mask = input_ids != tokenizer.pad_token_id
    # Last token may not be padding if we've already hit the max sequence length
    if not attention_mask[-1]:
        # last token is padding, always attended to even if it is padding
        attention_mask[-1] = 1
    attention_mask = attention_mask.to(torch.int64)
    return attention_mask


def find_last_matching_index(tensor_a: torch.Tensor, tensor_b: torch.Tensor):
    """Returns the last index of `tensor_a` that matches `tensor_b`. Specifically, this checks whether the tensor_b
    is in the last tensor_b.shape[0] elements of tensor_a.

    Args:
        tensor_a (torch.Tensor): The first tensor.
        tensor_b (torch.Tensor): The second tensor.

    Returns:
        int: The last index of `tensor_a` that matches `tensor_b`. Returns -1 if there is no matching index.

    Example:
        >>> import torch
        >>> tensor_a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        >>> tensor_b = torch.tensor([6, 7, 8])
        >>> last_matching_index = find_last_matching_index(tensor_a, tensor_b)
        >>> last_matching_index
        5
    """
    last_index = -1

    tensor_a_length = tensor_a.shape[0]
    tensor_b_length = tensor_b.shape[0]

    # Get the last tensor_b_length elements of tensor_a.
    tensor_a_truncated = tensor_a[-tensor_b_length:]

    # Find the last matching index.
    for i in range(tensor_b_length):
        if torch.equal(tensor_a_truncated[i:], tensor_b[: tensor_b_length - i]):
            last_index = tensor_a_length - tensor_b_length + i
            break

    return last_index


def pad_target_tensor_for_fine_tuning(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    model_inputs: torch.Tensor,
    of_name: str,
) -> Dict[str, torch.Tensor]:
    """Pad and adjust target tensors for fine-tuning LLMS models.

    This function is used to pad and adjust the target tensors with IGNORE_INDEX_TOKEN_ID based on the model inputs and
    predictions during the fine-tuning process of Language Models. Here's what this function does:
        1. If none of the tokens from the target were in the model inputs, we create a tensor of the length of model
            inputs with value IGNORE_INDEX_TOKEN_IDs. This ignores this row from affecting loss.
        2. If the target tokens were entirely inside the model inputs, we want to pad all the tokens in model_inputs
            coming from the input with IGNORE_INDEX_TOKEN_IDs and leave the target tokens as is. This ensures that all
            of the target tokens are used during loss computation.
        3. In the scenario that only some part of the target tokens were in the model inputs, we want to pad the model
            inputs until that point and only leave the partial tokens of the target as is. This ensures that we will
            only compute loss on the target tokens that were in the model inputs.

    Args:
        targets (Dict[str, torch.Tensor]): A dictionary containing the target tensors.
        predictions (Dict[str, torch.Tensor]): A dictionary containing the predicted tensors.
        model_inputs (torch.Tensor): The input tensor passed into the model's forward pass.
        of_name (str): The name of the target tensor to be padded and adjusted.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the updated target
        dictionaries.
    """
    target_length = targets.get(of_name).size()[1]
    prediction_length = predictions[of_name].get(PREDICTIONS).size()[1]

    if target_length == prediction_length:
        return targets

    updated_targets = []
    for idx, target in enumerate(targets[of_name]):
        # Remove any leading IGNORE_INDEX_TOKEN_IDs in the target that were temporarily added for alignment
        end_index = (target != IGNORE_INDEX_TOKEN_ID).nonzero()[0]
        target = target[end_index:]
        target_device = target.device

        # See if any part of the target was in the tensor passed into the model's forward pass
        last_matching_index = find_last_matching_index(model_inputs[idx], target)

        # If the last matching index is -1, it means that the input tensor passed into the model was truncated
        # and did not contain the target tensor. In this case, we need to truncate the target tensors as well
        # and just set it to a tensor of IGNORE_INDEX_TOKEN_ID so that we don't compute loss on this target tensor.
        if last_matching_index == -1:
            updated_targets.append(torch.full((prediction_length,), IGNORE_INDEX_TOKEN_ID).to(device=target_device))

        # If the last matching index is not -1, it means that the input tensor passed into the model was not
        # truncated and contained either a part of the target tensor or the entire target tensor. In this case,
        # we need to set the target tensor to the part of the target tensor that was passed into the model while
        # also padding it to the correct length with IGNORE_INDEX_TOKEN_ID.
        else:
            padding = torch.full((last_matching_index,), IGNORE_INDEX_TOKEN_ID).to(device=target_device)
            updated_targets.append(torch.cat((padding, target), dim=-1)[:prediction_length])

    targets[of_name] = torch.stack(updated_targets).to(device=targets.get(of_name).device, dtype=torch.int64)

    return targets


def generate_merged_ids(
    input_ids: torch.tensor, target_ids: torch.tensor, tokenizer: PreTrainedTokenizer, max_sequence_length: int = None
):
    """Generate merged input and target IDs tensor.

    This function merges the input_ids and target_ids together to create a unified tensor
    to pass into the model. It also returns attention masks for the merged tensors.

    Args:
        input_ids (torch.Tensor): The input IDs tensor.
        target_ids (torch.Tensor or None): The target IDs tensor or None.
        max_sequence_length (int or None): The maximum sequence length to pad or truncate to.
        tokenizer (PreTrainedTokenizer): The tokenizer used to encode the input_ids and target_ids.

    Returns:
        torch.Tensor: The merged input and target IDs tensor.
        torch.Tensor: The attention masks for the merged tensor.
    """
    merged_input_and_targets = []
    lengths = []

    eos_tensor = torch.tensor([tokenizer.eos_token_id]).to(target_ids[0].device)

    # Merge input_ids and target_ids by concatenating them together.
    # We remove the left padding from both input_ids and target_ids before concatenating them.
    for input_id_sample, target_id_sample in zip(input_ids, target_ids):
        input_id_sample_no_padding = remove_left_padding(input_id_sample, tokenizer)[0]
        target_id_sample_no_padding = remove_left_padding(target_id_sample, tokenizer)[0]
        target_id_sample_no_padding = torch.cat((target_id_sample_no_padding, eos_tensor), dim=-1)

        merged_sample_ids = torch.cat((input_id_sample_no_padding, target_id_sample_no_padding), dim=-1)
        # If the merged tensor is longer than the maximum sequence length, we truncate it.
        if max_sequence_length and merged_sample_ids.shape[0] > max_sequence_length:
            merged_sample_ids = merged_sample_ids[:max_sequence_length]

        merged_input_and_targets.append(merged_sample_ids)
        lengths.append(merged_sample_ids.shape[0])

    # Since we remove the left padding from the target_ids, the merged input_ids and target_ids
    # may not have the same lengths. We need to align them to the same length by adding left padding
    # and generate an attention mask for just the part of the input that is not padding.
    max_length = max(lengths)
    attention_masks = []
    for i, merged_sample_ids in enumerate(merged_input_and_targets):
        merged_input_and_targets[i] = add_left_padding(merged_sample_ids, max_length)
        attention_masks.append(create_attention_mask(merged_input_and_targets[i], tokenizer))

    return torch.stack(merged_input_and_targets), torch.stack(attention_masks)


def _get_decoded_targets_and_predictions(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, Dict[str, torch.Tensor]],
    tokenizer: PreTrainedTokenizer,
    of_name: str,
):
    """Returns the decoded targets and predictions, accounting for IGNORE_INDEX_TOKEN_ID."""
    sanitized_targets = torch.where(targets[of_name] != IGNORE_INDEX_TOKEN_ID, targets[of_name], tokenizer.pad_token_id)
    sanitized_predictions = torch.where(
        predictions[of_name][PREDICTIONS] != IGNORE_INDEX_TOKEN_ID,
        predictions[of_name][PREDICTIONS],
        tokenizer.pad_token_id,
    )
    decoded_targets = tokenizer.batch_decode(sanitized_targets, skip_special_tokens=True)
    decoded_predictions = tokenizer.batch_decode(sanitized_predictions, skip_special_tokens=True)
    return decoded_targets, decoded_predictions


def get_realigned_target_and_prediction_tensors_for_inference(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, Dict[str, torch.Tensor]],
    of_name: str,
    tokenizer: PreTrainedTokenizer,
    pad_value: int = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Realigns the target tensor with the predictions.

    This is necessary for text metrics that require the target and prediction to be of the same length.

    Args:
        targets: The target tensor.
        predictions: The prediction tensor.
        of_name: The output feature's name.
        tokenizer: The HF tokenizer.
        pad_direction: The direction to pad the tensors. Can be 'left' or 'right'.
            Defaults to 'right'.

    Returns:
        Tuple of realigned (targets, decoded_targets, predictions, decoded_predictions).
        - targets is a map of feature name -> tensor of token ids.
        - predictions is a map from output feature name -> map of tensors with the following items:
            - "predictions": tensor of token ids.
            - "probabilities": tensor of probabilities.
            - "logits": tensor of logits.
    """
    target_length = targets.get(of_name).size()[1]
    prediction_length = predictions[of_name].get(PREDICTIONS).size()[1]

    if target_length == prediction_length:
        return targets, predictions

    if not pad_value:
        pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    zeros_to_add = (
        target_length - prediction_length if target_length > prediction_length else prediction_length - target_length
    )

    # We don't want to modify the original targets and predictions tensors, so we create a copy of them.
    _targets = copy.deepcopy(targets)
    _predictions = copy.deepcopy(predictions)

    # Align target and prediction tensors for text to text metric computation
    if target_length > prediction_length:
        # Pad the predictions.
        _predictions[of_name][PREDICTIONS] = F.pad(
            _predictions[of_name][PREDICTIONS], (0, zeros_to_add), value=pad_value
        ).to(torch.int64)

        _predictions[of_name][PROBABILITIES] = F.pad(_predictions[of_name][PROBABILITIES], (0, 0, 0, zeros_to_add)).to(
            torch.float32
        )

        _predictions[of_name][LOGITS] = F.pad(_predictions[of_name][LOGITS], (0, 0, 0, zeros_to_add)).to(torch.float32)
    else:
        _targets[of_name] = F.pad(_targets[of_name], (0, zeros_to_add), value=pad_value).to(torch.int64)

    return _targets, _predictions


def update_embedding_layer(model: AutoModelForCausalLM, config_obj: LLMTrainerConfig) -> AutoModelForCausalLM:
    """Updates the embedding layer of the model to use the 8-bit embedding layer from bitsandbytes.nn.modules.

    This is necessary when using 8-bit optimizers from bitsandbytes.
    See: https://github.com/TimDettmers/bitsandbytes#tldr
    """
    # If we're using an 8-bit optimizer, we need to replace the embedding layer with a custom embedding layer from
    # bnb.nn.modules.Embedding.
    if hasattr(config_obj, "optimizer") and config_obj.optimizer.is_8bit:
        embedding_layer, module_path = find_embedding_layer_with_path(model)
        if embedding_layer is None:
            raise ValueError(
                "Could not find an embedding layer in the model. This is required when using 8-bit optimizers"
                "  since a custom 8-bit embedding layer is used in place of the original embedding layer."
            )

        # Initialize the BNB embedding layer with the same parameters and weights as the original embedding layer.
        bnb_embedding = Embedding(
            num_embeddings=embedding_layer.num_embeddings,
            embedding_dim=embedding_layer.embedding_dim,
            padding_idx=embedding_layer.padding_idx,
            max_norm=embedding_layer.max_norm,
            norm_type=embedding_layer.norm_type,
            scale_grad_by_freq=embedding_layer.scale_grad_by_freq,
            sparse=embedding_layer.sparse,
            _weight=embedding_layer.weight,
            device=model.device,
        )

        # Update the model's original embedding layer to use the BNB embedding layer using the module_path
        # returned by find_embedding_layer_with_path.
        module_path = module_path.split(".")
        module = model
        for module_name in module_path[:-1]:
            module = getattr(module, module_name)
        setattr(module, module_path[-1], bnb_embedding)

        # Set the get input embeddings lambda function to return the BNB embedding layer
        model.get_input_embeddings = lambda: bnb_embedding

        logger.info("Updated the pretrained embedding layer to use the embedding layer from bitsandbytes.")

    return model


def create_text_streamer(tokenizer: PreTrainedTokenizer) -> TextStreamer:
    """Creates a TextStreamer object for streaming text to stdout during generation."""
    return TextStreamer(tokenizer=tokenizer, skip_prompt=True)
