import logging
import re
from typing import Any, Dict, Union

import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import CATEGORY, TEXT
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder
from ludwig.schema.decoders.llm_decoders import CategoryParserDecoderConfig, TextParserDecoderConfig
from ludwig.utils.strings_utils import get_tokenizer

logger = logging.getLogger(__name__)


# TODO(Arnav): Refactor to split into strategies like splitters
class Matcher:
    def __init__(self, match: Dict[str, Dict[str, Any]]):
        self.match = match

    def contains(self, decoded_input: str, value: str) -> bool:
        return value in decoded_input

    def regex(self, decoded_input: str, regex_pattern: str) -> bool:
        """Perform a regex match on a given text using a specified regex pattern.

        Parameters:
        text (str): The text to perform the match on.
        regex_pattern (str): The regex pattern to use for the match.

        Returns:
        A list of match objects.
        """
        # Compile the regex pattern
        matches = []
        try:
            regex = re.compile(regex_pattern)
            # Perform the match
            matches = regex.findall(decoded_input)
        except Exception:
            logger.warning(f"Regex pattern {regex_pattern} could not be compiled.")
        # If there is a match, matches is a non-empty list, so we can use this
        # to infer if there was a match or not and return a bool
        return len(matches) > 0

    def __call__(self, decoded_input: str) -> Union[str, None]:
        # Greedy match on first label that matches the input
        for label, label_def in self.match.items():
            label_def_type = label_def["type"]
            label_def_value = label_def["value"]

            if label_def_type == "contains":
                is_match = self.contains(decoded_input, label_def_value)
            elif label_def_type == "regex":
                is_match = self.regex(decoded_input, label_def_value)
            else:
                raise ValueError(
                    f"{label_def_type} is not a valid match `type`. Ludwig "
                    "currently supports `contains` and `regex` match types."
                )

            if is_match:
                return label
        return None


@DeveloperAPI
@register_decoder("text_parser", [TEXT])
class TextParserDecoder(Decoder):
    def __init__(
        self,
        input_size: int,
        tokenizer: str,
        pretrained_model_name_or_path: str,
        vocab_file: str,
        max_new_tokens: int,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size

        # Tokenizer
        self.tokenizer_type = tokenizer
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.vocab_file = vocab_file

        # Load tokenizer required for decoding the output from the generate
        # function of the text input feature for LLMs.
        self.tokenizer = get_tokenizer(self.tokenizer_type, self.vocab_file, self.pretrained_model_name_or_path)
        self.tokenizer_vocab_size = self.tokenizer.tokenizer.vocab_size

        # Maximum number of new tokens that will be generated
        self.max_new_tokens = max_new_tokens

    @staticmethod
    def get_schema_cls():
        return TextParserDecoderConfig

    @property
    def input_shape(self):
        return self.input_size

    def forward(self, inputs, **kwargs):
        # Extract the sequences tensor from the LLMs forward pass
        raw_generated_output_sequences = inputs.sequences
        # Get the input sequence passed into the forward pass of the LLM model
        llm_model_inputs = kwargs.get("llm_model_inputs", None)

        # Remove the input sequence from the generated output sequence(s)
        if raw_generated_output_sequences.size()[0] == 1:
            generated_outputs = raw_generated_output_sequences[:, llm_model_inputs.size()[1] :]
        else:
            generated_outputs = []
            input_ids_lens = [input_ids.size()[0] for input_ids in raw_generated_output_sequences]
            for idx, input_id_len in enumerate(input_ids_lens):
                # Remove the input sequence from the generated sequence
                generated_sequence = raw_generated_output_sequences[idx][input_id_len:]
                # Pad the sequence if it is shorter than the max_new_tokens for downstream metric computation
                if generated_sequence.size()[0] < self.max_new_tokens:
                    generated_sequence = torch.nn.functional.pad(
                        generated_sequence, (0, self.max_new_tokens - generated_sequence.size()[0]), "constant", 0
                    )
                generated_outputs.append(generated_sequence)
            # Stack the predictions for each example in the batch
            generated_outputs = torch.stack(generated_outputs, dim=0)

        return {
            "predictions": generated_outputs,
            # TODO(Arnav): Add support for probabilities and logits
            "probabilities": torch.zeros((len(generated_outputs), self.max_new_tokens, self.tokenizer_vocab_size)),
            "logits": torch.zeros((len(generated_outputs), self.max_new_tokens, self.tokenizer_vocab_size)),
        }


@DeveloperAPI
@register_decoder("category_parser", [CATEGORY])
class CategoryParserDecoder(Decoder):
    def __init__(
        self,
        input_size: int,
        match: Dict[str, Dict[str, Any]],
        tokenizer: str,
        pretrained_model_name_or_path: str,
        vocab_file: str,
        str2idx: Dict[str, int],
        fallback_label: str,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.fallback_label = fallback_label
        self.str2idx = str2idx
        self.num_labels = len(str2idx)

        # Create Matcher object to perform matching on the decoded output
        self.matcher = Matcher(match)

        # Tokenizer
        self.tokenizer_type = tokenizer
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.vocab_file = vocab_file

        # Load tokenizer required for decoding the output from the generate
        # function of the text input feature for LLMs.
        self.tokenizer = get_tokenizer(self.tokenizer_type, self.vocab_file, self.pretrained_model_name_or_path)

    @staticmethod
    def get_schema_cls():
        return CategoryParserDecoderConfig

    @property
    def input_shape(self):
        return self.input_size

    def forward(self, inputs, **kwargs):
        # Extract the sequences tensor from the LLMs forward pass
        raw_generated_output_sequences = inputs.sequences
        # Get the input sequence passed into the forward pass of the LLM model
        llm_model_inputs = kwargs.get("llm_model_inputs", None)

        # Remove the input sequence from the generated output sequence(s)
        if raw_generated_output_sequences.size()[0] == 1:
            generated_outputs = raw_generated_output_sequences[:, llm_model_inputs.size()[1] :]
        else:
            generated_outputs = []
            input_ids_lens = [input_ids.size()[0] for input_ids in raw_generated_output_sequences]
            for idx, input_id_len in enumerate(input_ids_lens):
                # Remove the input sequence from the generated sequence
                generated_sequence = raw_generated_output_sequences[idx][input_id_len:]
                generated_outputs.append(generated_sequence)
            # Stack the predictions for each example in the batch
            generated_outputs = torch.stack(generated_outputs, dim=0)

        # Decode generated outputs from the LLM's generate function.
        decoded_outputs = self.tokenizer.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
        print(f"Decoded output: {decoded_outputs}")

        # Parse labels based on matching criteria and return probability vectors
        matched_labels = []
        probabilities = []
        logits = []
        for output in decoded_outputs:
            matched_label = self.matcher(output)
            idx = self.str2idx[matched_label] if matched_label in self.str2idx else self.str2idx[self.fallback_label]

            # Append the index of the matched label
            matched_labels.append(idx)

            # Append the probability vector for the matched label
            probability_vec = [0] * self.num_labels
            probability_vec[idx] = 1
            probabilities.append(probability_vec)

            # TODO(Arnav): Figure out how to compute logits. For now, we return
            # a tensor of zeros.
            logits.append([0] * self.num_labels)

        return {
            "predictions": torch.tensor(matched_labels),
            "probabilities": torch.tensor(probabilities, dtype=torch.float32),
            "logits": torch.tensor(logits, dtype=torch.float32),
        }
