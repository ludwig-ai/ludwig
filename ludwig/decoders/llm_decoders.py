import logging
import re
from typing import Any, Dict, Union

import torch

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import CATEGORY
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder

# from ludwig.schema.features.preprocessing.category import CategoryOutputPreprocessingConfig
from ludwig.schema.decoders.llm_decoders import ParserDecoderConfig
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
@register_decoder("parser", [CATEGORY])
class ParserDecoder(Decoder):
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
        return ParserDecoderConfig

    @property
    def input_shape(self):
        return self.input_size

    def forward(self, inputs, **kwargs):
        # Decode inputs from the text input feature's generate function.
        decoded_outputs = self.tokenizer.tokenizer.batch_decode(inputs, skip_special_tokens=True)
        print(f"Decoded output: {decoded_outputs}")

        # Parse labels based on matching criteria
        # TODO: Figure out why neutral (ID 1) is mapping to negative (ID 2)
        # Related to how str2idx/idx2str is calculated in the metadata during preprocessing,
        # see create_vocabulary_single_token
        matched_labels = []
        for output in decoded_outputs:
            matched_label = self.matcher(output)
            if matched_label in self.str2idx:
                matched_labels.append(self.str2idx[matched_label])
            else:
                matched_labels.append(self.str2idx[self.fallback_label])

        return torch.tensor(matched_labels)
