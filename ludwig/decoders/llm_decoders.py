import json
import logging
import math
import re
from typing import Any

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import CATEGORY, LOGITS, PREDICTIONS, PROBABILITIES, TEXT
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder
from ludwig.decoders.utils import extract_generated_tokens
from ludwig.schema.decoders.llm_decoders import CategoryExtractorDecoderConfig, TextExtractorDecoderConfig
from ludwig.utils.strings_utils import get_tokenizer

logger = logging.getLogger(__name__)


class Matcher:
    """Match a decoded LLM output string against a set of labelled patterns.

    Parameters
    ----------
    match:
        Dictionary mapping category label strings to pattern definitions.
        Each definition is a dict with keys ``type`` (``"contains"``,
        ``"regex"``, or ``"json_schema"``) and ``value`` (the substring or
        regex pattern to match).

    ``__call__`` performs a greedy first-match scan and returns the first
    label whose pattern matches.
    """

    def __init__(self, match: dict[str, dict[str, Any]]):
        self.match = match

    def contains(self, decoded_input: str, value: str) -> bool:
        """Return True if *value* is a substring of *decoded_input*."""
        return value in decoded_input

    def regex(self, decoded_input: str, regex_pattern: str) -> bool:
        """Return True if *regex_pattern* matches anywhere in *decoded_input*.

        Parameters
        ----------
        decoded_input:
            The LLM-generated text to search.
        regex_pattern:
            A Python ``re``-compatible regular expression.

        Returns
        -------
        bool
            True when at least one match is found; False otherwise.
            Compilation failures are logged as warnings and treated as no-match.
        """
        matches = []
        try:
            regex = re.compile(regex_pattern)
            matches = regex.findall(decoded_input)
        except Exception:
            logger.warning(f"Regex pattern {regex_pattern} could not be compiled.")
        return len(matches) > 0

    def json_schema(self, decoded_input: str, expected_value: str) -> bool:
        """Return True if the JSON-decoded output equals *expected_value*.

        The method attempts to parse *decoded_input* as JSON.  If successful
        it compares the (string-coerced) top-level value to *expected_value*.
        Any parse error results in False so callers can fall back gracefully.

        Parameters
        ----------
        decoded_input:
            Raw text produced by the LLM (markdown code fences are stripped).
        expected_value:
            Category label string to compare against the parsed JSON value.
        """
        text = decoded_input.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text.strip())
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                parsed = (
                    parsed.get("label")
                    or parsed.get("value")
                    or parsed.get("category")
                    or next(iter(parsed.values()), None)
                )
            return str(parsed).lower() == expected_value.lower()
        except (json.JSONDecodeError, TypeError, ValueError):
            return False

    def __call__(self, decoded_input: str) -> str | None:
        """Return the first label whose pattern matches *decoded_input*, or None."""
        for label, label_def in self.match.items():
            label_def_type = label_def["type"]
            label_def_value = label_def["value"]

            if label_def_type == "contains":
                is_match = self.contains(decoded_input, label_def_value)
            elif label_def_type == "regex":
                is_match = self.regex(decoded_input, label_def_value)
            elif label_def_type == "json_schema":
                is_match = self.json_schema(decoded_input, label_def_value)
            else:
                raise ValueError(
                    f"{label_def_type} is not a valid match `type`. Ludwig "
                    "currently supports `contains`, `regex`, and `json_schema` match types."
                )

            if is_match:
                return label
        return None


class CategoryVocabularyLogitsProcessor(LogitsProcessor):
    """HuggingFace LogitsProcessor that restricts generation to valid category prefixes.

    At each decoding step the processor masks (sets to ``-inf``) every token
    whose addition to the current partial sequence cannot be a prefix of any
    known category label.  This steers greedy / beam-search decoding towards
    producing only recognisable label strings.

    Parameters
    ----------
    category_labels:
        The set of valid category label strings (lower-cased comparison).
    tokenizer:
        A HuggingFace tokenizer used to encode each label into token ids.
    eos_token_id:
        Token id of the end-of-sequence token; always kept unmasked.
    """

    def __init__(self, category_labels: list[str], tokenizer, eos_token_id: int):
        self.eos_token_id = eos_token_id
        self._label_token_ids: list[list[int]] = []
        for label in category_labels:
            ids = tokenizer.encode(label.lower(), add_special_tokens=False)
            self._label_token_ids.append(ids)

        self._prefix_cache: dict[tuple[int, ...], set[int]] = {}
        self._build_prefix_cache()

    def _build_prefix_cache(self):
        """Pre-populate the prefix cache for all label token sequences."""
        for ids in self._label_token_ids:
            for depth in range(len(ids)):
                prefix = tuple(ids[:depth])
                allowed = self._prefix_cache.setdefault(prefix, set())
                allowed.add(ids[depth])
        self._prefix_cache.setdefault((), set())

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Mask tokens that cannot extend a valid category prefix.

        Parameters
        ----------
        input_ids:
            Shape ``(batch_size, sequence_length)``.
        scores:
            Shape ``(batch_size, vocab_size)``.

        Returns
        -------
        torch.FloatTensor
            Modified *scores* with invalid token positions set to ``-inf``.
        """
        for batch_idx in range(input_ids.shape[0]):
            generated_ids = input_ids[batch_idx]
            allowed: set[int] | None = None
            for tail_len in range(len(generated_ids) + 1):
                tail = tuple(generated_ids[-tail_len:].tolist()) if tail_len > 0 else ()
                if tail in self._prefix_cache:
                    allowed = self._prefix_cache[tail]
                    break

            if allowed is not None:
                mask = torch.full((scores.shape[1],), -math.inf, device=scores.device, dtype=scores.dtype)
                for token_id in allowed:
                    if token_id < scores.shape[1]:
                        mask[token_id] = 0.0
                if self.eos_token_id is not None and self.eos_token_id < scores.shape[1]:
                    mask[self.eos_token_id] = 0.0
                scores[batch_idx] = scores[batch_idx] + mask
        return scores


@DeveloperAPI
@register_decoder("text_extractor", [TEXT])
class TextExtractorDecoder(Decoder):
    """Decoder for free-form text generation from an LLM.

    Extracts the generated token sequences from the raw LLM output and
    computes real token-level log-probabilities from the generation scores
    returned by ``model.generate(..., output_scores=True)``.

    ``LOGITS`` contains per-step vocabulary logits stacked as a tensor of
    shape ``(batch, max_new_tokens, vocab_size)``.  ``PROBABILITIES`` is the
    softmax of ``LOGITS`` over the vocabulary dimension.

    Parameters
    ----------
    input_size:
        Size of the input representation (vocabulary size of the LM head).
    decoder_config:
        An instance of ``TextExtractorDecoderConfig``.
    """

    def __init__(
        self,
        input_size: int,
        decoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = decoder_config
        self.input_size = input_size

        self.tokenizer_type = self.config.tokenizer
        self.pretrained_model_name_or_path = self.config.pretrained_model_name_or_path
        self.vocab_file = self.config.vocab_file

        self.tokenizer = get_tokenizer(self.tokenizer_type, self.vocab_file, self.pretrained_model_name_or_path)
        if hasattr(self.tokenizer, "tokenizer"):
            self.tokenizer_vocab_size = self.tokenizer.tokenizer.vocab_size
        else:
            self.tokenizer_vocab_size = len(self.tokenizer.vocab)

        # TODO(geoffrey): figure out where self.max_sequence_length is used. If not used, consider removing it.
        # It's confusing to have both this and `max_new_tokens` as a mandatory param in the `forward` function.
        self.max_sequence_length = self.config.max_new_tokens
        self.match_strategy = getattr(self.config, "match_strategy", "contains")

    @staticmethod
    def get_schema_cls():
        return TextExtractorDecoderConfig

    @property
    def input_shape(self):
        return self.input_size

    def get_prediction_set(self):
        return {LOGITS, PREDICTIONS, PROBABILITIES}

    def forward(
        self,
        inputs: list[torch.Tensor],
        input_lengths: list[int],
        max_new_tokens: int,
        generation_scores: list[tuple[torch.Tensor, ...]] | None = None,
    ):
        """Extract predictions and compute logits / probabilities.

        Parameters
        ----------
        inputs:
            List of raw generated sequence tensors (one per batch item),
            each including the prompt tokens.
        input_lengths:
            Number of prompt tokens for each batch item.
        max_new_tokens:
            Maximum number of tokens that can be generated.
        generation_scores:
            Per-sample generation scores from
            ``model.generate(..., output_scores=True)``.  Each element is a
            tuple of ``(vocab_size,)`` tensors—one per generated token.
            When None, zero-filled tensors are returned for backward compat.

        Returns
        -------
        dict
            Keys: ``PREDICTIONS``, ``PROBABILITIES``, ``LOGITS``.
        """
        generated_outputs = extract_generated_tokens(
            raw_generated_output_sequences=inputs,
            input_lengths=input_lengths,
            max_new_tokens=max_new_tokens,
            pad_sequence=True,
        )
        for output in generated_outputs:
            if output.shape[0] > max_new_tokens:
                raise ValueError(
                    f"Output {output} is longer than the max_new_tokens {max_new_tokens} during decoding. "
                    f"This should never happen– please file an issue on GitHub."
                )

        generated_outputs = torch.stack(generated_outputs, dim=0)
        outputs_device = generated_outputs.device
        batch_size = generated_outputs.shape[0]

        if generation_scores is not None:
            logits_list = []
            for sample_scores in generation_scores:
                if len(sample_scores) == 0:
                    sample_logits = torch.zeros(max_new_tokens, self.tokenizer_vocab_size, device=outputs_device)
                else:
                    stacked = torch.stack(list(sample_scores), dim=0).to(outputs_device)
                    num_generated = stacked.shape[0]
                    vocab_size = min(stacked.shape[1], self.tokenizer_vocab_size)
                    padded = torch.zeros(max_new_tokens, self.tokenizer_vocab_size, device=outputs_device)
                    padded[: min(num_generated, max_new_tokens), :vocab_size] = stacked[
                        : min(num_generated, max_new_tokens), :vocab_size
                    ]
                    sample_logits = padded
                logits_list.append(sample_logits)

            logits_tensor = torch.stack(logits_list, dim=0)
            probabilities_tensor = F.softmax(logits_tensor, dim=-1)
        else:
            logits_tensor = torch.zeros(batch_size, max_new_tokens, self.tokenizer_vocab_size, device=outputs_device)
            probabilities_tensor = torch.zeros(
                batch_size, max_new_tokens, self.tokenizer_vocab_size, device=outputs_device
            )

        return {
            PREDICTIONS: generated_outputs,
            PROBABILITIES: probabilities_tensor,
            LOGITS: logits_tensor,
        }


@DeveloperAPI
@register_decoder("category_extractor", [CATEGORY])
class CategoryExtractorDecoder(Decoder):
    """Decoder that maps LLM-generated text to a discrete category label.

    Category prediction proceeds in three stages:

    1. **Generation**: The parent LLM model generates token sequences.
    2. **Parsing**: Generated text is decoded and matched against label
       patterns using :class:`Matcher`.
    3. **Probability estimation**: When ``generation_scores`` are available,
       category probabilities are derived from token logits by mapping each
       label's first token to its score and normalising via softmax.

    Constrained decoding is available when ``constrain_to_vocabulary=True``.
    Call ``get_logits_processor()`` and pass the result to ``model.generate``.

    Parameters
    ----------
    decoder_config:
        An instance of ``CategoryExtractorDecoderConfig``.
    """

    def __init__(
        self,
        decoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = decoder_config

        self.input_size = self.config.input_size
        self.fallback_label = self.config.fallback_label
        self.str2idx = self.config.str2idx
        self.vocab_size = len(self.config.str2idx)
        self.match_strategy = getattr(self.config, "match_strategy", "contains")
        self.constrain_to_vocabulary = getattr(self.config, "constrain_to_vocabulary", False)

        self.matcher = Matcher(self.config.match)

        self.tokenizer_type = self.config.tokenizer
        self.pretrained_model_name_or_path = self.config.pretrained_model_name_or_path
        self.vocab_file = self.config.vocab_file

        self.tokenizer = get_tokenizer(self.tokenizer_type, self.vocab_file, self.pretrained_model_name_or_path)

        # Pre-compute first-token ids for logit-based probability estimation.
        self._label_first_token_ids: dict[str, int] = {}
        if hasattr(self.tokenizer, "tokenizer"):
            hf_tok = self.tokenizer.tokenizer
            for label in self.str2idx:
                ids = hf_tok.encode(label.lower(), add_special_tokens=False)
                if ids:
                    self._label_first_token_ids[label] = ids[0]

    @staticmethod
    def get_schema_cls():
        return CategoryExtractorDecoderConfig

    @property
    def input_shape(self):
        return self.input_size

    def get_prediction_set(self):
        return {LOGITS, PREDICTIONS, PROBABILITIES}

    def get_logits_processor(self) -> LogitsProcessorList | None:
        """Return a LogitsProcessorList for constrained decoding, or None.

        When ``constrain_to_vocabulary`` is True this returns a list containing
        a :class:`CategoryVocabularyLogitsProcessor` that restricts generation
        to valid category label prefixes.

        Returns
        -------
        LogitsProcessorList or None
        """
        if not self.constrain_to_vocabulary:
            return None
        if not hasattr(self.tokenizer, "tokenizer"):
            logger.warning(
                "constrain_to_vocabulary=True requires an HF tokenizer. " "Falling back to unconstrained generation."
            )
            return None

        hf_tok = self.tokenizer.tokenizer
        labels = list(self.str2idx.keys())
        processor = CategoryVocabularyLogitsProcessor(labels, hf_tok, hf_tok.eos_token_id)
        return LogitsProcessorList([processor])

    def _compute_category_logits_from_scores(
        self,
        generation_scores: tuple[torch.Tensor, ...],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute a ``(num_categories,)`` logit vector from token generation scores.

        For each known category label we look up the logit score of the
        label's first sub-word token at the first generated position.

        Parameters
        ----------
        generation_scores:
            Tuple of per-step score tensors.  Only index 0 is used.
        device:
            Target device for the output tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(num_categories,)`` with logit scores; unknown labels
            receive ``-inf``.
        """
        cat_logits = torch.full((self.vocab_size,), -math.inf, device=device, dtype=torch.float32)
        if not generation_scores or not self._label_first_token_ids:
            return cat_logits

        first_step_scores = generation_scores[0].to(device)
        for label, label_idx in self.str2idx.items():
            token_id = self._label_first_token_ids.get(label)
            if token_id is not None and token_id < first_step_scores.shape[0]:
                cat_logits[label_idx] = first_step_scores[token_id]
        return cat_logits

    def forward(
        self,
        inputs: list[torch.Tensor],
        input_lengths: list[int],
        max_new_tokens: int,
        generation_scores: list[tuple[torch.Tensor, ...]] | None = None,
    ):
        """Extract category predictions and compute probabilities.

        Parameters
        ----------
        inputs:
            List of raw generated sequence tensors (one per batch item).
        input_lengths:
            Number of prompt tokens for each batch item.
        max_new_tokens:
            Maximum number of tokens that may be generated.
        generation_scores:
            Per-sample generation scores from
            ``model.generate(..., output_scores=True)``.  Each element is a
            tuple of ``(vocab_size,)`` tensors—one per generated token.
            When None, one-hot probability vectors are used (original behaviour).

        Returns
        -------
        dict
            Keys: ``PREDICTIONS`` (int category indices),
            ``PROBABILITIES`` (``(batch, num_classes)`` float),
            ``LOGITS`` (same shape, raw scores before softmax).
        """
        generated_outputs = extract_generated_tokens(
            raw_generated_output_sequences=inputs,
            input_lengths=input_lengths,
            max_new_tokens=max_new_tokens,
            pad_sequence=False,
        )
        outputs_device = generated_outputs[0].device

        decoded_outputs = self.tokenizer.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)

        matched_labels = []
        probabilities = []
        logits = []

        for sample_idx, output in enumerate(decoded_outputs):
            output = output.lower()

            matched_label = self.matcher(output)
            idx = self.str2idx[matched_label] if matched_label in self.str2idx else self.str2idx[self.fallback_label]
            matched_labels.append(idx)

            if generation_scores is not None and sample_idx < len(generation_scores):
                cat_logits = self._compute_category_logits_from_scores(generation_scores[sample_idx], outputs_device)
            else:
                # Fall back to one-hot hard assignment (original behaviour).
                cat_logits = torch.full((self.vocab_size,), -math.inf, device=outputs_device, dtype=torch.float32)
                cat_logits[idx] = 0.0

            logits.append(cat_logits)

            # Replace -inf with a large negative number so softmax gives 0 prob.
            finite_logits = cat_logits.clone()
            finite_logits[finite_logits == -math.inf] = -1e9
            probs = F.softmax(finite_logits, dim=0)
            probabilities.append(probs)

        return {
            PREDICTIONS: torch.tensor(matched_labels, device=outputs_device),
            PROBABILITIES: torch.stack(probabilities, dim=0),
            LOGITS: torch.stack(logits, dim=0),
        }
