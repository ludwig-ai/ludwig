"""Tests for MultimodalCollator."""

from unittest.mock import MagicMock

import pytest
import torch

from ludwig.data.multimodal_collator import MultimodalCollator


def _make_processor(output_key="input_ids"):
    """Build a mock processor that returns a dict with one tensor under output_key."""

    def call(text, images, return_tensors, padding, **kwargs):
        batch_size = len(text)
        result = {output_key: torch.zeros(batch_size, 10, dtype=torch.long)}
        return result

    proc = MagicMock(side_effect=call)
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    def tokenize(texts, return_tensors, padding, **kwargs):
        return {"input_ids": torch.ones(len(texts), 5, dtype=torch.long)}

    tokenizer.side_effect = tokenize
    proc.tokenizer = tokenizer
    return proc


def _make_examples(n: int = 3, include_labels: bool = False) -> list[dict]:
    examples = []
    for i in range(n):
        ex = {"image": f"img_{i}.jpg", "text": f"text {i}"}
        if include_labels:
            ex["labels"] = f"label {i}"
        examples.append(ex)
    return examples


class TestMultimodalCollatorBasic:
    def test_returns_dict(self):
        proc = _make_processor()
        collator = MultimodalCollator(proc)
        result = collator(_make_examples(2))
        assert isinstance(result, dict)

    def test_calls_processor_with_texts_and_images(self):
        proc = _make_processor()
        collator = MultimodalCollator(proc)
        examples = _make_examples(3)
        collator(examples)
        call_kwargs = proc.call_args[1]
        assert call_kwargs["text"] == ["text 0", "text 1", "text 2"]
        assert call_kwargs["images"] == ["img_0.jpg", "img_1.jpg", "img_2.jpg"]

    def test_no_labels_by_default(self):
        proc = _make_processor()
        collator = MultimodalCollator(proc)
        result = collator(_make_examples(2, include_labels=False))
        assert "labels" not in result

    def test_max_length_passed_to_processor(self):
        proc = _make_processor()
        collator = MultimodalCollator(proc, max_length=64)
        collator(_make_examples(2))
        call_kwargs = proc.call_args[1]
        assert call_kwargs["max_length"] == 64
        assert call_kwargs["truncation"] is True

    def test_custom_keys(self):
        proc = _make_processor()
        collator = MultimodalCollator(proc, image_key="img", text_key="caption")
        examples = [{"img": "a.jpg", "caption": "hello"}, {"img": "b.jpg", "caption": "world"}]
        collator(examples)
        call_kwargs = proc.call_args[1]
        assert call_kwargs["images"] == ["a.jpg", "b.jpg"]
        assert call_kwargs["text"] == ["hello", "world"]


class TestMultimodalCollatorLabels:
    def test_labels_added_when_present(self):
        proc = _make_processor()
        collator = MultimodalCollator(proc)
        result = collator(_make_examples(2, include_labels=True))
        assert "labels" in result

    def test_labels_shape_matches_tokenizer_output(self):
        proc = _make_processor()
        collator = MultimodalCollator(proc)
        result = collator(_make_examples(3, include_labels=True))
        # tokenizer mock returns shape (n, 5)
        assert result["labels"].shape == (3, 5)

    def test_mixed_labels_raises_value_error(self):
        """Some examples have labels, some don't → should raise ValueError."""
        proc = _make_processor()
        collator = MultimodalCollator(proc)
        examples = _make_examples(3, include_labels=True)
        examples[1].pop("labels")  # remove one label
        with pytest.raises(ValueError, match="missing"):
            collator(examples)

    def test_labels_pad_replaced_with_minus100(self):
        proc = _make_processor()
        # Tokenizer returns 0 for padding; collator should replace with -100
        collator = MultimodalCollator(proc)
        result = collator(_make_examples(2, include_labels=True))
        # All tokens are 1 (from mock), none replaced → no -100 in this case
        assert (result["labels"] != 0).all()

    def test_device_resolution_without_input_ids_key(self):
        """Collator must not KeyError when processor emits pixel_values instead of input_ids."""
        proc = _make_processor(output_key="pixel_values")
        collator = MultimodalCollator(proc)
        # Should complete without raising KeyError
        result = collator(_make_examples(2, include_labels=True))
        assert "labels" in result

    def test_no_tokenizer_raises_value_error(self):
        proc = _make_processor()
        del proc.tokenizer  # remove tokenizer attribute
        collator = MultimodalCollator(proc)
        with pytest.raises((ValueError, AttributeError)):
            collator(_make_examples(2, include_labels=True))
