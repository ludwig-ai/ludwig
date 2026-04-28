"""Multimodal data collator for vision-language fine-tuning.

Bridges Ludwig's tabular-style dataset (one row = one example) with HuggingFace's multimodal
``AutoProcessor`` interface used by Qwen2-VL / LLaVA / InternVL.  Each processor accepts
text + images and emits a single ``BatchFeature`` suitable for ``AutoModelForVision2Seq``.

The collator is intentionally thin — Ludwig already does tokenization and image preprocessing
through feature-level encoders, so at collate time we only need to stack tensors into the
shape the VLM expects.  For end-to-end VLM training that keeps the ``AutoProcessor`` as the
single source of truth for tokenization, pass ``use_processor=True`` and the raw columns
(``images`` as a list of PIL Images / paths, ``text`` as strings) — the collator then calls
``processor(text=..., images=..., return_tensors="pt")``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class MultimodalCollator:
    """Collate image+text batches for a HuggingFace VLM.

    Args:
        processor: an ``AutoProcessor`` instance (e.g. ``Qwen2VLProcessor``). The collator
            calls the processor with the raw ``text`` and ``images`` columns pulled from the
            incoming dict of per-example records.
        image_key: column name in each example dict that holds the image (PIL / path / bytes).
        text_key: column name that holds the text / prompt.
        label_key: column name holding the target text (for fine-tuning).  When present, it
            is tokenised by the processor's tokenizer and placed under ``labels`` in the
            returned batch with proper -100 masking on prompt tokens.
        max_length: optional max token length for truncation of text / labels.
    """

    def __init__(
        self,
        processor: Any,
        *,
        image_key: str = "image",
        text_key: str = "text",
        label_key: str = "labels",
        max_length: int | None = None,
    ) -> None:
        self.processor = processor
        self.image_key = image_key
        self.text_key = text_key
        self.label_key = label_key
        self.max_length = max_length

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        images = [ex[self.image_key] for ex in examples]
        texts = [ex[self.text_key] for ex in examples]
        labels = [ex.get(self.label_key) for ex in examples]

        kwargs = {"text": texts, "images": images, "return_tensors": "pt", "padding": True}
        if self.max_length is not None:
            kwargs["truncation"] = True
            kwargs["max_length"] = self.max_length
        batch = self.processor(**kwargs)

        # Fine-tuning path: turn the label strings into token ids with -100 masking on prompt tokens.
        n_labels = sum(1 for label in labels if label is not None)
        if n_labels > 0:
            if n_labels != len(labels):
                missing = [i for i, label in enumerate(labels) if label is None]
                raise ValueError(
                    f"MultimodalCollator: {len(missing)} of {len(labels)} examples are missing "
                    f"'{self.label_key}' (indices {missing}). Provide labels for all examples "
                    "in the batch or none at all."
                )
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is None:
                raise ValueError("MultimodalCollator: processor has no .tokenizer; cannot produce labels")
            label_ids = tokenizer(
                labels,
                return_tensors="pt",
                padding=True,
                truncation=self.max_length is not None,
                max_length=self.max_length,
            )["input_ids"]
            # Replace pad tokens with -100 so the loss skips them.
            pad_id = tokenizer.pad_token_id
            if pad_id is not None:
                label_ids = label_ids.masked_fill(label_ids == pad_id, -100)
            batch["labels"] = label_ids.to(batch["input_ids"].device)

        return batch
