"""Regression tests for top-level transformers import safety.

When torchao and PyTorch are version-mismatched, transformers' lazy loader
raises ModuleNotFoundError for classes defined in modeling_utils.py
(notably PreTrainedModel). Ludwig modules that are imported at startup
must not have top-level imports of those classes, or `from ludwig.api
import LudwigModel` will crash before the user can do anything.

These tests simulate that broken environment and verify the imports survive.
See: https://github.com/ludwig-ai/ludwig/issues/4142
"""

import sys

import pytest


def _patch_transformers_pretrained_model_broken():
    """Monkey-patch the transformers module so __getattr__('PreTrainedModel') raises.

    This reproduces the failure mode where torchao calls
    torch.utils._pytree.register_constant (added in PyTorch 2.5+) at import
    time, causing transformers' lazy loader for modeling_utils.py classes to
    raise ModuleNotFoundError.
    """
    import transformers

    original_class = transformers.__class__

    class BrokenTransformers(original_class):
        def __getattr__(self, name):
            if name == "PreTrainedModel":
                raise ModuleNotFoundError(
                    "Could not import module 'PreTrainedModel'. Are this object's requirements defined correctly?"
                )
            return super().__getattr__(name)

    transformers.__class__ = BrokenTransformers
    return transformers, original_class


def _restore_transformers(transformers_module, original_class):
    transformers_module.__class__ = original_class


def _evict_ludwig_modules(*substrings):
    """Remove cached Ludwig modules containing any of the given substrings."""
    to_del = [k for k in sys.modules if any(s in k for s in substrings)]
    for k in to_del:
        del sys.modules[k]


@pytest.fixture()
def broken_pretrained_model():
    """Fixture that makes transformers.PreTrainedModel unavailable for the test."""

    transformers_module, original_class = _patch_transformers_pretrained_model_broken()
    _evict_ludwig_modules("llm_utils", "text_feature", "hf_utils")
    yield
    _restore_transformers(transformers_module, original_class)
    _evict_ludwig_modules("llm_utils", "text_feature", "hf_utils")


def test_llm_utils_imports_without_pretrained_model(broken_pretrained_model):
    """ludwig.utils.llm_utils must be importable when PreTrainedModel is broken.

    Regression: PreTrainedModel was imported at module level in llm_utils.py.
    Fix: moved to TYPE_CHECKING + from __future__ import annotations.
    """
    import ludwig.utils.llm_utils  # must not raise  # noqa: F401


def test_text_feature_imports_without_pretrained_model(broken_pretrained_model):
    """ludwig.features.text_feature must be importable when PreTrainedModel is broken.

    Regression: PreTrainedTokenizer (which shares the same broken lazy-loader
    path) was imported at module level in text_feature.py.
    Fix: moved to TYPE_CHECKING + from __future__ import annotations.
    """
    import ludwig.features.text_feature  # must not raise  # noqa: F401


def test_ludwig_api_imports_without_pretrained_model(broken_pretrained_model):
    """from ludwig.api import LudwigModel must succeed when PreTrainedModel is broken.

    This is the exact import that the end-user runs and that was failing.
    It exercises the full startup import chain:
      ludwig.api → ludwig.backend → ... → ludwig.encoders.text_encoders
                 → ludwig.utils.llm_utils → transformers.PreTrainedModel (broken)
    """
    _evict_ludwig_modules("ludwig")
    import ludwig.utils.llm_utils  # noqa: F401
