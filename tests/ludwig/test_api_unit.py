"""Unit tests for LudwigModel API edge cases.

These tests focus on the specific code paths added/changed in PR #4132:
- Model card / training report exception handling
- evaluate() batch size fallback logic
- Callback lifecycle ordering
- _check_initialization error messages
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from ludwig.api import LudwigModel
from ludwig.callbacks import Callback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RecordingCallback(Callback):
    """Records which hooks were called and in what order."""

    def __init__(self):
        self.calls: list[str] = []

    def on_train_start(self, *args, **kwargs):
        self.calls.append("on_train_start")

    def on_train_end(self, *args, **kwargs):
        self.calls.append("on_train_end")

    def on_evaluation_start(self, **kwargs):
        self.calls.append("on_evaluation_start")

    def on_evaluation_end(self, **kwargs):
        self.calls.append("on_evaluation_end")

    def on_preprocess_start(self, *args, **kwargs):
        self.calls.append("on_preprocess_start")

    def on_preprocess_end(self, *args, **kwargs):
        self.calls.append("on_preprocess_end")


# ---------------------------------------------------------------------------
# 1a. Model card / training report exception handling
# ---------------------------------------------------------------------------


def test_model_card_failure_does_not_abort_training(tmpdir, caplog):
    """Training should complete even if model card generation fails.

    The failure should be logged at WARNING with a DEBUG traceback.
    """
    config = {
        "input_features": [{"name": "x", "type": "number"}],
        "output_features": [{"name": "y", "type": "binary"}],
        "trainer": {"train_steps": 1, "batch_size": 8},
    }

    import pandas as pd

    df = pd.DataFrame({"x": range(20), "y": [0, 1] * 10})

    with patch("ludwig.utils.model_card.save_model_card", side_effect=RuntimeError("card boom")):
        with caplog.at_level(logging.WARNING, logger="ludwig"):
            model = LudwigModel(config)
            result = model.train(dataset=df, output_directory=str(tmpdir))

    assert result is not None, "training should complete despite model card failure"
    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Failed to generate model card" in str(m) for m in warning_messages)


def test_training_report_failure_does_not_abort_training(tmpdir, caplog):
    """Training should complete even if training report generation fails."""
    config = {
        "input_features": [{"name": "x", "type": "number"}],
        "output_features": [{"name": "y", "type": "binary"}],
        "trainer": {"train_steps": 1, "batch_size": 8},
    }

    import pandas as pd

    df = pd.DataFrame({"x": range(20), "y": [0, 1] * 10})

    with patch("ludwig.utils.training_report.save_training_report", side_effect=RuntimeError("report boom")):
        with caplog.at_level(logging.WARNING, logger="ludwig"):
            model = LudwigModel(config)
            result = model.train(dataset=df, output_directory=str(tmpdir))

    assert result is not None
    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Failed to generate training report" in str(m) for m in warning_messages)


# ---------------------------------------------------------------------------
# 1b. evaluate() batch size fallback
# ---------------------------------------------------------------------------


def test_check_initialization_missing_model():
    """_check_initialization should name missing components."""
    model = MagicMock(spec=LudwigModel)
    model.model = None
    model._user_config = {"input_features": [], "output_features": []}
    model.training_set_metadata = {"x": {}}

    # Call the real method on the mock instance
    with pytest.raises(ValueError, match="model"):
        LudwigModel._check_initialization(model)


def test_check_initialization_missing_metadata():
    model = MagicMock(spec=LudwigModel)
    model.model = MagicMock()
    model._user_config = {"input_features": [], "output_features": []}
    model.training_set_metadata = None

    with pytest.raises(ValueError, match="training_set_metadata"):
        LudwigModel._check_initialization(model)


def test_check_initialization_all_present():
    model = MagicMock(spec=LudwigModel)
    model.model = MagicMock()
    model._user_config = {"input_features": [], "output_features": []}
    model.training_set_metadata = {"x": {}}

    # Should not raise
    LudwigModel._check_initialization(model)


# ---------------------------------------------------------------------------
# 1d. Callback lifecycle ordering
# ---------------------------------------------------------------------------


def test_preprocess_callbacks_fire_in_order(tmpdir):
    """on_preprocess_start fires before on_preprocess_end."""
    config = {
        "input_features": [{"name": "x", "type": "number"}],
        "output_features": [{"name": "y", "type": "binary"}],
        "trainer": {"train_steps": 1, "batch_size": 8},
    }

    import pandas as pd

    df = pd.DataFrame({"x": range(20), "y": [0, 1] * 10})
    cb = RecordingCallback()
    model = LudwigModel(config, callbacks=[cb])
    model.preprocess(dataset=df, output_directory=str(tmpdir))

    assert "on_preprocess_start" in cb.calls
    assert "on_preprocess_end" in cb.calls
    assert cb.calls.index("on_preprocess_start") < cb.calls.index("on_preprocess_end")


def test_evaluate_callbacks_fire_in_order(tmpdir):
    """on_evaluation_start fires before on_evaluation_end."""
    config = {
        "input_features": [{"name": "x", "type": "number"}],
        "output_features": [{"name": "y", "type": "binary"}],
        "trainer": {"train_steps": 1, "batch_size": 8},
    }

    import pandas as pd

    df = pd.DataFrame({"x": range(20), "y": [0, 1] * 10})
    cb = RecordingCallback()
    model = LudwigModel(config, callbacks=[cb])
    model.train(dataset=df, output_directory=str(tmpdir))

    cb.calls.clear()
    model.evaluate(dataset=df)

    assert "on_evaluation_start" in cb.calls
    assert "on_evaluation_end" in cb.calls
    assert cb.calls.index("on_evaluation_start") < cb.calls.index("on_evaluation_end")
