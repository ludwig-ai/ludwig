"""Unit tests for LudwigModel API edge cases."""

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
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
            # logging_level=WARNING so LudwigModel.__init__ doesn't override caplog to ERROR
            model = LudwigModel(config, logging_level=logging.WARNING)
            result = model.train(dataset=df, output_directory=str(tmpdir))

    assert result is not None, "training should complete despite model card failure"
    assert any("Failed to generate model card" in m for m in caplog.messages)


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
            model = LudwigModel(config, logging_level=logging.WARNING)
            result = model.train(dataset=df, output_directory=str(tmpdir))

    assert result is not None
    assert any("Failed to generate training report" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# 1b. evaluate() batch size fallback
# ---------------------------------------------------------------------------


def _make_evaluate_mock(trainer_dict: dict):
    """Return a MagicMock LudwigModel wired for evaluate() batch size tests."""
    # Don't pass spec= here — config_obj is an instance attribute set in __init__,
    # not a class-level attribute, so spec=LudwigModel would block access to it.
    model = MagicMock()
    model.callbacks = []
    model.config_obj.trainer.to_dict.return_value = trainer_dict

    dataset_mock = MagicMock()
    metadata_mock = MagicMock()
    model._preprocess_for_prediction.return_value = (dataset_mock, metadata_mock)

    predictor = MagicMock()
    predictor.batch_evaluation.return_value = ({}, MagicMock())

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=predictor)
    ctx.__exit__ = MagicMock(return_value=False)
    model.backend.create_predictor.return_value = ctx

    model.backend.df_engine.df_lib = pd
    model.model.output_features = {}
    return model, predictor


def _run_evaluate(model):
    """Call LudwigModel.evaluate() with all saving and collection disabled."""
    return LudwigModel.evaluate(
        model,
        dataset=MagicMock(),
        collect_predictions=False,
        collect_overall_stats=False,
        skip_save_unprocessed_output=True,
        skip_save_predictions=True,
        skip_save_eval_stats=True,
    )


def test_evaluate_uses_eval_batch_size_from_config():
    """Evaluate() should use eval_batch_size from trainer config when batch_size arg is None."""
    model, predictor = _make_evaluate_mock({"eval_batch_size": 64, "batch_size": 32})
    model.backend.is_coordinator.return_value = False
    predictor.batch_evaluation.return_value = ({}, {})

    _run_evaluate(model)

    model.backend.create_predictor.assert_called_once_with(model.model, batch_size=64)


def test_evaluate_falls_back_to_batch_size_when_eval_batch_size_absent():
    """Evaluate() should fall back to batch_size when eval_batch_size is not in trainer config."""
    model, predictor = _make_evaluate_mock({"batch_size": 16})
    model.backend.is_coordinator.return_value = False
    predictor.batch_evaluation.return_value = ({}, {})

    _run_evaluate(model)

    model.backend.create_predictor.assert_called_once_with(model.model, batch_size=16)


def test_evaluate_raises_when_no_batch_size_in_config():
    """Evaluate() must raise ValueError when neither batch_size nor eval_batch_size are set."""
    model, _ = _make_evaluate_mock({})  # empty trainer dict — no batch sizes

    with pytest.raises(ValueError, match="batch_size not specified"):
        _run_evaluate(model)


# ---------------------------------------------------------------------------
# 1c. forecast() boundary conditions
# ---------------------------------------------------------------------------


def _make_forecast_mock(output_features, input_features=None):
    """Return a MagicMock LudwigModel wired for forecast() boundary tests."""
    model = MagicMock(spec=LudwigModel)
    model.callbacks = []
    model.config_obj.output_features = output_features
    model.config_obj.input_features = input_features or []

    dataset_mock = MagicMock()
    model.backend.df_engine.df_lib = pd
    return model, dataset_mock


def test_forecast_raises_when_no_timeseries_input_feature():
    """Forecast() should raise ValueError when no timeseries input feature is present."""
    from unittest.mock import patch as _patch

    # No timeseries input features
    input_features = [MagicMock(type="number", preprocessing=MagicMock(window_size=5))]
    input_features[0].type = "number"  # not TIMESERIES

    model = MagicMock()
    model.callbacks = []
    model.config_obj.input_features = input_features

    df = pd.DataFrame({"x": range(5)})

    with _patch("ludwig.api.load_dataset_uris", return_value=(df, None, None, None)):
        with _patch("ludwig.api.load_dataset", return_value=df):
            with pytest.raises(ValueError, match="timeseries"):
                LudwigModel.forecast(model, dataset=df, horizon=3)


def test_forecast_returns_dataframe_for_valid_config():
    """Forecast() should return a DataFrame with the output feature column for a valid config."""
    from ludwig.constants import TIMESERIES

    window_size = 3
    df = pd.DataFrame({"x": range(window_size + 2), "y": [float(i) for i in range(window_size + 2)]})

    in_feat = MagicMock()
    in_feat.type = TIMESERIES
    in_feat.preprocessing.window_size = window_size

    out_feat = MagicMock()
    out_feat.type = TIMESERIES
    out_feat.column = "y"
    out_feat.name = "y"

    model = MagicMock()
    model.callbacks = []
    model.config_obj.input_features = [in_feat]
    model.config_obj.output_features = [out_feat]
    model.backend.is_coordinator.return_value = False

    def fake_predict(dataset, **kwargs):
        preds = pd.DataFrame({"y_predictions": [pd.Series([float(len(dataset))])]})
        return preds, None

    model.predict.side_effect = fake_predict

    horizon = 3
    with patch("ludwig.api.load_dataset_uris", return_value=(df, None, None, None)):
        with patch("ludwig.api.load_dataset", return_value=df):
            result = LudwigModel.forecast(model, dataset=df, horizon=horizon)

    assert isinstance(result, pd.DataFrame)
    assert "y" in result.columns
    assert len(result) <= horizon


# ---------------------------------------------------------------------------
# _check_initialization
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
