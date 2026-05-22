"""StudioCallback — built-in Ludwig callback for Ludwig Studio integration.

Writes newline-delimited JSON to ``<output_dir>/metrics.jsonl`` so that the
Ludwig Studio backend can pick up events incrementally and persist them to its
database / broadcast via WebSocket.

Usage::

    from ludwig.callbacks.studio import StudioCallback

    model = LudwigModel(config, callbacks=[StudioCallback(run_id="my-run", output_dir="/tmp/run")])
    model.train(dataset=df)

Event schema
------------
Each line is a JSON object with a ``"type"`` discriminator field.

Metric event (emitted on every epoch end):

.. code-block:: json

    {
        "type": "metric",
        "run_id": "...",
        "epoch": 3,
        "step": 1500,
        "split": "train",
        "feature": "label",
        "metric": "accuracy",
        "value": 0.93,
        "progress_pct": 0.6,
        "eta_seconds": 240.0,
        "timestamp": 1718000000.0
    }

Phase event (emitted at pipeline phase transitions):

.. code-block:: json

    {
        "type": "phase",
        "run_id": "...",
        "phase": "training",
        "epoch": 0,
        "total_epochs": 10,
        "total_steps": 5000,
        "steps_per_epoch": 500,
        "timestamp": 1718000000.0
    }
"""

import json
import time
from pathlib import Path
from typing import Any

from ludwig.callbacks import Callback
from ludwig.utils.data_utils import NumpyEncoder


class _NDJSONChannel:
    """Append-only newline-delimited JSON file handle, opened lazily."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh = None

    def open(self) -> None:
        if self._fh is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._path, "a", buffering=1)

    def emit(self, event: dict[str, Any]) -> None:
        self.open()
        self._fh.write(json.dumps(event, cls=NumpyEncoder) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __del__(self) -> None:
        self.close()


class StudioCallback(Callback):
    """Streams training lifecycle events to a newline-delimited JSON file.

    Args:
        run_id: Unique identifier for this training run (used to key events in
            the Studio database).
        output_dir: Directory where ``metrics.jsonl`` is written. Must be
            writable. Created automatically if it does not exist.
        group_id: Optional hyperopt group ID. When set, trial events are also
            written to ``<group_output_dir>/trials.jsonl``.
        group_output_dir: Root output directory for the hyperopt group. Required
            when ``group_id`` is provided.
    """

    def __init__(
        self,
        run_id: str,
        output_dir: str,
        group_id: str | None = None,
        group_output_dir: str | None = None,
    ):
        self.run_id = run_id
        self.group_id = group_id
        self._ch = _NDJSONChannel(Path(output_dir) / "metrics.jsonl")
        self._trial_ch = _NDJSONChannel(Path(group_output_dir) / "trials.jsonl") if group_output_dir else None
        # Hyperopt trial tracking
        self._trial_idx: int = 0
        self._trial_best_metric_value: float | None = None
        self._trial_best_metric_name: str | None = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _emit(self, event: dict[str, Any]) -> None:
        event.setdefault("run_id", self.run_id)
        event.setdefault("timestamp", time.time())
        self._ch.emit(event)

    def _emit_trial(self, event: dict[str, Any]) -> None:
        if self._trial_ch is None:
            return
        event.setdefault("group_id", self.group_id)
        event.setdefault("timestamp", time.time())
        self._trial_ch.emit(event)

    def _progress(self, progress_tracker) -> dict[str, Any]:
        """Extract ETA / progress fields from a ProgressTracker (if available)."""
        out: dict[str, Any] = {}
        if hasattr(progress_tracker, "progress_pct"):
            out["progress_pct"] = round(progress_tracker.progress_pct, 4)
        if hasattr(progress_tracker, "eta_seconds"):
            eta = progress_tracker.eta_seconds
            out["eta_seconds"] = round(eta, 1) if eta is not None else None
        return out

    # ── Phase hooks ───────────────────────────────────────────────────────────

    def on_preprocess_start(self, config, **kwargs) -> None:
        self._emit({"type": "phase", "phase": "preprocessing", "epoch": None, "total_epochs": None})

    def on_train_start(self, model, config, config_fp, **kwargs) -> None:
        total = (config or {}).get("trainer", {}).get("epochs", None)
        self._emit({"type": "phase", "phase": "training", "epoch": 0, "total_epochs": total})

    def on_trainer_train_setup(self, trainer, save_path: str, is_coordinator: bool, **kwargs) -> None:
        if not is_coordinator:
            return
        steps_per_epoch = getattr(trainer, "steps_per_epoch", 0)
        total_steps = getattr(trainer, "total_steps", 0)
        epochs = getattr(trainer, "epochs", None)
        self._emit(
            {
                "type": "phase",
                "phase": "training_setup",
                "epoch": 0,
                "total_epochs": epochs,
                "total_steps": total_steps,
                "steps_per_epoch": steps_per_epoch,
            }
        )

    def on_eval_end(self, trainer, progress_tracker, save_path: str, **kwargs) -> None:
        self._emit(
            {
                "type": "phase",
                "phase": "evaluation",
                "epoch": progress_tracker.epoch,
                "total_epochs": None,
                **self._progress(progress_tracker),
            }
        )

    def on_train_end(self, output_directory: str, **kwargs) -> None:
        self._emit({"type": "phase", "phase": "completed", "epoch": None, "total_epochs": None})
        self._ch.close()

    # ── Metric hook ───────────────────────────────────────────────────────────

    def on_epoch_end(self, trainer, progress_tracker, save_path: str, **kwargs) -> None:
        epoch = progress_tracker.epoch
        step = getattr(progress_tracker, "steps", epoch)
        progress = self._progress(progress_tracker)

        for split, metrics_dict in [
            ("train", progress_tracker.train_metrics),
            ("validation", progress_tracker.validation_metrics),
            ("test", progress_tracker.test_metrics),
        ]:
            for feature, feature_metrics in metrics_dict.items():
                for metric_name, values in feature_metrics.items():
                    if not values:
                        continue
                    raw = values[-1]
                    # TrainerMetric namedtuple has .value; plain tuple uses index [-1]
                    value = getattr(raw, "value", raw[-1]) if hasattr(raw, "__len__") else raw
                    self._emit(
                        {
                            "type": "metric",
                            "epoch": epoch,
                            "step": step,
                            "split": split,
                            "feature": feature,
                            "metric": metric_name,
                            "value": float(value),
                            **progress,
                        }
                    )

        # Track best validation metric for hyperopt trial reporting
        if self.group_id is not None:
            best_val = getattr(progress_tracker, "best_eval_metric_value", None)
            if best_val is not None:
                self._trial_best_metric_value = float(best_val)
            vfield = getattr(trainer, "validation_field", None)
            vmetric = getattr(trainer, "validation_metric", None)
            if vfield and vmetric:
                self._trial_best_metric_name = f"{vfield}/{vmetric}"

    # ── Hyperopt hooks ────────────────────────────────────────────────────────

    def on_hyperopt_trial_start(self, parameters: dict[str, Any], **kwargs) -> None:
        self._trial_best_metric_value = None
        self._trial_best_metric_name = None
        self._emit_trial(
            {
                "type": "trial_start",
                "trial_idx": self._trial_idx,
                "parameters": dict(parameters) if parameters else {},
            }
        )

    def on_hyperopt_trial_end(self, parameters: dict[str, Any], **kwargs) -> None:
        self._emit_trial(
            {
                "type": "trial_end",
                "trial_idx": self._trial_idx,
                "parameters": dict(parameters) if parameters else {},
                "metric_score": self._trial_best_metric_value,
                "metric_name": self._trial_best_metric_name,
            }
        )
        self._trial_idx += 1

    def on_hyperopt_end(self, experiment_name: str, **kwargs) -> None:
        self._emit_trial({"type": "hyperopt_end", "trial_count": self._trial_idx})
        if self._trial_ch:
            self._trial_ch.close()
