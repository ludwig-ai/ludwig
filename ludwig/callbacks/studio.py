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


class StudioCallback(Callback):
    """Streams training lifecycle events to a newline-delimited JSON file.

    Args:
        run_id: Unique identifier for this training run (used to key events in
            the Studio database).
        output_dir: Directory where ``metrics.jsonl`` is written. Must be
            writable. Created automatically if it does not exist.
    """

    def __init__(self, run_id: str, output_dir: str):
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self._fh = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _open(self) -> None:
        if self._fh is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # line-buffered (buffering=1) so each emit is immediately visible
            self._fh = open(self.output_dir / "metrics.jsonl", "a", buffering=1)

    def _emit(self, event: dict[str, Any]) -> None:
        self._open()
        event.setdefault("run_id", self.run_id)
        event.setdefault("timestamp", time.time())
        self._fh.write(json.dumps(event, default=str) + "\n")

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
        if self._fh:
            self._fh.close()
            self._fh = None

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
