"""Tests for trainer mixins."""

import time

from ludwig.trainers.mixins import (
    BatchSizeTuningMixin,
    CheckpointMixin,
    EarlyStoppingMixin,
    MetricsMixin,
    ProfilingMixin,
)


class TestCheckpointMixin:
    def setup_method(self):
        self.mixin = CheckpointMixin.__new__(CheckpointMixin)

    def test_checkpoint_at_epoch_end(self):
        assert self.mixin.should_checkpoint(steps=5, steps_per_checkpoint=100, epoch_end=True)

    def test_checkpoint_at_step_interval(self):
        assert self.mixin.should_checkpoint(steps=100, steps_per_checkpoint=100)

    def test_no_checkpoint_mid_interval(self):
        assert not self.mixin.should_checkpoint(steps=50, steps_per_checkpoint=100)

    def test_no_checkpoint_when_disabled(self):
        assert not self.mixin.should_checkpoint(steps=100, steps_per_checkpoint=0)


class TestEarlyStoppingMixin:
    def setup_method(self):
        self.mixin = EarlyStoppingMixin.__new__(EarlyStoppingMixin)

    def test_stop_when_no_improvement(self):
        assert self.mixin.should_early_stop(steps_since_improvement=10, early_stop_rounds=10)

    def test_no_stop_when_improving(self):
        assert not self.mixin.should_early_stop(steps_since_improvement=3, early_stop_rounds=10)

    def test_no_stop_when_disabled_zero(self):
        assert not self.mixin.should_early_stop(steps_since_improvement=100, early_stop_rounds=0)

    def test_no_stop_when_disabled_negative(self):
        assert not self.mixin.should_early_stop(steps_since_improvement=100, early_stop_rounds=-1)


class TestMetricsMixin:
    def setup_method(self):
        self.mixin = MetricsMixin.__new__(MetricsMixin)

    def test_format_simple_metrics(self):
        metrics = {"label": {"accuracy": 0.9512, "loss": 0.1234}}
        result = self.mixin.format_metrics(metrics)
        assert "label.accuracy=0.9512" in result
        assert "label.loss=0.1234" in result

    def test_format_with_prefix(self):
        metrics = {"label": {"accuracy": 0.85}}
        result = self.mixin.format_metrics(metrics, prefix="val")
        assert "val.label.accuracy=0.8500" in result

    def test_format_empty_metrics(self):
        result = self.mixin.format_metrics({})
        assert result == ""

    def test_format_skips_non_float(self):
        metrics = {"label": {"accuracy": 0.9, "predictions": [1, 2, 3]}}
        result = self.mixin.format_metrics(metrics)
        assert "accuracy" in result
        assert "predictions" not in result


class TestBatchSizeTuningMixin:
    def test_returns_max_when_no_oom(self):
        mixin = BatchSizeTuningMixin.__new__(BatchSizeTuningMixin)
        # Without actual model/dataset, find_max_batch_size just returns max
        result = mixin.find_max_batch_size(model=None, dataset=None, max_batch_size=512)
        assert result == 512


class TestProfilingMixin:
    def setup_method(self):
        self.mixin = ProfilingMixin.__new__(ProfilingMixin)
        self.mixin.__init_profiling__()

    def test_start_stop_timer(self):
        self.mixin.start_timer("train_step")
        time.sleep(0.01)
        elapsed = self.mixin.stop_timer("train_step")
        assert elapsed > 0

    def test_timing_accumulates(self):
        self.mixin.start_timer("eval")
        time.sleep(0.01)
        self.mixin.stop_timer("eval")
        self.mixin.start_timer("eval")
        time.sleep(0.01)
        self.mixin.stop_timer("eval")
        summary = self.mixin.get_timing_summary()
        assert "eval" in summary
        assert summary["eval"] > 0.01

    def test_stop_nonexistent_timer(self):
        elapsed = self.mixin.stop_timer("nonexistent")
        assert elapsed == 0.0

    def test_timing_summary_empty(self):
        assert self.mixin.get_timing_summary() == {}
