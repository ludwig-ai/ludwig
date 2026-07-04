import itertools
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ludwig.backend import LOCAL_BACKEND
from ludwig.callbacks import Callback
from ludwig.constants import COLUMN, MISSING_VALUE_STRATEGY, NAME, PREPROCESSING, PROC_COLUMN, TYPE
from ludwig.data.dataframe.pandas import PandasEngine
from ludwig.data.preprocessing import build_data
from ludwig.data.preprocessing_progress import get_total_partitions, PreprocessingProgressTracker


class ProgressCollector(Callback):
    def __init__(self):
        self.values = []

    def on_preprocess_progress(self, progress, **kwargs):
        self.values.append(progress)


# ---------------------------------------------------------------------------
# Pandas / local backend
# ---------------------------------------------------------------------------


def test_tracker_fires_progress_callbacks():
    """Each map_partitions call fires a callback synchronously (no sleep needed).

    With total=3: increments fire 1/3, 2/3, 3/3=1.0; stop() fires 1.0 again.
    Intermediate (strictly < 1.0) count = total - 1 = 2.
    """
    collector = ProgressCollector()
    n = 3
    tracker = PreprocessingProgressTracker(total=n, callbacks=[collector], use_ray=False)
    tracker.start()

    engine = PandasEngine()
    series = pd.Series([1, 2, 3])
    for _ in range(n):
        engine.map_partitions(series, lambda s: s, progress_tracker=tracker)

    tracker.stop()

    intermediate = [v for v in collector.values if v < 1.0]
    assert len(intermediate) == n - 1, f"Expected {n - 1} intermediate values, got {intermediate}"
    assert collector.values[-1] == 1.0


def test_tracker_exact_fractions():
    """Progress values should be 1/N, 2/N, ..., N/N = 1.0."""
    collector = ProgressCollector()
    n = 5
    tracker = PreprocessingProgressTracker(total=n, callbacks=[collector], use_ray=False)
    tracker.start()

    engine = PandasEngine()
    series = pd.Series(range(10))
    for _ in range(n):
        engine.map_partitions(series, lambda s: s, progress_tracker=tracker)

    tracker.stop()

    # Collect distinct values before the final 1.0
    unique = sorted({v for v in collector.values if v < 1.0})
    expected = [k / n for k in range(1, n)]
    assert unique == pytest.approx(expected), f"Unexpected progress steps: {unique}"
    assert collector.values[-1] == 1.0


def test_tracker_no_callbacks_does_not_crash():
    tracker = PreprocessingProgressTracker(total=2, callbacks=[], use_ray=False)
    tracker.start()

    engine = PandasEngine()
    series = pd.Series([1, 2, 3])
    engine.map_partitions(series, lambda s: s, progress_tracker=tracker)
    engine.map_partitions(series, lambda s: s, progress_tracker=tracker)

    tracker.stop()


def test_map_partitions_without_tracker_unchanged():
    engine = PandasEngine()
    series = pd.Series([1, 2, 3])
    result = engine.map_partitions(series, lambda s: s * 2)
    pd.testing.assert_series_equal(result, series * 2)


def test_get_total_partitions_pandas():
    cols = {"a": pd.Series([1, 2, 3]), "b": pd.Series([4, 5, 6])}
    assert get_total_partitions(cols, use_ray=False) == 1


def test_progress_monotonically_increases():
    collector = ProgressCollector()
    tracker = PreprocessingProgressTracker(total=5, callbacks=[collector], use_ray=False)
    tracker.start()

    engine = PandasEngine()
    series = pd.Series(range(10))
    for _ in range(5):
        engine.map_partitions(series, lambda s: s, progress_tracker=tracker)

    tracker.stop()

    # All values should be non-decreasing
    for a, b in itertools.pairwise(collector.values):
        assert b >= a, f"Progress went backwards: {a} -> {b}"
    assert collector.values[-1] == 1.0


def test_progress_capped_at_one_when_overcounted():
    """If a feature calls map_partitions more times than the denominator, progress stays <= 1.0."""
    collector = ProgressCollector()
    tracker = PreprocessingProgressTracker(total=2, callbacks=[collector], use_ray=False)
    tracker.start()

    engine = PandasEngine()
    series = pd.Series([1, 2, 3])
    # 4 calls against a denominator of 2
    for _ in range(4):
        engine.map_partitions(series, lambda s: s, progress_tracker=tracker)

    tracker.stop()

    assert all(v <= 1.0 for v in collector.values), f"Progress exceeded 1.0: {collector.values}"
    assert collector.values[-1] == 1.0


def test_start_stop_without_increments():
    """stop() always fires 1.0 even if no increments happened."""
    collector = ProgressCollector()
    tracker = PreprocessingProgressTracker(total=5, callbacks=[collector], use_ray=False)
    tracker.start()
    tracker.stop()

    assert collector.values[-1] == 1.0


# ---------------------------------------------------------------------------
# Ray backend (mocked — no real Ray cluster needed)
# ---------------------------------------------------------------------------


def _make_mock_actor(total):
    """Returns a mock Ray actor that simulates the _ProgressActor interface."""
    state = {"completed": 0, "total": total}

    actor = MagicMock()
    # increment.remote() increments synchronously in the mock
    actor.increment.remote.side_effect = lambda: state.__setitem__("completed", state["completed"] + 1)
    # get_completed.remote() returns a future-like that ray.get() resolves
    actor.get_completed.remote.side_effect = lambda: state["completed"]
    actor.get_total.remote.side_effect = lambda: state["total"]
    return actor, state


def test_ray_tracker_fires_via_poll():
    """Ray backend: poll thread fires callbacks; stop() drains actor then fires 1.0."""
    collector = ProgressCollector()

    mock_actor, state = _make_mock_actor(total=4)

    with (
        patch("ludwig.data.preprocessing_progress._make_ray_actor", return_value=mock_actor),
        patch("ray.get", side_effect=lambda fut, **kw: fut),
        patch("ray.kill"),
    ):
        tracker = PreprocessingProgressTracker(total=4, callbacks=[collector], use_ray=True)
        tracker.start()

        # Simulate 4 partition completions: actor incremented directly
        for _ in range(4):
            state["completed"] += 1

        tracker.stop()

    assert collector.values[-1] == 1.0


def test_ray_stop_drains_actor_queue():
    """stop() calls get_completed.remote() before firing 1.0, ensuring actor queue is drained."""
    collector = ProgressCollector()
    mock_actor, state = _make_mock_actor(total=2)

    drain_calls = []

    def mock_ray_get(fut, **kw):
        drain_calls.append(fut)
        return fut  # fut is already the value in our mock

    with (
        patch("ludwig.data.preprocessing_progress._make_ray_actor", return_value=mock_actor),
        patch("ray.get", side_effect=mock_ray_get),
        patch("ray.kill"),
    ):
        tracker = PreprocessingProgressTracker(total=2, callbacks=[collector], use_ray=True)
        tracker.start()
        state["completed"] = 2
        tracker.stop()

    # stop() must have called get_completed.remote() to drain the queue
    assert mock_actor.get_completed.remote.called
    assert collector.values[-1] == 1.0


# ---------------------------------------------------------------------------
# Large-dataset simulation (pandas, stress)
# ---------------------------------------------------------------------------


def test_large_feature_count_pandas():
    """Simulate 50 features × 1 pandas partition: all 50 intermediate values emitted."""
    collector = ProgressCollector()
    n = 50
    tracker = PreprocessingProgressTracker(total=n, callbacks=[collector], use_ray=False)
    tracker.start()

    engine = PandasEngine()
    series = pd.Series(range(1000))
    for _ in range(n):
        engine.map_partitions(series, lambda s: s * 2, progress_tracker=tracker)

    tracker.stop()

    intermediate = [v for v in collector.values if v < 1.0]
    # The nth increment fires 1.0 directly, so strictly-intermediate count = n - 1
    assert len(intermediate) == n - 1, f"Expected {n - 1} intermediate ticks, got {len(intermediate)}"
    assert collector.values[-1] == 1.0
    # Strictly increasing
    for a, b in itertools.pairwise(intermediate):
        assert b > a


# ---------------------------------------------------------------------------
# Regression: feature-level increment (build_data loop)
# ---------------------------------------------------------------------------


def test_progress_fires_for_features_not_calling_map_partitions():
    """Regression test for issue #4195.

    Most feature types (category, binary, number without normalization) use
    map_objects or direct series operations rather than map_partitions.  The
    old implementation incremented the counter only inside the map_partitions
    monkey-patch, so those features produced zero progress callbacks.

    The fix moves the increment to build_data's feature loop: one tick after
    each add_feature_data() call, regardless of which engine operation the
    feature uses internally.

    This test verifies the fix by patching add_feature_data to never call
    map_partitions.  With the old code the counter stays at 0 and only stop()'s
    forced 1.0 is emitted; with the new code every feature fires a callback.
    """
    n_features = 5
    feature_names = [f"feat_{i}" for i in range(n_features)]

    feature_configs = [
        {NAME: name, TYPE: "number", COLUMN: name, PROC_COLUMN: f"{name}_proc"} for name in feature_names
    ]
    # Minimal preprocessing metadata — fill_with_const with no outlier strategy
    # so that handle_missing_values/handle_outliers are no-ops.
    training_set_metadata = {
        name: {
            PREPROCESSING: {
                MISSING_VALUE_STRATEGY: "fill_with_const",
                "computed_fill_value": 0.0,
            }
        }
        for name in feature_names
    }
    input_cols = {name: pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]) for name in feature_names}

    collector = ProgressCollector()
    tracker = PreprocessingProgressTracker(total=n_features, callbacks=[collector], use_ray=False)
    tracker.start()

    # Patch add_feature_data to do nothing — simulating features that never
    # call map_partitions (binary, category, ...).
    mock_feature_type = MagicMock()
    mock_feature_type.add_feature_data.return_value = None

    with patch("ludwig.data.preprocessing.get_from_registry", return_value=mock_feature_type):
        build_data(input_cols, feature_configs, training_set_metadata, LOCAL_BACKEND, False, tracker)

    tracker.stop()

    intermediate = [v for v in collector.values if v < 1.0]

    # OLD code: 0 intermediate callbacks (map_partitions never called → counter
    # never incremented → only stop()'s forced 1.0 is present).
    # NEW code: n_features - 1 intermediate callbacks (one per feature, the
    # last one fires 1.0 directly and is not counted as intermediate).
    assert len(intermediate) == n_features - 1, (
        f"Expected {n_features - 1} intermediate callbacks (one per feature). "
        f"Got {collector.values!r}. "
        "This fails with the old map_partitions-only implementation."
    )
    assert collector.values[-1] == 1.0
    # Strictly increasing
    for a, b in itertools.pairwise(collector.values):
        assert b >= a, f"Progress went backwards: {a} -> {b}"
