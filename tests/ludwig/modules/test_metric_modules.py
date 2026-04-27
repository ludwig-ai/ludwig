"""Tests for Ludwig metric modules.

Distributed sync architecture
------------------------------
Ludwig's ``LudwigMetric.sync_context()`` overrides the torchmetrics base-class
implementation.  Understanding the call flow is critical to avoiding subtle bugs:

1. torchmetrics wraps every ``Metric.compute()`` call in ``sync_context()`` internally
   (see ``torchmetrics/metric.py::wrapped_func``).  This means sync happens automatically
   whenever ``compute()`` is called.

2. Ludwig's override asks the registered distributed strategy for a gather function:
   - ``AccelerateStrategy`` → returns ``torchmetrics.utilities.distributed.gather_all_tensors``
   - ``LocalStrategy``      → returns ``None`` (no-op when torch.distributed is absent)

3. **Ray TorchTrainer special case** (``eval_fn`` in ``ludwig/backend/ray.py``):
   TorchTrainer always initialises ``torch.distributed`` before entering the worker
   function.  Even though ``LocalStrategy`` is registered there, ``torch.distributed`` is
   available.  The override detects this and falls back to ``gather_all_tensors`` so that
   accumulator state is all-gathered across workers before ``compute()`` runs, giving
   metrics computed over the **full** dataset rather than one worker's shard.

Common pitfall
--------------
Do NOT wrap ``metric_fn.compute()`` in an explicit ``sync_context()`` from the outside
(e.g. from ``get_metrics()``).  torchmetrics calls ``sync_context()`` internally inside
``compute()``.  A manual outer call sets ``_is_synced = True``; when ``compute()`` then
calls ``sync_context()`` again it raises::

    TorchMetricsUserError: The Metric has already been synced.

See ``test_double_sync_raises`` below for a regression test that pins this behaviour.
"""

from unittest.mock import patch

import pytest
import torch
import torch.distributed
from torchmetrics.utilities.exceptions import TorchMetricsUserError

from ludwig.distributed import init_dist_strategy
from ludwig.modules import metric_modules
from ludwig.modules.metric_modules import RMSEMetric
from ludwig.schema.features.loss.loss import (
    BWCEWLossConfig,
    SigmoidCrossEntropyLossConfig,
    SoftmaxCrossEntropyLossConfig,
)

# Required for local testing.
init_dist_strategy("local")


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize("output", [torch.tensor(6).float()])
def test_rmse_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.RMSEMetric()
    with metric.sync_context():
        metric.update(preds, target)
        assert output == metric.compute()


@pytest.mark.parametrize("preds", [torch.tensor([0.2, 0.3, 0.8, 0.1])])
@pytest.mark.parametrize("target", [torch.tensor([0, 0, 1, 1])])
@pytest.mark.parametrize("output", [torch.tensor(0.5)])
def test_roc_auc_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.BinaryAUROCMetric(task="binary")
    with metric.sync_context():
        metric.update(preds, target)
        assert output == metric.compute()


@pytest.mark.parametrize("preds", [torch.tensor([0.2, 0.3, 0.8, 0.1, 0.8])])
@pytest.mark.parametrize("target", [torch.tensor([0, 0, 1, 1, 0])])
@pytest.mark.parametrize("output", [torch.tensor(0.6667).float()])
def test_specificity_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.SpecificityMetric()
    with metric.sync_context():
        metric.update(preds, target)
        assert torch.isclose(output, metric.compute(), rtol=0.0001)


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize("output", [torch.tensor(0.7527).float()])
def test_rmspe_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.RMSPEMetric()
    with metric.sync_context():
        metric.update(preds, target)
        assert torch.isclose(output, metric.compute(), rtol=0.0001)


@pytest.mark.parametrize(
    "preds,target,num_outputs,output",
    [
        (torch.arange(3), torch.arange(3, 6), 1, torch.tensor(-12.5)),
        (torch.arange(6).reshape(3, 2), torch.arange(6, 12).reshape(3, 2), 2, torch.tensor(-12.5)),
    ],
)
def test_r2_score(preds: torch.Tensor, target: torch.Tensor, num_outputs: int, output: torch.Tensor):
    metric = metric_modules.R2Score(num_outputs=num_outputs)
    with metric.sync_context():
        metric.update(preds, target)
        assert metric.compute() == output


def test_r2_score_single_sample():
    metric = metric_modules.R2Score(num_outputs=1)
    with metric.sync_context():
        metric.update(preds=torch.tensor([0.8]), target=torch.arange(1))
        assert torch.isnan(metric.compute())


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize("output", [torch.tensor(-21.4655).float()])
def test_bwcewl_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.BWCEWLMetric(BWCEWLossConfig())
    with metric.sync_context():
        metric.update(preds, target)
        assert torch.isclose(output, metric.compute(), rtol=0.0001)


@pytest.mark.parametrize("preds", [torch.tensor([[0.5, 0.5], [0.2, 0.8], [0.6, 0.4]])])
@pytest.mark.parametrize("target", [torch.tensor([1, 1, 0])])
@pytest.mark.parametrize("output", [torch.tensor(0.5763)])
def test_softmax_cross_entropy_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.SoftmaxCrossEntropyMetric(SoftmaxCrossEntropyLossConfig())
    with metric.sync_context():
        metric.update(preds, target)
        assert torch.isclose(output, metric.compute(), rtol=0.0001)


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize("output", [torch.tensor(-21.4655).float()])
def test_sigmoid_cross_entropy_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.SigmoidCrossEntropyMetric(SigmoidCrossEntropyLossConfig())
    with metric.sync_context():
        metric.update(preds, target)
        assert torch.isclose(output, metric.compute(), rtol=0.0001)


@pytest.mark.parametrize(
    "preds,target,output",
    [
        (
            torch.tensor([[0, 1], [3, 2], [4, 5]]),
            torch.tensor([[0, 1], [1, 2], [4, 5]]),
            torch.tensor(0.8),
        ),
        (
            torch.tensor([[0, 1, 2], [1, 3, 4], [3, 4, 5]]),
            torch.tensor([[0, 1, 2], [1, 1, 4], [3, 4, 5]]),
            torch.tensor(0.8750),
        ),
        (
            torch.tensor([[1, 5, 1, 5, 1, 5, 12, 12, 12], [10, 1, 5, 1, 5, 12, 12, 12, 12]]),
            torch.tensor([[1, 9, 5, 7, 5, 9, 13, 6, 0], [1, 9, 7, 13, 4, 7, 7, 7, 0]]),
            torch.tensor(0.05555555),
        ),
    ],
)
def test_token_accuracy_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.TokenAccuracyMetric()
    with metric.sync_context():
        metric.update(preds, target)
        assert torch.allclose(metric.compute(), output)


def test_sequence_accuracy_metric():
    target = torch.tensor(
        [
            [1, 6, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 6, 5, 4, 0],
            [1, 6, 5, 4, 0],
            [1, 6, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 6, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 6, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
        ]
    )
    preds = torch.tensor(
        [
            [1, 6, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 6, 5, 4, 0],
            [1, 6, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 6, 5, 4, 0],
            [1, 6, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 6, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 6, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
            [1, 4, 5, 4, 0],
        ]
    )
    metric = metric_modules.SequenceAccuracyMetric()
    with metric.sync_context():
        metric.update(preds, target)
        assert torch.isclose(metric.compute(), torch.tensor(0.8438), rtol=0.0001)


@pytest.mark.parametrize("preds", [torch.arange(6)])
@pytest.mark.parametrize("target", [torch.tensor([0, 1, 2, 1, 4, 5]).float()])
@pytest.mark.parametrize("output", [torch.tensor(0.7500).float()])
@pytest.mark.parametrize("one_hot", [False, True])
def test_category_accuracy(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor, one_hot: bool):
    if one_hot:
        target = torch.nn.functional.one_hot(target.long(), num_classes=6).float()
    metric = metric_modules.CategoryAccuracy(num_classes=6)
    with metric.sync_context():
        metric.update(preds, target)
        assert torch.isclose(output, metric.compute(), rtol=0.0001)


@pytest.mark.parametrize("preds", [torch.arange(6)])
@pytest.mark.parametrize("target", [torch.tensor([0, 1, 2, 1, 4, 5]).float()])
@pytest.mark.parametrize("output", [torch.tensor(0.8333).float()])
@pytest.mark.parametrize("one_hot", [False, True])
def test_category_accuracy_micro(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor, one_hot: bool):
    if one_hot:
        target = torch.nn.functional.one_hot(target.long(), num_classes=6).float()
    metric = metric_modules.CategoryAccuracyMicro(num_classes=6)
    with metric.sync_context():
        metric.update(preds, target)
        assert torch.isclose(output, metric.compute(), rtol=0.0001)


@pytest.mark.parametrize(
    "preds,target,output,k",
    [
        (
            torch.tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]]),
            torch.tensor([0, 1, 2]),
            torch.tensor(0.6667).float(),
            2,
        )
    ],
)
def test_hits_at_k_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor, k: int):
    metric = metric_modules.HitsAtKMetric(num_classes=3, top_k=k)
    with metric.sync_context():
        metric.update(preds, target)
        assert torch.isclose(output, metric.compute(), rtol=0.0001)


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize("output", [torch.tensor(6).float()])
def test_mae_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.MAEMetric()
    with metric.sync_context():
        metric.update(preds, target)
        assert output == metric.compute()


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize("output", [torch.tensor(36).float()])
def test_mse_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.MSEMetric()
    with metric.sync_context():
        metric.update(preds, target)
        assert output == metric.compute()


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize("output", [torch.tensor(0.7365440726280212)])
def test_mape_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.MAPEMetric()
    with metric.sync_context():
        metric.update(preds, target)
        assert output.item() == metric.compute().item()


@pytest.mark.parametrize("preds", [torch.tensor([[0, 1], [1, 1]])])
@pytest.mark.parametrize("target", [torch.tensor([[1, 0], [1, 1]])])
@pytest.mark.parametrize("output", [torch.tensor(0.5)])
def test_jaccard_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.JaccardMetric()
    with metric.sync_context():
        metric.update(preds, target)
        assert output == metric.compute()


def test_char_error_rate():
    metric = metric_modules.CharErrorRateMetric()
    with metric.sync_context():
        metric.update(
            ["this is the prediction", "there is an other sample"], ["this is the reference", "there is another one"]
        )
        assert torch.isclose(torch.tensor(0.3415), metric.compute(), rtol=0.5)


# ---------------------------------------------------------------------------
# Distributed sync_context behaviour
# ---------------------------------------------------------------------------


class TestSyncContextDispatch:
    """Unit tests for the gather-function selection logic in LudwigMetric.sync_context().

    These tests validate the three dispatch paths described in the module docstring
    without requiring a real multi-process distributed environment.  The mock helpers
    simulate the two dimensions that control dispatch:

    * Whether the registered Ludwig strategy provides a gather function.
    * Whether torch.distributed is initialised.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_metric() -> RMSEMetric:
        """Fresh RMSEMetric with LocalStrategy registered (mirrors eval_fn setup)."""
        init_dist_strategy("local")
        return RMSEMetric()

    @staticmethod
    def _passthrough_gather(tensor, **kwargs):
        """Simulate gather_all_tensors on a single process: return a one-element list."""
        return [tensor]

    # ------------------------------------------------------------------
    # Path 1: LocalStrategy + no torch.distributed → sync is a no-op
    # ------------------------------------------------------------------

    def test_local_strategy_no_distributed_is_noop(self):
        """With LocalStrategy and no torch.distributed, compute() works without error."""
        metric = self._make_metric()
        preds = torch.arange(6).reshape(3, 2).float()
        target = torch.arange(6, 12).reshape(3, 2).float()
        metric.update(preds, target)
        # No distributed → sync_context is a no-op → compute() must succeed
        result = metric.compute()
        assert result == torch.tensor(6).float()

    def test_local_strategy_no_distributed_gather_fn_is_none(self):
        """With LocalStrategy and no torch.distributed, sync() receives dist_sync_fn=None."""
        metric = self._make_metric()
        with patch.object(metric, "sync") as mock_sync, patch.object(metric, "unsync"):
            with metric.sync_context():
                pass
        mock_sync.assert_called_once()
        kwargs = mock_sync.call_args[1]
        assert kwargs["dist_sync_fn"] is None, (
            "Expected no gather function when LocalStrategy is active and " "torch.distributed is not initialised."
        )

    # ------------------------------------------------------------------
    # Path 2: LocalStrategy + torch.distributed initialised → fallback
    # ------------------------------------------------------------------

    def test_distributed_fallback_selected_when_torch_dist_initialized(self):
        """When LocalStrategy is active but torch.distributed is initialised, sync_context should fall back to
        torchmetrics' gather_all_tensors (not None).

        This is the Ray TorchTrainer / eval_fn scenario: TorchTrainer always calls
        torch.distributed.init_process_group() before eval_fn runs, so torch.distributed.is_initialized() is True even
        though Ludwig registered LocalStrategy for the eval pass.
        """
        from torchmetrics.utilities.distributed import gather_all_tensors

        metric = self._make_metric()
        with (
            patch.object(torch.distributed, "is_available", return_value=True),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(metric, "sync") as mock_sync,
            patch.object(metric, "unsync"),
        ):
            with metric.sync_context():
                pass

        mock_sync.assert_called_once()
        kwargs = mock_sync.call_args[1]
        assert kwargs["dist_sync_fn"] is gather_all_tensors, (
            "Expected gather_all_tensors as the fallback gather function when "
            "LocalStrategy is active but torch.distributed is initialised."
        )

    def test_distributed_fallback_compute_produces_correct_result(self):
        """End-to-end: with the fallback active, compute() returns the correct value.

        We mock gather_all_tensors to be a single-process passthrough so we can exercise the full sync → compute path
        without a real distributed environment.
        """
        metric = self._make_metric()
        preds = torch.arange(6).reshape(3, 2).float()
        target = torch.arange(6, 12).reshape(3, 2).float()
        metric.update(preds, target)

        with (
            patch.object(torch.distributed, "is_available", return_value=True),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch(
                "torchmetrics.utilities.distributed.gather_all_tensors",
                side_effect=self._passthrough_gather,
            ),
        ):
            result = metric.compute()

        assert result == torch.tensor(6).float()

    # ------------------------------------------------------------------
    # Path 3: AccelerateStrategy → strategy's gather fn takes precedence
    # ------------------------------------------------------------------

    def test_accelerate_strategy_gather_fn_used_not_fallback(self):
        """When AccelerateStrategy is active its gather function is used directly, even if torch.distributed is
        also initialised."""
        from torchmetrics.utilities.distributed import gather_all_tensors

        init_dist_strategy("accelerate")
        metric = RMSEMetric()

        # Mock AccelerateStrategy.gather_all_tensors_fn to return a sentinel so we can
        # distinguish it from the torch.distributed fallback path.
        sentinel_gather = object()
        with (
            patch(
                "ludwig.distributed.accelerate.AccelerateStrategy.gather_all_tensors_fn",
                return_value=sentinel_gather,
            ),
            patch.object(torch.distributed, "is_available", return_value=True),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(metric, "sync") as mock_sync,
            patch.object(metric, "unsync"),
        ):
            with metric.sync_context():
                pass

        kwargs = mock_sync.call_args[1]
        assert (
            kwargs["dist_sync_fn"] is sentinel_gather
        ), "Expected the strategy's gather function, not the torch.distributed fallback."
        assert kwargs["dist_sync_fn"] is not gather_all_tensors

        # Restore LocalStrategy for subsequent tests.
        init_dist_strategy("local")


# ---------------------------------------------------------------------------
# Regression: double-sync raises TorchMetricsUserError
# ---------------------------------------------------------------------------


class TestDoubleSyncRegression:
    """Regression tests for the double-sync bug.

    torchmetrics calls sync_context() internally inside compute().  If an outer call to
    sync_context() is added (e.g. from get_metrics()), the metric ends up synced twice
    when torch.distributed is active, raising TorchMetricsUserError.

    This is a canary: if these tests start *passing* without the error, something has
    changed in torchmetrics' internals that removes the auto-sync inside compute() and we
    need to revisit the sync strategy.
    """

    @staticmethod
    def _passthrough_gather(tensor, **kwargs):
        return [tensor]

    def test_double_sync_raises_when_distributed_active(self):
        """Wrapping compute() in a manual sync_context() must raise when distributed is active.

        This documents WHY get_metrics() must NOT call sync_context() explicitly.
        """
        init_dist_strategy("local")
        metric = RMSEMetric()
        preds = torch.arange(6).reshape(3, 2).float()
        target = torch.arange(6, 12).reshape(3, 2).float()
        metric.update(preds, target)

        with (
            patch.object(torch.distributed, "is_available", return_value=True),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch(
                "torchmetrics.utilities.distributed.gather_all_tensors",
                side_effect=self._passthrough_gather,
            ),
        ):
            with pytest.raises(TorchMetricsUserError, match="already been synced"):
                # This is the BAD pattern: do not do this in production code.
                with metric.sync_context():  # outer sync → sets _is_synced = True
                    metric.compute()  # torchmetrics calls sync_context() again → ERROR

    def test_no_double_sync_without_explicit_outer_call(self):
        """Calling compute() directly (no outer sync_context) must NOT raise, even when distributed is active.

        This is the correct usage pattern.
        """
        init_dist_strategy("local")
        metric = RMSEMetric()
        preds = torch.arange(6).reshape(3, 2).float()
        target = torch.arange(6, 12).reshape(3, 2).float()
        metric.update(preds, target)

        with (
            patch.object(torch.distributed, "is_available", return_value=True),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch(
                "torchmetrics.utilities.distributed.gather_all_tensors",
                side_effect=self._passthrough_gather,
            ),
        ):
            # Correct pattern: just call compute() — sync happens automatically inside.
            result = metric.compute()

        assert result == torch.tensor(6).float()
