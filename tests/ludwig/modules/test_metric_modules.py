import pytest
import torch

from ludwig.constants import ACCURACY, LOSS, ROC_AUC, ROOT_MEAN_SQUARED_ERROR
from ludwig.modules import metric_modules
from ludwig.modules.metric_modules import (
    get_best_function,
    get_improved_fun,
    get_initial_validation_value,
)


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize("output", [torch.tensor(6).float()])
def test_rmse_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.RMSEMetric()
    metric.update(preds, target)
    assert output == metric.compute()


@pytest.mark.parametrize("preds", [torch.tensor([0.2, 0.3, 0.8, 0.1])])
@pytest.mark.parametrize("target", [torch.tensor([0, 0, 1, 1])])
@pytest.mark.parametrize("output", [torch.tensor(0.5)])
def test_roc_auc_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.ROCAUCMetric()
    metric.update(preds, target)
    assert output == metric.compute()


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize("output", [torch.tensor(0.7527).float()])
def test_rmspe_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.RMSPEMetric()
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
    metric.update(preds, target)
    assert metric.compute() == output


def test_r2_score_single_sample():
    metric = metric_modules.R2Score(num_outputs=1)
    metric.update(preds=torch.tensor([0.8]), target=torch.arange(1))
    assert torch.isnan(metric.compute())


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize("output", [torch.tensor(-21.4655).float()])
def test_bwcewl_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.BWCEWLMetric()
    metric.update(preds, target)
    assert torch.isclose(output, metric.compute(), rtol=0.0001)


@pytest.mark.parametrize("preds", [torch.tensor([[0.5, 0.5], [0.2, 0.8], [0.6, 0.4]])])
@pytest.mark.parametrize("target", [torch.tensor([1, 1, 0])])
@pytest.mark.parametrize("output", [torch.tensor(0.5763)])
def test_softmax_cross_entropy_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.SoftmaxCrossEntropyMetric()
    metric.update(preds, target)
    assert torch.isclose(output, metric.compute(), rtol=0.0001)


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize("output", [torch.tensor(-21.4655).float()])
def test_sigmoid_cross_entropy_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.SigmoidCrossEntropyMetric()
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
    metric.update(preds, target)
    assert torch.allclose(metric.compute(), output)


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2)])
@pytest.mark.parametrize("target", [torch.tensor([[0, 1], [2, 1], [4, 5]]).float()])
@pytest.mark.parametrize("output", [torch.tensor(0.8333).float()])
def test_category_accuracy(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.CategoryAccuracy()
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
    metric = metric_modules.HitsAtKMetric(top_k=k)
    metric.update(preds, target)
    assert torch.isclose(output, metric.compute(), rtol=0.0001)


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize("output", [torch.tensor(6).float()])
def test_mae_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.MAEMetric()
    metric.update(preds, target)
    assert output == metric.compute()


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize("output", [torch.tensor(36).float()])
def test_mse_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.MSEMetric()
    metric.update(preds, target)
    assert output == metric.compute()


@pytest.mark.parametrize("preds", [torch.tensor([[0, 1], [1, 1]])])
@pytest.mark.parametrize("target", [torch.tensor([[1, 0], [1, 1]])])
@pytest.mark.parametrize("output", [torch.tensor(0.5)])
def test_jaccard_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.JaccardMetric()
    metric.update(preds, target)
    assert output == metric.compute()


# ---- Utility function tests ----


def test_get_improved_fun_minimize():
    """For a minimize metric (loss), a lower value is an improvement."""
    improved_fn = get_improved_fun(LOSS)
    assert improved_fn(0.5, 1.0) is True
    assert improved_fn(1.0, 0.5) is False
    assert improved_fn(1.0, 1.0) is False


def test_get_improved_fun_maximize():
    """For a maximize metric (accuracy), a higher value is an improvement."""
    improved_fn = get_improved_fun(ACCURACY)
    assert improved_fn(1.0, 0.5) is True
    assert improved_fn(0.5, 1.0) is False
    assert improved_fn(1.0, 1.0) is False


def test_get_initial_validation_value_minimize():
    """Initial validation value for a minimize metric should be +inf."""
    val = get_initial_validation_value(LOSS)
    assert val == float("inf")


def test_get_initial_validation_value_maximize():
    """Initial validation value for a maximize metric should be -inf."""
    val = get_initial_validation_value(ACCURACY)
    assert val == float("-inf")


def test_get_best_function_minimize():
    """Best function for a minimize metric should be min."""
    best_fn = get_best_function(LOSS)
    assert best_fn is min


def test_get_best_function_maximize():
    """Best function for a maximize metric should be max."""
    best_fn = get_best_function(ACCURACY)
    assert best_fn is max


# ---- LudwigMetric kwargs filtering test ----


def test_ludwig_metric_ignores_unknown_kwargs():
    """Instantiating a concrete LudwigMetric subclass with extra/unknown kwargs should not raise."""
    # CategoryAccuracy is a simple concrete subclass that goes through LudwigMetric.__init__
    metric = metric_modules.CategoryAccuracy(
        unknown_kwarg_1="foo",
        unknown_kwarg_2=42,
    )
    # The metric should be usable; verify by running a trivial update/compute cycle
    metric.update(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))
    assert metric.compute() == torch.tensor(1.0)


# ---- ROCAUCMetric error-path tests ----


def test_roc_auc_rejects_multidimensional_preds():
    """ROCAUCMetric.update should raise RuntimeError for multi-dimensional inputs."""
    metric = metric_modules.ROCAUCMetric()
    preds_2d = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    target_2d = torch.tensor([[1, 0], [0, 1]])
    with pytest.raises(RuntimeError, match="Only binary tasks supported"):
        metric.update(preds_2d, target_2d)


def test_roc_auc_rejects_out_of_range_preds():
    """ROCAUCMetric.update should raise RuntimeError when predictions are outside [0, 1]."""
    metric = metric_modules.ROCAUCMetric()
    preds = torch.tensor([-0.1, 0.5, 1.2])
    target = torch.tensor([0, 1, 1])
    with pytest.raises(RuntimeError, match="Only binary tasks supported"):
        metric.update(preds, target)
