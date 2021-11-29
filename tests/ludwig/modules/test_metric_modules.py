import pytest

import torch

from ludwig.constants import BINARY, BINARY_WEIGHTED_CROSS_ENTROPY, SOFTMAX_CROSS_ENTROPY, CATEGORY, \
    SIGMOID_CROSS_ENTROPY, SET
from ludwig.modules import metric_modules


@pytest.mark.parametrize('preds', [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize('target', [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize('output', [torch.tensor(6).float()])
def test_rmse_metric(
        preds: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor
):
    metric = metric_modules.RMSEMetric()
    metric.update(preds, target)
    assert output == metric.compute()


@pytest.mark.parametrize('preds', [torch.tensor([0.2, 0.3, 0.8, 0.1])])
@pytest.mark.parametrize('target', [torch.tensor([0, 0, 1, 1])])
@pytest.mark.parametrize('output', [torch.tensor(0.5)])
def test_roc_auc_metric(
        preds: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor
):
    metric = metric_modules.ROCAUCMetric()
    metric.update(preds, target)
    assert output == metric.compute()


@pytest.mark.parametrize('preds', [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize('target', [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize('output', [torch.tensor(0.7527).float()])
def test_rmspe_metric(
        preds: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor
):
    metric = metric_modules.RMSPEMetric()
    metric.update(preds, target)
    assert torch.isclose(output, metric.compute(), rtol=0.0001)


@pytest.mark.parametrize(
    'preds,target,num_outputs,output',
    [
        (torch.arange(3), torch.arange(3, 6), 1, torch.tensor(-12.5)), 
        (
            torch.arange(6).reshape(3, 2), torch.arange(6, 12).reshape(3, 2),
            2, torch.tensor(-12.5)
        )
    ]
)
def test_r2_score(
        preds: torch.Tensor,
        target: torch.Tensor,
        num_outputs: int,
        output: torch.Tensor
):
    metric = metric_modules.R2Score(num_outputs=num_outputs)
    metric.update(preds, target)
    assert metric.compute() == output


@pytest.mark.parametrize('preds', [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize('target', [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize('output', [torch.tensor(-21.4655).float()])
def test_bwcewl_metric(
        preds: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor
):
    metric = metric_modules.get_metric_cls(BINARY, BINARY_WEIGHTED_CROSS_ENTROPY)()
    metric.update(preds, target)
    assert torch.isclose(output, metric.compute(), rtol=0.0001)


@pytest.mark.parametrize(
    'preds',
    [torch.tensor([[0.5, 0.5], [0.2, 0.8], [0.6, 0.4]])]
)
@pytest.mark.parametrize('target', [torch.tensor([1, 1, 0])])
@pytest.mark.parametrize('output', [torch.tensor(0.5763)])
def test_softmax_cross_entropy_metric(
        preds: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor
):
    metric = metric_modules.get_metric_cls(CATEGORY, SOFTMAX_CROSS_ENTROPY)()
    metric.update(preds, target)
    assert torch.isclose(output, metric.compute(), rtol=0.0001)


@pytest.mark.parametrize('preds', [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize('target', [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize('output', [torch.tensor(-42.9311).float()])
def test_sigmoid_cross_entropy_metric(
        preds: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor
):
    metric = metric_modules.get_metric_cls(SET, SIGMOID_CROSS_ENTROPY)()
    metric.update(preds, target)
    assert torch.isclose(output, metric.compute(), rtol=0.0001)


@pytest.mark.parametrize('preds', [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize(
    'target', [torch.tensor([[0, 1], [2, 1], [4, 5]]).float()])
@pytest.mark.parametrize('output', [torch.tensor(0.8).float()])
def test_token_accuracy_metric(
        preds: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor
):
    metric = metric_modules.TokenAccuracyMetric()
    metric.update(preds, target)
    assert metric.compute() == output


@pytest.mark.parametrize('preds', [torch.arange(6).reshape(3, 2)])
@pytest.mark.parametrize(
    'target', [torch.tensor([[0, 1], [2, 1], [4, 5]]).float()])
@pytest.mark.parametrize('output', [torch.tensor(0.8333).float()])
def test_category_accuracy(
        preds: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor
):
    metric = metric_modules.CategoryAccuracy()
    metric.update(preds, target)
    assert torch.isclose(output, metric.compute(), rtol=0.0001)


@pytest.mark.parametrize(
    'preds,target,output,k',
    [
        (
            torch.tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]]),
            torch.tensor([0, 1, 2]),
            torch.tensor(0.6667).float(),
            2
        )
    ]
)
def test_hits_at_k_metric(
        preds: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor,
        k: int
):
    metric = metric_modules.HitsAtKMetric(top_k=k)
    metric.update(preds, target)
    assert torch.isclose(output, metric.compute(), rtol=0.0001)


@pytest.mark.parametrize('preds', [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize('target', [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize('output', [torch.tensor(6).float()])
def test_mae_metric(
        preds: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor
):
    metric = metric_modules.MAEMetric()
    metric.update(preds, target)
    assert output == metric.compute()


@pytest.mark.parametrize('preds', [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize('target', [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize('output', [torch.tensor(36).float()])
def test_mse_metric(
        preds: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor
):
    metric = metric_modules.MSEMetric()
    metric.update(preds, target)
    assert output == metric.compute()


@pytest.mark.parametrize('preds', [torch.tensor([[0, 1], [1, 1]])])
@pytest.mark.parametrize('target', [torch.tensor([[1, 0], [1, 1]])])
@pytest.mark.parametrize('output', [torch.tensor(0.5)])
def test_jaccard_metric(
        preds: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor
):
    metric = metric_modules.JaccardMetric()
    metric.update(preds, target)
    assert output == metric.compute()
