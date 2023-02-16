import pytest
import torch

from ludwig.modules import metric_modules


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
    metric = metric_modules.BinaryAUROCMetric(task="binary")
    metric.update(preds, target)
    assert output == metric.compute()


@pytest.mark.parametrize("preds", [torch.tensor([0.2, 0.3, 0.8, 0.1, 0.8])])
@pytest.mark.parametrize("target", [torch.tensor([0, 0, 1, 1, 0])])
@pytest.mark.parametrize("output", [torch.tensor(0.6667).float()])
def test_specificity_metric(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.SpecificityMetric()
    metric.update(preds, target)
    assert torch.isclose(output, metric.compute(), rtol=0.0001)


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
    metric.update(preds, target)
    assert torch.isclose(metric.compute(), torch.tensor(0.8438), rtol=0.0001)


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2)])
@pytest.mark.parametrize("target", [torch.tensor([[0, 1], [2, 1], [4, 5]]).float()])
@pytest.mark.parametrize("output", [torch.tensor(0.7500).float()])
def test_category_accuracy(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    metric = metric_modules.CategoryAccuracy(num_classes=6)
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
