import contextlib
from typing import Optional, Union

import pytest
import torch
from marshmallow import ValidationError

from ludwig.modules import loss_modules
from ludwig.schema.features.loss.loss import (
    BWCEWLossConfig,
    HuberLossConfig,
    MAELossConfig,
    MAPELossConfig,
    MSELossConfig,
    RMSELossConfig,
    RMSPELossConfig,
    SigmoidCrossEntropyLossConfig,
    SoftmaxCrossEntropyLossConfig,
)


def from_float(v: float) -> torch.Tensor:
    return torch.tensor(v).float()


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize("output", [torch.tensor(36).float()])
def test_mse_loss(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    loss = loss_modules.MSELoss(MSELossConfig())
    assert loss(preds, target) == output


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize("output", [torch.tensor(6).float()])
def test_mae_loss(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    loss = loss_modules.MAELoss(MAELossConfig())
    assert loss(preds, target) == output


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize("output", [torch.tensor(0.7365440726280212)])
def test_mape_loss(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    loss = loss_modules.MAPELoss(MAPELossConfig())
    assert loss(preds, target) == output


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize("output", [torch.tensor(6).float()])
def test_rmse_loss(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    loss = loss_modules.RMSELoss(RMSELossConfig())
    assert loss(preds, target) == output


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize("output", [torch.tensor(0.7527).float()])
def test_rmspe_loss(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    loss = loss_modules.RMSPELoss(RMSPELossConfig())
    assert torch.isclose(loss(preds, target), output, rtol=0.0001)


@pytest.mark.parametrize("preds", [torch.tensor([[0.1, 0.2]]).float()])
@pytest.mark.parametrize("target", [torch.tensor([[0.0, 0.2]]).float()])
@pytest.mark.parametrize("output", [torch.tensor(707.1068).float()])
def test_rmspe_loss_zero_targets(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    loss = loss_modules.RMSPELoss(RMSPELossConfig())
    assert torch.isclose(loss(preds, target), output, rtol=0.0001)


@pytest.mark.parametrize(
    "confidence_penalty,positive_class_weight,robust_lambda,output",
    [
        (0.0, None, 0, from_float(-21.4655)),
        (2.0, None, 0, from_float(-21.1263)),
        (0.0, 2.0, 0, from_float(-20.1222)),
        (0.0, None, 2, from_float(22.4655)),
        (2, 2, 2, from_float(21.4614)),
    ],
)
@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
def test_bwcew_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    confidence_penalty: float,
    positive_class_weight: Optional[float],
    robust_lambda: int,
    output: torch.Tensor,
):
    loss = loss_modules.BWCEWLoss(
        BWCEWLossConfig(
            positive_class_weight=positive_class_weight,
            robust_lambda=robust_lambda,
            confidence_penalty=confidence_penalty,
        )
    )
    assert torch.isclose(loss(preds, target), output)


@pytest.mark.parametrize("preds", [torch.tensor([[0.5, 0.5], [0.2, 0.8], [0.6, 0.4]])])
@pytest.mark.parametrize("target", [torch.tensor([1, 1, 0])])
@pytest.mark.parametrize("output", [torch.tensor(0.5763)])
def test_softmax_cross_entropy_loss(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    loss = loss_modules.SoftmaxCrossEntropyLoss(SoftmaxCrossEntropyLossConfig())
    assert torch.isclose(loss(preds, target), output, rtol=0.0001)


@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
@pytest.mark.parametrize("output", [torch.tensor(-21.4655).float()])
def test_sigmoid_cross_entropy_loss(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    loss = loss_modules.SigmoidCrossEntropyLoss(SigmoidCrossEntropyLossConfig())
    assert torch.isclose(loss(preds, target), output)


@pytest.mark.parametrize(
    "delta,output",
    [
        (1.0, from_float(5.5000)),
        (0.5, from_float(2.8750)),
        (2.0, from_float(10.0)),
        (0.0, ValidationError),
    ],
)
@pytest.mark.parametrize("preds", [torch.arange(6).reshape(3, 2).float()])
@pytest.mark.parametrize("target", [torch.arange(6, 12).reshape(3, 2).float()])
def test_huber_loss(
    preds: torch.Tensor, target: torch.Tensor, delta: float, output: Union[torch.Tensor, type[Exception]]
):
    with pytest.raises(output) if not isinstance(output, torch.Tensor) else contextlib.nullcontext():
        loss = loss_modules.HuberLoss(HuberLossConfig.from_dict({"delta": delta}))
        value = loss(preds, target)
        assert value == output
