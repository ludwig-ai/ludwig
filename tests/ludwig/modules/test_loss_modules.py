import contextlib
from typing import Optional, Type, Union

import pytest
import torch
from marshmallow import ValidationError

from ludwig.features.category_feature import CategoryOutputFeature
from ludwig.features.set_feature import SetOutputFeature
from ludwig.features.text_feature import TextOutputFeature
from ludwig.modules import loss_modules
from ludwig.schema.features.loss.loss import (
    BWCEWLossConfig,
    CORNLossConfig,
    HuberLossConfig,
    MAELossConfig,
    MAPELossConfig,
    MSELossConfig,
    RMSELossConfig,
    RMSPELossConfig,
    SigmoidCrossEntropyLossConfig,
    SoftmaxCrossEntropyLossConfig,
)
from ludwig.schema.model_config import ModelConfig
from tests.integration_tests.utils import category_feature, set_feature, text_feature


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
    preds: torch.Tensor, target: torch.Tensor, delta: float, output: Union[torch.Tensor, Type[Exception]]
):
    with pytest.raises(output) if not isinstance(output, torch.Tensor) else contextlib.nullcontext():
        loss = loss_modules.HuberLoss(HuberLossConfig.from_dict({"delta": delta}))
        value = loss(preds, target)
        assert value == output


@pytest.mark.parametrize("preds", [torch.tensor([[0.25, 0.2, 0.55], [0.2, 0.35, 0.45], [0.8, 0.1, 0.1]])])
@pytest.mark.parametrize("target", [torch.tensor([2, 1, 0])])
@pytest.mark.parametrize("output", [torch.tensor(0.7653)])
def test_corn_loss(preds: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    loss = loss_modules.CORNLoss(CORNLossConfig())
    assert torch.isclose(loss(preds, target), output, rtol=0.0001)


def test_dict_class_weights_category():
    input_features = [text_feature()]
    output_features = [category_feature(decoder={"vocab_size": 3})]
    config = {
        "input_features": input_features,
        "output_features": output_features,
    }

    # Set class weights as dictionary on config
    class_weights_dict = {"token_1": 0.1, "token_2": 0.2, "token_3": 0.3}
    config["output_features"][0]["loss"] = {"type": "softmax_cross_entropy", "class_weights": class_weights_dict}

    # Mock feature metadata
    feature_metadata = {
        "idx2str": ["token_1", "token_2", "token_3"],
        "str2idx": {"token_1": 0, "token_2": 1, "token_3": 2},
        "str2freq": {"token_1": 300, "token_2": 200, "token_3": 100},
        "vocab_size": 3,
        "preprocessing": {
            "missing_value_strategy": "drop_row",
            "fill_value": "<UNK>",
            "computed_fill_value": "<UNK>",
            "lowercase": False,
            "most_common": 10000,
            "cache_encoder_embeddings": False,
        },
    }

    model_config = ModelConfig.from_dict(config)

    CategoryOutputFeature.update_config_with_metadata(
        feature_config=model_config.output_features[0],
        feature_metadata=feature_metadata,
    )

    assert model_config.output_features[0].loss.class_weights == [0.1, 0.2, 0.3]


def test_dict_class_weights_text():
    input_features = [text_feature()]
    output_features = [text_feature(decoder={"vocab_size": 3, "max_sequence_length": 10})]
    config = {
        "input_features": input_features,
        "output_features": output_features,
    }

    # Set class weights as dictionary on config
    class_weights_dict = {
        "<EOS>": 0,
        "<SOS>": 0,
        "<PAD>": 0,
        "<UNK>": 0,
        "token_1": 0.5,
        "token_2": 0.4,
        "token_3": 0.1,
    }
    config["output_features"][0]["loss"] = {
        "type": "sequence_softmax_cross_entropy",
        "class_weights": class_weights_dict,
    }

    # Mock feature metadata
    feature_metadata = {
        "idx2str": ["<EOS>", "<SOS>", "<PAD>", "<UNK>", "token_1", "token_2", "token_3"],
        "str2idx": {"<EOS>": 0, "<SOS>": 1, "<PAD>": 2, "<UNK>": 3, "token_1": 4, "token_2": 5, "token_3": 6},
        "str2freq": {"<EOS>": 0, "<SOS>": 0, "<PAD>": 0, "<UNK>": 0, "token_1": 300, "token_2": 200, "token_3": 100},
        "str2idf": None,
        "vocab_size": 7,
        "max_sequence_length": 9,
        "max_sequence_length_99ptile": 9.0,
        "pad_idx": 2,
        "padding_symbol": "<PAD>",
        "unknown_symbol": "<UNK>",
        "index_name": None,
        "preprocessing": {
            "prompt": {
                "retrieval": {"type": None, "index_name": None, "model_name": None, "k": 0},
                "task": None,
                "template": None,
            },
            "pretrained_model_name_or_path": None,
            "tokenizer": "space_punct",
            "vocab_file": None,
            "sequence_length": None,
            "max_sequence_length": 256,
            "most_common": 20000,
            "padding_symbol": "<PAD>",
            "unknown_symbol": "<UNK>",
            "padding": "right",
            "lowercase": True,
            "missing_value_strategy": "drop_row",
            "fill_value": "<UNK>",
            "computed_fill_value": "<UNK>",
            "ngram_size": 2,
            "cache_encoder_embeddings": False,
            "compute_idf": False,
        },
    }

    model_config = ModelConfig.from_dict(config)

    TextOutputFeature.update_config_with_metadata(
        feature_config=model_config.output_features[0],
        feature_metadata=feature_metadata,
    )

    assert model_config.output_features[0].loss.class_weights == [0, 0, 0, 0, 0.5, 0.4, 0.1]


def test_dict_class_weights_set():
    input_features = [category_feature()]
    output_features = [set_feature()]
    config = {
        "input_features": input_features,
        "output_features": output_features,
    }

    # Set class weights as dictionary on config
    class_weights_dict = {"token_1": 0.1, "token_2": 0.2, "token_3": 0.3, "<UNK>": 0}
    config["output_features"][0]["loss"] = {"type": "sigmoid_cross_entropy", "class_weights": class_weights_dict}

    # Mock feature metadata
    feature_metadata = {
        "idx2str": ["token_1", "token_2", "token_3", "<UNK>"],
        "str2idx": {"token_1": 0, "token_2": 1, "token_3": 2, "<UNK>": 3},
        "str2freq": {"token_1": 300, "token_2": 200, "token_3": 100, "<UNK>": 0},
        "vocab_size": 4,
        "max_set_size": 3,
        "preprocessing": {
            "tokenizer": "space",
            "missing_value_strategy": "drop_row",
            "fill_value": "<UNK>",
            "computed_fill_value": "<UNK>",
            "lowercase": False,
            "most_common": 10000,
        },
    }

    model_config = ModelConfig.from_dict(config)

    SetOutputFeature.update_config_with_metadata(
        feature_config=model_config.output_features[0],
        feature_metadata=feature_metadata,
    )

    assert model_config.output_features[0].loss.class_weights == [0.1, 0.2, 0.3, 0]
