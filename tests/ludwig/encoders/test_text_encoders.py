import os

import pytest
import torch

from ludwig.encoders import text_encoders
from tests.integration_tests.parameter_update_utils import check_module_parameters_updated


@pytest.mark.parametrize(
    "encoder_cls",
    [
        text_encoders.ALBERTEncoder,
        # text_encoders.BERTEncoder,
        # text_encoders.XLMEncoder,
        # text_encoders.GPTEncoder,
        # text_encoders.RoBERTaEncoder,
        # text_encoders.GPT2Encoder,
        # text_encoders.DistilBERTEncoder,
        # text_encoders.TransformerXLEncoder,
        # text_encoders.CTRLEncoder,
        # text_encoders.CamemBERTEncoder,
        # text_encoders.MT5Encoder,
        # text_encoders.XLMRoBERTaEncoder,
        # text_encoders.LongformerEncoder,
        # text_encoders.ELECTRAEncoder,
        # text_encoders.FlauBERTEncoder,
        # text_encoders.T5Encoder,
        # text_encoders.XLNetEncoder,
        # text_encoders.DistilBERTEncoder,
    ],
)
def test_hf_pretrained_default_model(tmpdir, encoder_cls):
    encoder = encoder_cls(
        use_pretrained=True,
        reduce_output="sum",
        max_sequence_length=20,
        pretrained_kwargs=dict(cache_dir=tmpdir),
    )
    inputs = torch.rand((2, 20)).type(encoder.input_dtype)
    outputs = encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == encoder.output_shape


@pytest.mark.parametrize("pretrained_model_name_or_path", ["bert-base-uncased"])
@pytest.mark.parametrize("reduce_output", [None, "sum", "cls_pooled"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_auto_transformer_encoder(pretrained_model_name_or_path: str, reduce_output: str, max_sequence_length: int):
    encoder = text_encoders.AutoTransformerEncoder(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(encoder.input_dtype)
    outputs = encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_albert_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    albert_encoder = text_encoders.ALBERTEncoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(albert_encoder.input_dtype)
    outputs = albert_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == albert_encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [None, "cls_pooled", "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_bert_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    bert = text_encoders.BERTEncoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(bert.input_dtype)
    outputs = bert(inputs)
    assert outputs["encoder_output"].shape[1:] == bert.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", ["last", "sum", "mean"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_xlm_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    xlm_encoder = text_encoders.XLMEncoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(xlm_encoder.input_dtype)
    outputs = xlm_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == xlm_encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_gpt_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    gpt_encoder = text_encoders.GPTEncoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(gpt_encoder.input_dtype)
    outputs = gpt_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == gpt_encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", ["cls_pooled", "sum", None])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_roberta_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    roberta_encoder = text_encoders.RoBERTaEncoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(roberta_encoder.input_dtype)
    outputs = roberta_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == roberta_encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_gpt2_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    gpt_encoder = text_encoders.GPT2Encoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(gpt_encoder.input_dtype)
    outputs = gpt_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == gpt_encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_distil_bert(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    distil_bert_encoder = text_encoders.DistilBERTEncoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(distil_bert_encoder.input_dtype)
    outputs = distil_bert_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == distil_bert_encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_transfoxl_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    transfo = text_encoders.TransformerXLEncoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.randint(10, (2, max_sequence_length)).type(transfo.input_dtype)
    outputs = transfo(inputs)
    assert outputs["encoder_output"].shape[1:] == transfo.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_ctrl_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    encoder = text_encoders.CTRLEncoder(
        max_sequence_length,
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
    )
    inputs = torch.rand((2, max_sequence_length)).type(encoder.input_dtype)
    outputs = encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [None, "cls_pooled"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_camembert_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    encoder = text_encoders.CamemBERTEncoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(encoder.input_dtype)
    outputs = encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_mt5_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    mt5_encoder = text_encoders.MT5Encoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(mt5_encoder.input_dtype)
    outputs = mt5_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == mt5_encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_xlmroberta_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    xlmroberta_encoder = text_encoders.XLMRoBERTaEncoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length,
    )
    inputs = torch.rand((2, max_sequence_length)).type(xlmroberta_encoder.input_dtype)
    outputs = xlmroberta_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == xlmroberta_encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [None, "cls_pooled"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_longformer_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    encoder = text_encoders.LongformerEncoder(
        use_pretrained=use_pretrained, reduce_output=reduce_output, max_sequence_length=max_sequence_length
    )
    inputs = torch.rand((2, max_sequence_length)).type(encoder.input_dtype)
    outputs = encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_electra_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    encoder = text_encoders.ELECTRAEncoder(
        use_pretrained=use_pretrained, reduce_output=reduce_output, max_sequence_length=max_sequence_length
    )
    inputs = torch.rand((2, max_sequence_length)).type(encoder.input_dtype)
    outputs = encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_flaubert_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    encoder = text_encoders.FlauBERTEncoder(
        use_pretrained=use_pretrained, reduce_output=reduce_output, max_sequence_length=max_sequence_length
    )
    inputs = torch.rand((2, max_sequence_length)).type(encoder.input_dtype)
    outputs = encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_t5_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    encoder = text_encoders.T5Encoder(
        use_pretrained=use_pretrained, reduce_output=reduce_output, max_sequence_length=max_sequence_length
    )
    inputs = torch.rand((2, max_sequence_length)).type(encoder.input_dtype)
    outputs = encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == encoder.output_shape


@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", [None, "sum"])
@pytest.mark.parametrize("max_sequence_length", [20])
def test_xlnet_encoder(use_pretrained: bool, reduce_output: str, max_sequence_length: int):
    xlnet_encoder = text_encoders.XLNetEncoder(
        use_pretrained=use_pretrained, reduce_output=reduce_output, max_sequence_length=max_sequence_length
    )
    inputs = torch.rand((2, max_sequence_length)).type(xlnet_encoder.input_dtype)
    outputs = xlnet_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == xlnet_encoder.output_shape


@pytest.mark.parametrize("trainable", [True, False])
def test_distilbert_param_updates(trainable: bool):
    max_sequence_length = 20
    distil_bert_encoder = text_encoders.DistilBERTEncoder(
        use_pretrained=False,
        max_sequence_length=max_sequence_length,
        trainable=trainable,
    )

    # send a random input through the model with its initial weights
    inputs = torch.rand((2, max_sequence_length)).type(distil_bert_encoder.input_dtype)
    outputs = distil_bert_encoder(inputs)

    # perform a backward pass to update the model params
    target = torch.randn(outputs["encoder_output"].shape)
    check_module_parameters_updated(distil_bert_encoder, (inputs,), target)

    # send the same input through the model again. should be different if trainable, else the same
    outputs2 = distil_bert_encoder(inputs)

    encoder_output1 = outputs["encoder_output"]
    encoder_output2 = outputs2["encoder_output"]

    if trainable:
        # Outputs should be different if the model was updated
        assert not torch.equal(encoder_output1, encoder_output2)
    else:
        # Outputs should be the same if the model wasn't updated
        assert torch.equal(encoder_output1, encoder_output2)
