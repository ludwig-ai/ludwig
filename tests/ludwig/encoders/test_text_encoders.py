import pytest
import torch

from ludwig.encoders import text_encoders
from tests.integration_tests.utils import slow


@slow
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
    inputs = torch.rand((2, max_sequence_length)).type(albert_encoder.input_dtype)
    outputs = albert_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == albert_encoder.output_shape


@slow
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


@slow
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


@slow
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


@slow
@pytest.mark.parametrize("use_pretrained", [False])
@pytest.mark.parametrize("reduce_output", ["cls_pooled", "sum"])
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


@slow
@pytest.mark.parametrize("use_pretrained", [True, False])
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


@slow
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


@slow
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


@slow
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


@slow
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


@slow
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


@slow
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


@slow
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
