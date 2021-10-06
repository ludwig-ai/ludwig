import pytest

import torch

from ludwig.encoders import text_encoders


@pytest.mark.parametrize('use_pretrained', [True, False])
@pytest.mark.parametrize('reduce_output', [None, 'cls_pooled', 'sum'])
@pytest.mark.parametrize('max_sequence_length', [20])
def test_bert_encoder(
        use_pretrained: bool,
        reduce_output: str,
        max_sequence_length: int
):
    bert = text_encoders.BERTEncoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length
    )
    inputs = torch.rand((2, max_sequence_length)).type(bert.input_dtype)
    outputs = bert(inputs)
    assert outputs['encoder_output'].shape[1:] == bert.output_shape


# @pytest.mark.parametrize('use_pretrained', [True, False])
# @pytest.mark.parametrize('reduce_output', [None, 'cls_pooled', 'sum'])
# @pytest.mark.parametrize('max_sequence_length', [20])
# def test_gpt_encoder(
#         use_pretrained: bool,
#         reduce_output: str,
#         max_sequence_length: int
# ):
def test_gpt_encoder():
    gpt_encoder = text_encoders.GPTEncoder(
        use_pretrained=False,
        reduce_output=None,
        max_sequence_length=20
    )
    inputs = torch.rand((2, 20)).type(gpt_encoder.input_dtype)
    outputs = gpt_encoder(inputs)
    assert outputs['encoder_output'].shape[1:] == gpt_encoder.output_shape
