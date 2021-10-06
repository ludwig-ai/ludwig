import pytest

import torch

from ludwig.encoders.text_encoders import BERTEncoder


@pytest.mark.parametrize('use_pretrained', [True, False])
@pytest.mark.parametrize('reduce_output', [None, 'cls_pooled', 'sum'])
@pytest.mark.parametrize('max_sequence_length', [20])
def test_bert_encoder(
        use_pretrained: bool,
        reduce_output: str,
        max_sequence_length: int
):
    bert = BERTEncoder(
        use_pretrained=use_pretrained,
        reduce_output=reduce_output,
        max_sequence_length=max_sequence_length
    )
    inputs = torch.rand((2, max_sequence_length)).type(bert.input_dtype)
    outputs = bert(inputs)
    assert outputs['encoder_output'].shape[1:] == bert.output_shape
