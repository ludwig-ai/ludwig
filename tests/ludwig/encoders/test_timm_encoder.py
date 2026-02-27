import pytest
import torch

timm = pytest.importorskip("timm", reason="timm not installed")

from ludwig.encoders.image.timm import (  # noqa: E402
    TimmCAFormerEncoder,
    TimmConvFormerEncoder,
    TimmEncoder,
    TimmPoolFormerEncoder,
)


@pytest.mark.parametrize(
    "encoder_cls,model_name",
    [
        (TimmEncoder, "resnetv2_50"),
        (TimmCAFormerEncoder, "caformer_s18"),
        (TimmConvFormerEncoder, "convformer_s18"),
        (TimmPoolFormerEncoder, "poolformerv2_s12"),
    ],
    ids=["timm_resnet", "caformer", "convformer", "poolformer"],
)
def test_timm_encoder_forward(encoder_cls, model_name):
    encoder = encoder_cls(model_name=model_name, use_pretrained=False, trainable=True)

    # Get the expected input shape from the encoder
    input_shape = encoder.input_shape  # (C, H, W)
    batch = torch.randn(2, *input_shape)

    output = encoder(batch)
    assert "encoder_output" in output

    out_tensor = output["encoder_output"]
    assert out_tensor.shape[0] == 2
    assert out_tensor.shape[1:] == encoder.output_shape


@pytest.mark.parametrize("trainable", [True, False])
def test_timm_encoder_trainable(trainable):
    encoder = TimmCAFormerEncoder(model_name="caformer_s18", use_pretrained=False, trainable=trainable)

    for p in encoder.model.parameters():
        assert p.requires_grad == trainable


def test_timm_encoder_output_shape_property():
    encoder = TimmEncoder(model_name="caformer_s18", use_pretrained=False)
    assert len(encoder.output_shape) == 1
    assert encoder.output_shape[0] > 0
