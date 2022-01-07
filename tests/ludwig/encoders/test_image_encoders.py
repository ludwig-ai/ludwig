import pytest
import torch

from ludwig.encoders.image_encoders import MLPMixerEncoder, ResNetEncoder, Stacked2DCNN, ViTEncoder


@pytest.mark.parametrize("height,width,num_conv_layers,num_channels", [(224, 224, 5, 3)])
def test_stacked2d_cnn(height: int, width: int, num_conv_layers: int, num_channels: int):
    stacked_2d_cnn = Stacked2DCNN(
        height=height, width=width, num_conv_layers=num_conv_layers, num_channels=num_channels
    )
    inputs = torch.rand(2, num_channels, height, width)
    outputs = stacked_2d_cnn(inputs)
    assert outputs["encoder_output"].shape[1:] == stacked_2d_cnn.output_shape()


@pytest.mark.parametrize("height,width,num_channels", [(224, 224, 1), (224, 224, 3)])
def test_resnet_encoder(height: int, width: int, num_channels: int):
    resnet = ResNetEncoder(height=height, width=width, num_channels=num_channels)
    inputs = torch.rand(2, num_channels, height, width)
    outputs = resnet(inputs)
    assert outputs["encoder_output"].shape[1:] == resnet.output_shape()


@pytest.mark.parametrize("height,width,num_channels", [(224, 224, 3)])
def test_mlp_mixer_encoder(height: int, width: int, num_channels: int):
    mlp_mixer = MLPMixerEncoder(height=height, width=width, num_channels=num_channels)
    inputs = torch.rand(2, num_channels, height, width)
    outputs = mlp_mixer(inputs)
    assert outputs["encoder_output"].shape[1:] == mlp_mixer.output_shape()


@pytest.mark.parametrize("image_size,num_channels", [(224, 3)])
@pytest.mark.parametrize("use_pretrained", [True, False])
def test_vit_encoder(image_size: int, num_channels: int, use_pretrained: bool):
    vit = ViTEncoder(height=image_size, width=image_size, num_channels=num_channels, use_pretrained=use_pretrained)
    inputs = torch.rand(2, num_channels, image_size, image_size)
    outputs = vit(inputs)
    assert outputs["encoder_output"].shape[1:] == vit.output_shape()
