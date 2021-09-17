import pytest

import torch

from ludwig.encoders.image_encoders import Stacked2DCNN, ResNetEncoder,\
    MLPMixerEncoder


@pytest.mark.parametrize(
    'img_height,img_width,num_conv_layers,first_in_channels', [(224, 224, 5, 3)]
)
def test_stacked2d_cnn(
        img_height: int,
        img_width: int,
        num_conv_layers: int,
        first_in_channels: int
):
    
    stacked_2d_cnn = Stacked2DCNN(
        img_height, img_width, num_conv_layers=num_conv_layers, 
        first_in_channels=first_in_channels)
    inputs = torch.rand(2, first_in_channels, img_height, img_width)
    outputs = stacked_2d_cnn(inputs)
    assert outputs['encoder_output'].shape[1:] == stacked_2d_cnn.output_shape


@pytest.mark.parametrize('img_height,img_width', [(224, 224)])
def test_resnet_encoder(img_height: int, img_width: int):
    resnet = ResNetEncoder(img_height, img_width)
    inputs = torch.rand(2, 3, img_height, img_width)
    outputs = resnet(inputs)
    assert outputs['encoder_output'].shape[1:] == resnet.output_shape


@pytest.mark.parametrize('img_height,img_width,in_channels', [(224, 224, 3)])
def test_mlp_mixer_encoder(img_height: int, img_width: int, in_channels:int):
    mlp_mixer = MLPMixerEncoder(img_height, img_width, in_channels)
    inputs = torch.rand(2, in_channels, img_height, img_width)
    outputs = mlp_mixer(inputs)
    assert outputs['encoder_output'].shape[1:] == mlp_mixer.output_shape
