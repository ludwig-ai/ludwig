import pytest

from ludwig.modules.mlp_mixer_modules import MixerBlock, MLP, MLPMixer

from .test_utils import assert_output_shapes


@pytest.mark.parametrize("in_features,hidden_size,out_features", [(3, 8, 8), (8, 64, 32)])
def test_mlp(in_features: int, hidden_size: int, out_features: int):
    assert_output_shapes(module=MLP(in_features, hidden_size, out_features), input_shape=(in_features,))


@pytest.mark.parametrize("embed_size,n_patches,token_dim,channel_dim", [(512, 49, 2048, 256)])
def test_mixer_block(
    embed_size: int,
    n_patches: int,
    token_dim: int,
    channel_dim: int,
):
    assert_output_shapes(
        module=MixerBlock(embed_size, n_patches, token_dim, channel_dim), input_shape=(n_patches, embed_size)
    )


@pytest.mark.parametrize("img_height,img_width,in_channels", [(224, 224, 3)])
def test_mlp_mixer(img_height: int, img_width: int, in_channels: int):
    assert_output_shapes(module=MLPMixer(img_height, img_width, in_channels), input_shape=(3, img_height, img_width))
