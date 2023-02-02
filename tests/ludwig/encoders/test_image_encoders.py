from typing import Union

import pytest
import torch

from ludwig.encoders.image.base import MLPMixerEncoder, ResNetEncoder, Stacked2DCNN, ViTEncoder
from ludwig.encoders.image.torchvision import (
    TVAlexNetEncoder,
    TVConvNeXtEncoder,
    TVDenseNetEncoder,
    TVEfficientNetEncoder,
    TVGoogLeNetEncoder,
    TVInceptionV3Encoder,
    TVMaxVitEncoder,
    TVMNASNetEncoder,
    TVMobileNetV2Encoder,
    TVMobileNetV3Encoder,
    TVRegNetEncoder,
    TVResNetEncoder,
    TVResNeXtEncoder,
    TVShuffleNetV2Encoder,
    TVSqueezeNetEncoder,
    TVSwinTransformerEncoder,
    TVVGGEncoder,
    TVViTEncoder,
    TVWideResNetEncoder,
)
from ludwig.utils.image_utils import torchvision_model_registry
from ludwig.utils.misc_utils import set_random_seed
from tests.integration_tests.parameter_update_utils import check_module_parameters_updated

RANDOM_SEED = 1919


@pytest.mark.parametrize("height,width,num_conv_layers,num_channels", [(224, 224, 5, 3)])
def test_stacked2d_cnn(height: int, width: int, num_conv_layers: int, num_channels: int):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    stacked_2d_cnn = Stacked2DCNN(
        height=height, width=width, num_conv_layers=num_conv_layers, num_channels=num_channels
    )
    inputs = torch.rand(2, num_channels, height, width)
    outputs = stacked_2d_cnn(inputs)
    assert outputs["encoder_output"].shape[1:] == stacked_2d_cnn.output_shape

    # check for parameter updating
    target = torch.randn(outputs["encoder_output"].shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(stacked_2d_cnn, (inputs,), target)

    assert tpc == upc, f"Not all expected parameters updated.  Parameters not updated {not_updated}."


@pytest.mark.parametrize("height,width,num_channels", [(224, 224, 1), (224, 224, 3)])
def test_resnet_encoder(height: int, width: int, num_channels: int):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    resnet = ResNetEncoder(height=height, width=width, num_channels=num_channels)
    inputs = torch.rand(2, num_channels, height, width)
    outputs = resnet(inputs)
    assert outputs["encoder_output"].shape[1:] == resnet.output_shape

    # check for parameter updating
    target = torch.randn(outputs["encoder_output"].shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(resnet, (inputs,), target)

    assert tpc == upc, f"Not all expected parameters updated.  Parameters not updated {not_updated}."


@pytest.mark.parametrize("height,width,num_channels", [(224, 224, 3)])
def test_mlp_mixer_encoder(height: int, width: int, num_channels: int):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    mlp_mixer = MLPMixerEncoder(height=height, width=width, num_channels=num_channels)
    inputs = torch.rand(2, num_channels, height, width)
    outputs = mlp_mixer(inputs)
    assert outputs["encoder_output"].shape[1:] == mlp_mixer.output_shape

    # check for parameter updating
    target = torch.randn(outputs["encoder_output"].shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(mlp_mixer, (inputs,), target)

    assert tpc == upc, f"Not all expected parameters updated.  Parameters not updated {not_updated}."


@pytest.mark.parametrize("image_size,num_channels", [(224, 3)])
@pytest.mark.parametrize("use_pretrained", [True, False])
def test_vit_encoder(image_size: int, num_channels: int, use_pretrained: bool):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    vit = ViTEncoder(
        height=image_size,
        width=image_size,
        num_channels=num_channels,
        use_pretrained=use_pretrained,
        output_attentions=True,
    )
    inputs = torch.rand(2, num_channels, image_size, image_size)
    outputs = vit(inputs)
    assert outputs["encoder_output"].shape[1:] == vit.output_shape
    config = vit.transformer.module.config
    num_patches = (224 // config.patch_size) ** 2 + 1  # patches of the image + cls_token
    attentions = outputs["attentions"]
    assert len(attentions) == config.num_hidden_layers
    assert attentions[0].shape == torch.Size([2, config.num_attention_heads, num_patches, num_patches])

    # check for parameter updating
    target = torch.randn(outputs["encoder_output"].shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(vit, (inputs,), target)

    assert tpc == upc, f"Not all expected parameters updated.  Parameters not updated {not_updated}."


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("saved_weights_in_checkpoint", [True, False])
@pytest.mark.parametrize(
    "use_pretrained",
    [
        False,
    ],
)
@pytest.mark.parametrize("model_variant", [v.variant_id for v in torchvision_model_registry["alexnet"].values()])
def test_tv_alexnet_encoder(
    model_variant: int,
    use_pretrained: bool,
    saved_weights_in_checkpoint: bool,
    trainable: bool,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVAlexNetEncoder(
        model_variant=model_variant,
        use_pretrained=use_pretrained,
        saved_weights_in_checkpoint=saved_weights_in_checkpoint,
        trainable=trainable,
    )
    inputs = torch.rand(2, *pretrained_model.input_shape)
    outputs = pretrained_model(inputs)
    assert outputs["encoder_output"].shape[1:] == pretrained_model.output_shape


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("saved_weights_in_checkpoint", [True, False])
@pytest.mark.parametrize(
    "use_pretrained",
    [
        False,
    ],
)
@pytest.mark.parametrize("model_variant", [v.variant_id for v in torchvision_model_registry["convnext"].values()])
def test_tv_convnext_encoder(
    model_variant: int,
    use_pretrained: bool,
    saved_weights_in_checkpoint: bool,
    trainable: bool,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVConvNeXtEncoder(
        model_variant=model_variant,
        use_pretrained=use_pretrained,
        saved_weights_in_checkpoint=saved_weights_in_checkpoint,
        trainable=trainable,
    )
    inputs = torch.rand(2, *pretrained_model.input_shape)
    outputs = pretrained_model(inputs)
    assert outputs["encoder_output"].shape[1:] == pretrained_model.output_shape


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("saved_weights_in_checkpoint", [True, False])
@pytest.mark.parametrize(
    "use_pretrained",
    [
        False,
    ],
)
@pytest.mark.parametrize("model_variant", [v.variant_id for v in torchvision_model_registry["densenet"].values()])
def test_tv_densenet_encoder(
    model_variant: int,
    use_pretrained: bool,
    saved_weights_in_checkpoint: bool,
    trainable: bool,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVDenseNetEncoder(
        model_variant=model_variant,
        use_pretrained=use_pretrained,
        saved_weights_in_checkpoint=saved_weights_in_checkpoint,
        trainable=trainable,
    )
    inputs = torch.rand(2, *pretrained_model.input_shape)
    outputs = pretrained_model(inputs)
    assert outputs["encoder_output"].shape[1:] == pretrained_model.output_shape


# test only model variants that do not require large amount of memory
LOW_MEMORY_EFFICIENTNET_VARIANTS = set(torchvision_model_registry["efficientnet"].keys()) - {"b6", "b7"}


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("saved_weights_in_checkpoint", [True, False])
@pytest.mark.parametrize(
    "use_pretrained",
    [
        False,
    ],
)
@pytest.mark.parametrize("model_variant", LOW_MEMORY_EFFICIENTNET_VARIANTS)
def test_tv_efficientnet_encoder(
    model_variant: int,
    use_pretrained: bool,
    saved_weights_in_checkpoint: bool,
    trainable: bool,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVEfficientNetEncoder(
        model_variant=model_variant,
        use_pretrained=use_pretrained,
        saved_weights_in_checkpoint=saved_weights_in_checkpoint,
        trainable=trainable,
    )
    inputs = torch.rand(2, *pretrained_model.input_shape)
    outputs = pretrained_model(inputs)
    assert outputs["encoder_output"].shape[1:] == pretrained_model.output_shape


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("saved_weights_in_checkpoint", [True, False])
@pytest.mark.parametrize(
    "use_pretrained",
    [
        False,
    ],
)
@pytest.mark.parametrize("model_variant", [v.variant_id for v in torchvision_model_registry["googlenet"].values()])
def test_tv_googlenet_encoder(
    model_variant: int,
    use_pretrained: bool,
    saved_weights_in_checkpoint: bool,
    trainable: bool,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVGoogLeNetEncoder(
        model_variant=model_variant,
        use_pretrained=use_pretrained,
        saved_weights_in_checkpoint=saved_weights_in_checkpoint,
        trainable=trainable,
    )
    inputs = torch.rand(2, *pretrained_model.input_shape)
    outputs = pretrained_model(inputs)
    assert outputs["encoder_output"].shape[1:] == pretrained_model.output_shape


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("saved_weights_in_checkpoint", [True, False])
@pytest.mark.parametrize(
    "use_pretrained",
    [
        False,
    ],
)
@pytest.mark.parametrize("model_variant", [v.variant_id for v in torchvision_model_registry["inceptionv3"].values()])
def test_tv_inceptionv3_encoder(
    model_variant: int,
    use_pretrained: bool,
    saved_weights_in_checkpoint: bool,
    trainable: bool,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVInceptionV3Encoder(
        model_variant=model_variant,
        use_pretrained=use_pretrained,
        saved_weights_in_checkpoint=saved_weights_in_checkpoint,
        trainable=trainable,
    )
    inputs = torch.rand(2, *pretrained_model.input_shape)
    outputs = pretrained_model(inputs)
    assert outputs["encoder_output"].shape[1:] == pretrained_model.output_shape


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("saved_weights_in_checkpoint", [True, False])
@pytest.mark.parametrize(
    "use_pretrained",
    [
        False,
    ],
)
@pytest.mark.parametrize("model_variant", [v.variant_id for v in torchvision_model_registry["maxvit"].values()])
def test_tv_maxvit_encoder(
    model_variant: int,
    use_pretrained: bool,
    saved_weights_in_checkpoint: bool,
    trainable: bool,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVMaxVitEncoder(
        model_variant=model_variant,
        use_pretrained=use_pretrained,
        saved_weights_in_checkpoint=saved_weights_in_checkpoint,
        trainable=trainable,
    )
    inputs = torch.rand(2, *pretrained_model.input_shape)
    outputs = pretrained_model(inputs)
    assert outputs["encoder_output"].shape[1:] == pretrained_model.output_shape


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("saved_weights_in_checkpoint", [True, False])
@pytest.mark.parametrize(
    "use_pretrained",
    [
        False,
    ],
)
@pytest.mark.parametrize("model_variant", [v.variant_id for v in torchvision_model_registry["mnasnet"].values()])
def test_tv_mnasnet_encoder(
    model_variant: int,
    use_pretrained: bool,
    saved_weights_in_checkpoint: bool,
    trainable: bool,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVMNASNetEncoder(
        model_variant=model_variant,
        use_pretrained=use_pretrained,
        saved_weights_in_checkpoint=saved_weights_in_checkpoint,
        trainable=trainable,
    )
    inputs = torch.rand(2, *pretrained_model.input_shape)
    outputs = pretrained_model(inputs)
    assert outputs["encoder_output"].shape[1:] == pretrained_model.output_shape


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("saved_weights_in_checkpoint", [True, False])
@pytest.mark.parametrize(
    "use_pretrained",
    [
        False,
    ],
)
@pytest.mark.parametrize("model_variant", [v.variant_id for v in torchvision_model_registry["mobilenetv2"].values()])
def test_tv_mobilenetv2_encoder(
    model_variant: int,
    use_pretrained: bool,
    saved_weights_in_checkpoint: bool,
    trainable: bool,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVMobileNetV2Encoder(
        model_variant=model_variant,
        use_pretrained=use_pretrained,
        saved_weights_in_checkpoint=saved_weights_in_checkpoint,
        trainable=trainable,
    )
    inputs = torch.rand(2, *pretrained_model.input_shape)
    outputs = pretrained_model(inputs)
    assert outputs["encoder_output"].shape[1:] == pretrained_model.output_shape


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("saved_weights_in_checkpoint", [True, False])
@pytest.mark.parametrize(
    "use_pretrained",
    [
        False,
    ],
)
@pytest.mark.parametrize("model_variant", [v.variant_id for v in torchvision_model_registry["mobilenetv3"].values()])
def test_tv_mobilenetv3_encoder(
    model_variant: int,
    use_pretrained: bool,
    saved_weights_in_checkpoint: bool,
    trainable: bool,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVMobileNetV3Encoder(
        model_variant=model_variant,
        use_pretrained=use_pretrained,
        saved_weights_in_checkpoint=saved_weights_in_checkpoint,
        trainable=trainable,
    )
    inputs = torch.rand(2, *pretrained_model.input_shape)
    outputs = pretrained_model(inputs)
    assert outputs["encoder_output"].shape[1:] == pretrained_model.output_shape


# test only model variants that do not require large amount of memory
LOW_MEMORY_REGNET_VARIANTS = set(torchvision_model_registry["regnet"].keys()) - {"y_128gf"}


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("saved_weights_in_checkpoint", [True, False])
@pytest.mark.parametrize(
    "use_pretrained",
    [
        False,
    ],
)
@pytest.mark.parametrize("model_variant", LOW_MEMORY_REGNET_VARIANTS)
def test_tv_regnet_encoder(
    model_variant: int,
    use_pretrained: bool,
    saved_weights_in_checkpoint: bool,
    trainable: bool,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVRegNetEncoder(
        model_variant=model_variant,
        use_pretrained=use_pretrained,
        saved_weights_in_checkpoint=saved_weights_in_checkpoint,
        trainable=trainable,
    )
    inputs = torch.rand(2, *pretrained_model.input_shape)
    outputs = pretrained_model(inputs)
    assert outputs["encoder_output"].shape[1:] == pretrained_model.output_shape


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("saved_weights_in_checkpoint", [True, False])
@pytest.mark.parametrize(
    "use_pretrained",
    [
        False,
    ],
)
@pytest.mark.parametrize("model_variant", [v.variant_id for v in torchvision_model_registry["resnet"].values()])
def test_tv_resnet_torch_encoder(
    model_variant: int,
    use_pretrained: bool,
    saved_weights_in_checkpoint: bool,
    trainable: bool,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVResNetEncoder(
        model_variant=model_variant,
        use_pretrained=use_pretrained,
        saved_weights_in_checkpoint=saved_weights_in_checkpoint,
        trainable=trainable,
    )
    inputs = torch.rand(2, *pretrained_model.input_shape)
    outputs = pretrained_model(inputs)
    assert outputs["encoder_output"].shape[1:] == pretrained_model.output_shape


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("saved_weights_in_checkpoint", [True, False])
@pytest.mark.parametrize(
    "use_pretrained",
    [
        False,
    ],
)
@pytest.mark.parametrize("model_variant", [v.variant_id for v in torchvision_model_registry["resnext"].values()])
def test_tv_resnext_encoder(
    model_variant: int,
    use_pretrained: bool,
    saved_weights_in_checkpoint: bool,
    trainable: bool,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVResNeXtEncoder(
        model_variant=model_variant,
        use_pretrained=use_pretrained,
        saved_weights_in_checkpoint=saved_weights_in_checkpoint,
        trainable=trainable,
    )
    inputs = torch.rand(2, *pretrained_model.input_shape)
    outputs = pretrained_model(inputs)
    assert outputs["encoder_output"].shape[1:] == pretrained_model.output_shape


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("saved_weights_in_checkpoint", [True, False])
@pytest.mark.parametrize(
    "use_pretrained",
    [
        False,
    ],
)
@pytest.mark.parametrize("model_variant", [v.variant_id for v in torchvision_model_registry["shufflenet_v2"].values()])
def test_tv_shufflenet_v2_encoder(
    model_variant: str,
    use_pretrained: bool,
    saved_weights_in_checkpoint: bool,
    trainable: bool,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVShuffleNetV2Encoder(
        model_variant=model_variant,
        use_pretrained=use_pretrained,
        saved_weights_in_checkpoint=saved_weights_in_checkpoint,
        trainable=trainable,
    )
    inputs = torch.rand(2, *pretrained_model.input_shape)
    outputs = pretrained_model(inputs)
    assert outputs["encoder_output"].shape[1:] == pretrained_model.output_shape


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("saved_weights_in_checkpoint", [True, False])
@pytest.mark.parametrize(
    "use_pretrained",
    [
        False,
    ],
)
@pytest.mark.parametrize("model_variant", [v.variant_id for v in torchvision_model_registry["squeezenet"].values()])
def test_tv_squeezenet_encoder(
    model_variant: str,
    use_pretrained: bool,
    saved_weights_in_checkpoint: bool,
    trainable: bool,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVSqueezeNetEncoder(
        model_variant=model_variant,
        use_pretrained=use_pretrained,
        saved_weights_in_checkpoint=saved_weights_in_checkpoint,
        trainable=trainable,
    )
    inputs = torch.rand(2, *pretrained_model.input_shape)
    outputs = pretrained_model(inputs)
    assert outputs["encoder_output"].shape[1:] == pretrained_model.output_shape


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("saved_weights_in_checkpoint", [True, False])
@pytest.mark.parametrize(
    "use_pretrained",
    [
        False,
    ],
)
@pytest.mark.parametrize(
    "model_variant", [v.variant_id for v in torchvision_model_registry["swin_transformer"].values()]
)
def test_tv_swin_transformer_encoder(
    model_variant: str,
    use_pretrained: bool,
    saved_weights_in_checkpoint: bool,
    trainable: bool,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVSwinTransformerEncoder(
        model_variant=model_variant,
        use_pretrained=use_pretrained,
        saved_weights_in_checkpoint=saved_weights_in_checkpoint,
        trainable=trainable,
    )
    inputs = torch.rand(2, *pretrained_model.input_shape)
    outputs = pretrained_model(inputs)
    assert outputs["encoder_output"].shape[1:] == pretrained_model.output_shape


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("saved_weights_in_checkpoint", [True, False])
@pytest.mark.parametrize(
    "use_pretrained",
    [
        False,
    ],
)
@pytest.mark.parametrize("model_variant", [v.variant_id for v in torchvision_model_registry["vgg"].values()])
def test_tv_vgg_encoder(
    model_variant: Union[int, str],
    use_pretrained: bool,
    saved_weights_in_checkpoint: bool,
    trainable: bool,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVVGGEncoder(
        model_variant=model_variant,
        use_pretrained=use_pretrained,
        saved_weights_in_checkpoint=saved_weights_in_checkpoint,
        trainable=trainable,
    )
    inputs = torch.rand(2, *pretrained_model.input_shape)
    outputs = pretrained_model(inputs)
    assert outputs["encoder_output"].shape[1:] == pretrained_model.output_shape


# test only VIT model variants that do not require large amount of memory
LOW_MEMORY_VIT_VARIANTS = set(torchvision_model_registry["vit"].keys()) - {"h_14"}


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("saved_weights_in_checkpoint", [True, False])
@pytest.mark.parametrize(
    "use_pretrained",
    [
        False,
    ],
)
@pytest.mark.parametrize("model_variant", LOW_MEMORY_VIT_VARIANTS)
def test_tv_vit_encoder(
    model_variant: str,
    use_pretrained: bool,
    saved_weights_in_checkpoint: bool,
    trainable: bool,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVViTEncoder(
        model_variant=model_variant,
        use_pretrained=use_pretrained,
        saved_weights_in_checkpoint=saved_weights_in_checkpoint,
        trainable=trainable,
    )
    inputs = torch.rand(2, *pretrained_model.input_shape)
    outputs = pretrained_model(inputs)
    assert outputs["encoder_output"].shape[1:] == pretrained_model.output_shape


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("saved_weights_in_checkpoint", [True, False])
@pytest.mark.parametrize(
    "use_pretrained",
    [
        False,
    ],
)
@pytest.mark.parametrize("model_variant", [v.variant_id for v in torchvision_model_registry["wide_resnet"].values()])
def test_tv_wide_resnet_encoder(
    model_variant: str,
    use_pretrained: bool,
    saved_weights_in_checkpoint: bool,
    trainable: bool,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVWideResNetEncoder(
        model_variant=model_variant,
        use_pretrained=use_pretrained,
        saved_weights_in_checkpoint=saved_weights_in_checkpoint,
        trainable=trainable,
    )
    inputs = torch.rand(2, *pretrained_model.input_shape)
    outputs = pretrained_model(inputs)
    assert outputs["encoder_output"].shape[1:] == pretrained_model.output_shape
