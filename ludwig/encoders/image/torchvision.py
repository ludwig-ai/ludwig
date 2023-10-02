import logging
import os
from abc import abstractmethod
from typing import Optional, Type, Union

import torch
import torchvision.models as tvm

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ENCODER_OUTPUT, IMAGE
from ludwig.encoders.image.base import ImageEncoder
from ludwig.encoders.registry import register_encoder
from ludwig.encoders.types import EncoderOutputDict
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.image.torchvision import (
    TVAlexNetEncoderConfig,
    TVConvNeXtEncoderConfig,
    TVDenseNetEncoderConfig,
    TVEfficientNetEncoderConfig,
    TVGoogLeNetEncoderConfig,
    TVInceptionV3EncoderConfig,
    TVMaxVitEncoderConfig,
    TVMNASNetEncoderConfig,
    TVMobileNetV2EncoderConfig,
    TVMobileNetV3EncoderConfig,
    TVRegNetEncoderConfig,
    TVResNetEncoderConfig,
    TVResNeXtEncoderConfig,
    TVShuffleNetV2EncoderConfig,
    TVSqueezeNetEncoderConfig,
    TVSwinTransformerEncoderConfig,
    TVVGGEncoderConfig,
    TVViTEncoderConfig,
    TVWideResNetEncoderConfig,
)
from ludwig.utils.image_utils import register_torchvision_model_variants, torchvision_model_registry, TVModelVariant

logger = logging.getLogger(__name__)


@DeveloperAPI
class TVBaseEncoder(ImageEncoder):
    def __init__(
        self,
        model_variant: Union[str, int] = None,
        use_pretrained: bool = True,
        saved_weights_in_checkpoint: bool = False,
        model_cache_dir: Optional[str] = None,
        trainable: bool = True,
        **kwargs,
    ):
        super().__init__()

        logger.debug(f" {self.name}")
        # map parameter input feature config names to internal names
        self.model_variant = model_variant
        self.use_pretrained = use_pretrained
        self.model_cache_dir = model_cache_dir

        # remove any Ludwig specific keyword parameters
        kwargs.pop("encoder_config", None)
        kwargs.pop("type", None)
        kwargs.pop("skip", None)

        # cache pre-trained models if requested
        # based on https://github.com/pytorch/vision/issues/616#issuecomment-428637564
        if self.model_cache_dir is not None:
            os.environ["TORCH_HOME"] = self.model_cache_dir

        # retrieve function to create requested model
        self.create_model = torchvision_model_registry[self.torchvision_model_type][
            self.model_variant
        ].create_model_function

        # get weight specification
        if use_pretrained and not saved_weights_in_checkpoint:
            weights_specification = torchvision_model_registry[self.torchvision_model_type][
                self.model_variant
            ].model_weights.DEFAULT
            logger.info(
                f"Instantiating torchvision image encoder '{self.torchvision_model_type}' with pretrained weights: "
                f"{weights_specification}."
            )
        else:
            weights_specification = None
            if saved_weights_in_checkpoint:
                logger.info(
                    f"Instantiating torchvision image encoder: '{self.torchvision_model_type}' "
                    "with weights saved in the checkpoint."
                )
            else:
                logger.info(
                    f"Instantiating torchvision image encoder: '{self.torchvision_model_type}' "
                    "with no pretrained weights."
                )

        # get torchvision transforms object
        transforms_obj = torchvision_model_registry[self.torchvision_model_type][
            self.model_variant
        ].model_weights.DEFAULT.transforms()

        # capture key attributes from torchvision transform for later use
        self.num_channels = len(transforms_obj.mean)
        self.normalize_mean = transforms_obj.mean
        self.normalize_std = transforms_obj.std
        self.crop_size = transforms_obj.crop_size

        logger.debug(f"  {self.torchvision_model_type}")
        # create pretrained model with pretrained weights or None for untrained model
        self.model = self.create_model(weights=weights_specification, **kwargs)

        # remove final classification layer
        self._remove_softmax_layer()

        # freeze parameters if requested
        for p in self.model.parameters():
            p.requires_grad_(trainable)

    def forward(self, inputs: torch.Tensor) -> EncoderOutputDict:
        return {ENCODER_OUTPUT: self.model(inputs)}

    @abstractmethod
    def _remove_softmax_layer(self):
        """Model specific method that allows the final softmax layer to be implemented in the Ludwig Decoder
        component.  The model specific implementation should change the final softmax layer in the torchvision
        model architecture to torch.nn.Identity().  This allows the output tensor from the preceding layer to be
        passed to the Ludwig Combiner and then to the Decoder.

        Returns: None
        """
        raise NotImplementedError()

    @property
    def output_shape(self) -> torch.Size:
        # create synthetic image and run through forward method
        inputs = torch.randn([1, *self.input_shape])
        output = self.model(inputs)
        return torch.Size(output.shape[1:])

    @property
    def input_shape(self) -> torch.Size:
        # expected shape after all pre-processing
        # len(transforms_obj.mean) determines the number of channels
        # transforms_obj.crop_size determines the height and width of image
        # [num_channels, height, width]
        return torch.Size([self.num_channels, *(2 * self.crop_size)])


@DeveloperAPI
@register_torchvision_model_variants(
    [
        TVModelVariant(variant_id="base", create_model_function=tvm.alexnet, model_weights=tvm.AlexNet_Weights),
    ]
)
@register_encoder("alexnet", IMAGE)
class TVAlexNetEncoder(TVBaseEncoder):
    # specify base model type
    torchvision_model_type: str = "alexnet"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")
        super().__init__(**kwargs)

    # TODO: discussion w/ justin
    # @property
    # def get_torchvision_model_type(self):
    #     return "alexnet"

    def _remove_softmax_layer(self):
        self.model.classifier[-1] = torch.nn.Identity()

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TVAlexNetEncoderConfig


@DeveloperAPI
@register_torchvision_model_variants(
    [
        TVModelVariant(
            variant_id="tiny", create_model_function=tvm.convnext_tiny, model_weights=tvm.ConvNeXt_Tiny_Weights
        ),
        TVModelVariant(
            variant_id="small", create_model_function=tvm.convnext_small, model_weights=tvm.ConvNeXt_Small_Weights
        ),
        TVModelVariant(
            variant_id="base", create_model_function=tvm.convnext_base, model_weights=tvm.ConvNeXt_Base_Weights
        ),
        TVModelVariant(
            variant_id="large", create_model_function=tvm.convnext_large, model_weights=tvm.ConvNeXt_Large_Weights
        ),
    ]
)
@register_encoder("convnext", IMAGE)
class TVConvNeXtEncoder(TVBaseEncoder):
    # specify base torchvision model
    torchvision_model_type: str = "convnext"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")
        super().__init__(**kwargs)

    def _remove_softmax_layer(self) -> None:
        self.model.classifier[-1] = torch.nn.Identity()

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TVConvNeXtEncoderConfig


@DeveloperAPI
@register_torchvision_model_variants(
    [
        TVModelVariant(121, tvm.densenet121, tvm.DenseNet121_Weights),
        TVModelVariant(161, tvm.densenet161, tvm.DenseNet161_Weights),
        TVModelVariant(169, tvm.densenet169, tvm.DenseNet169_Weights),
        TVModelVariant(201, tvm.densenet201, tvm.DenseNet201_Weights),
    ]
)
@register_encoder("densenet", IMAGE)
class TVDenseNetEncoder(TVBaseEncoder):
    # specify base torchvision model
    torchvision_model_type: str = "densenet"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")
        super().__init__(**kwargs)

    def _remove_softmax_layer(self) -> None:
        self.model.classifier = torch.nn.Identity()

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TVDenseNetEncoderConfig


@DeveloperAPI
@register_torchvision_model_variants(
    [
        TVModelVariant("b0", tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights),
        TVModelVariant("b1", tvm.efficientnet_b1, tvm.EfficientNet_B1_Weights),
        TVModelVariant("b2", tvm.efficientnet_b2, tvm.EfficientNet_B2_Weights),
        TVModelVariant("b3", tvm.efficientnet_b3, tvm.EfficientNet_B3_Weights),
        TVModelVariant("b4", tvm.efficientnet_b4, tvm.EfficientNet_B4_Weights),
        TVModelVariant("b5", tvm.efficientnet_b5, tvm.EfficientNet_B5_Weights),
        TVModelVariant("b6", tvm.efficientnet_b6, tvm.EfficientNet_B6_Weights),
        TVModelVariant("b7", tvm.efficientnet_b7, tvm.EfficientNet_B7_Weights),
        TVModelVariant("v2_s", tvm.efficientnet_v2_s, tvm.EfficientNet_V2_S_Weights),
        TVModelVariant("v2_m", tvm.efficientnet_v2_m, tvm.EfficientNet_V2_M_Weights),
        TVModelVariant("v2_l", tvm.efficientnet_v2_l, tvm.EfficientNet_V2_L_Weights),
    ]
)
@register_encoder("efficientnet", IMAGE)
class TVEfficientNetEncoder(TVBaseEncoder):
    # specify base torchvision model
    torchvision_model_type: str = "efficientnet"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")
        super().__init__(**kwargs)

    def _remove_softmax_layer(self) -> None:
        self.model.classifier[-1] = torch.nn.Identity()

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TVEfficientNetEncoderConfig


@DeveloperAPI
@register_torchvision_model_variants(
    [
        TVModelVariant("base", tvm.googlenet, tvm.GoogLeNet_Weights),
    ]
)
@register_encoder("googlenet", IMAGE)
class TVGoogLeNetEncoder(TVBaseEncoder):
    # specify base torchvision model
    torchvision_model_type: str = "googlenet"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")
        super().__init__(**kwargs)

        # if auxiliary network exists, eliminate auxiliary network
        # to resolve issue when loading a saved model which does not
        # contain the auxiliary network
        if self.model.aux_logits:
            self.model.aux_logits = False
            self.model.aux1 = None
            self.model.aux2 = None

    def _remove_softmax_layer(self) -> None:
        self.model.fc = torch.nn.Identity()

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TVGoogLeNetEncoderConfig


@DeveloperAPI
@register_torchvision_model_variants(
    [
        TVModelVariant("base", tvm.inception_v3, tvm.Inception_V3_Weights),
    ]
)
@register_encoder("inceptionv3", IMAGE)
class TVInceptionV3Encoder(TVBaseEncoder):
    # specify base torchvision model
    torchvision_model_type: str = "inceptionv3"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")
        super().__init__(**kwargs)

        # if auxiliary network exists, eliminate auxiliary network
        # to resolve issue when loading a saved model which does not
        # contain the auxiliary network
        if self.model.aux_logits:
            self.model.aux_logits = False
            self.model.AuxLogits = None

    def _remove_softmax_layer(self) -> None:
        self.model.fc = torch.nn.Identity()

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TVInceptionV3EncoderConfig


@DeveloperAPI
@register_torchvision_model_variants(
    [
        TVModelVariant("t", tvm.maxvit_t, tvm.MaxVit_T_Weights),
    ]
)
@register_encoder("maxvit", IMAGE)
class TVMaxVitEncoder(TVBaseEncoder):
    # specify base torchvision model
    torchvision_model_type: str = "maxvit"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")
        super().__init__(**kwargs)

    def _remove_softmax_layer(self) -> None:
        self.model.classifier[-1] = torch.nn.Identity()

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TVMaxVitEncoderConfig


@DeveloperAPI
@register_torchvision_model_variants(
    [
        TVModelVariant("0_5", tvm.mnasnet0_5, tvm.mnasnet.MNASNet0_5_Weights),
        TVModelVariant("0_75", tvm.mnasnet0_75, tvm.mnasnet.MNASNet0_75_Weights),
        TVModelVariant("1_0", tvm.mnasnet1_0, tvm.mnasnet.MNASNet1_0_Weights),
        TVModelVariant("1_3", tvm.mnasnet1_3, tvm.mnasnet.MNASNet1_3_Weights),
    ]
)
@register_encoder("mnasnet", IMAGE)
class TVMNASNetEncoder(TVBaseEncoder):
    # specify base torchvision model
    torchvision_model_type: str = "mnasnet"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")
        super().__init__(**kwargs)

    def _remove_softmax_layer(self) -> None:
        self.model.classifier[-1] = torch.nn.Identity()

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TVMNASNetEncoderConfig


@DeveloperAPI
@register_torchvision_model_variants(
    [
        TVModelVariant("base", tvm.mobilenet_v2, tvm.MobileNet_V2_Weights),
    ]
)
@register_encoder("mobilenetv2", IMAGE)
class TVMobileNetV2Encoder(TVBaseEncoder):
    # specify base torchvision model
    torchvision_model_type: str = "mobilenetv2"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")
        super().__init__(**kwargs)

    def _remove_softmax_layer(self) -> None:
        self.model.classifier[-1] = torch.nn.Identity()

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TVMobileNetV2EncoderConfig


@DeveloperAPI
@register_torchvision_model_variants(
    [
        TVModelVariant("small", tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights),
        TVModelVariant("large", tvm.mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights),
    ]
)
@register_encoder("mobilenetv3", IMAGE)
class TVMobileNetV3Encoder(TVBaseEncoder):
    # specify base torchvision model
    torchvision_model_type: str = "mobilenetv3"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")
        super().__init__(**kwargs)

    def _remove_softmax_layer(self) -> None:
        self.model.classifier[-1] = torch.nn.Identity()

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TVMobileNetV3EncoderConfig


@DeveloperAPI
@register_torchvision_model_variants(
    [
        TVModelVariant("x_16gf", tvm.regnet_x_16gf, tvm.RegNet_X_16GF_Weights),
        TVModelVariant("x_1_6gf", tvm.regnet_x_1_6gf, tvm.RegNet_X_1_6GF_Weights),
        TVModelVariant("x_32gf", tvm.regnet_x_32gf, tvm.RegNet_X_32GF_Weights),
        TVModelVariant("x_3_2gf", tvm.regnet_x_3_2gf, tvm.RegNet_X_3_2GF_Weights),
        TVModelVariant("x_400mf", tvm.regnet_x_400mf, tvm.RegNet_X_400MF_Weights),
        TVModelVariant("x_800mf", tvm.regnet_x_800mf, tvm.RegNet_X_800MF_Weights),
        TVModelVariant("x_8gf", tvm.regnet_x_8gf, tvm.RegNet_X_8GF_Weights),
        TVModelVariant("y_128gf", tvm.regnet_y_128gf, tvm.RegNet_Y_128GF_Weights),
        TVModelVariant("y_16gf", tvm.regnet_y_16gf, tvm.RegNet_Y_16GF_Weights),
        TVModelVariant("y_1_6gf", tvm.regnet_y_1_6gf, tvm.RegNet_Y_1_6GF_Weights),
        TVModelVariant("y_32gf", tvm.regnet_y_32gf, tvm.RegNet_Y_32GF_Weights),
        TVModelVariant("y_3_2gf", tvm.regnet_y_3_2gf, tvm.RegNet_Y_3_2GF_Weights),
        TVModelVariant("y_400mf", tvm.regnet_y_400mf, tvm.RegNet_Y_400MF_Weights),
        TVModelVariant("y_800mf", tvm.regnet_y_800mf, tvm.RegNet_Y_800MF_Weights),
        TVModelVariant("y_8gf", tvm.regnet_y_8gf, tvm.RegNet_Y_8GF_Weights),
    ]
)
@register_encoder("regnet", IMAGE)
class TVRegNetEncoder(TVBaseEncoder):
    # specify base torchvision model
    torchvision_model_type: str = "regnet"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")
        super().__init__(**kwargs)

    def _remove_softmax_layer(self) -> None:
        self.model.fc = torch.nn.Identity()

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TVRegNetEncoderConfig


@DeveloperAPI
@register_torchvision_model_variants(
    [
        TVModelVariant(18, tvm.resnet18, tvm.ResNet18_Weights),
        TVModelVariant(34, tvm.resnet34, tvm.ResNet34_Weights),
        TVModelVariant(50, tvm.resnet50, tvm.ResNet50_Weights),
        TVModelVariant(101, tvm.resnet101, tvm.ResNet101_Weights),
        TVModelVariant(152, tvm.resnet152, tvm.ResNet152_Weights),
    ]
)
@register_encoder("resnet", IMAGE)
class TVResNetEncoder(TVBaseEncoder):
    # specify base torchvision model
    torchvision_model_type: str = "resnet"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")
        super().__init__(**kwargs)

    def _remove_softmax_layer(self) -> None:
        self.model.fc = torch.nn.Identity()

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TVResNetEncoderConfig


@DeveloperAPI
@register_torchvision_model_variants(
    [
        TVModelVariant("50_32x4d", tvm.resnext50_32x4d, tvm.ResNeXt50_32X4D_Weights),
        TVModelVariant("101_328xd", tvm.resnext101_32x8d, tvm.ResNeXt101_32X8D_Weights),
        TVModelVariant("101_64x4d", tvm.resnext101_64x4d, tvm.ResNeXt101_64X4D_Weights),
    ]
)
@register_encoder("resnext", IMAGE)
class TVResNeXtEncoder(TVBaseEncoder):
    # specify base torchvision model
    torchvision_model_type: str = "resnext"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")
        super().__init__(**kwargs)

    def _remove_softmax_layer(self) -> None:
        self.model.fc = torch.nn.Identity()

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TVResNeXtEncoderConfig


@DeveloperAPI
@register_torchvision_model_variants(
    [
        TVModelVariant("x0_5", tvm.shufflenet_v2_x0_5, tvm.ShuffleNet_V2_X0_5_Weights),
        TVModelVariant("x1_0", tvm.shufflenet_v2_x1_0, tvm.ShuffleNet_V2_X1_0_Weights),
        TVModelVariant("x1_5", tvm.shufflenet_v2_x1_5, tvm.ShuffleNet_V2_X1_5_Weights),
        TVModelVariant("x2_0", tvm.shufflenet_v2_x2_0, tvm.ShuffleNet_V2_X2_0_Weights),
    ]
)
@register_encoder("shufflenet_v2", IMAGE)
class TVShuffleNetV2Encoder(TVBaseEncoder):
    # specify base torchvision model
    torchvision_model_type: str = "shufflenet_v2"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")
        super().__init__(**kwargs)

    def _remove_softmax_layer(self) -> None:
        self.model.fc = torch.nn.Identity()

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TVShuffleNetV2EncoderConfig


@DeveloperAPI
@register_torchvision_model_variants(
    [
        TVModelVariant("1_0", tvm.squeezenet1_0, tvm.SqueezeNet1_0_Weights),
        TVModelVariant("1_1", tvm.squeezenet1_1, tvm.SqueezeNet1_1_Weights),
    ]
)
@register_encoder("squeezenet", IMAGE)
class TVSqueezeNetEncoder(TVBaseEncoder):
    # specify base torchvision model
    torchvision_model_type: str = "squeezenet"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")
        super().__init__(**kwargs)

    def _remove_softmax_layer(self) -> None:
        # SqueezeNet does not have a final nn.Linear() layer
        # Use flatten output from last AdaptiveAvgPool2d layer
        # as encoder output.
        pass

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TVSqueezeNetEncoderConfig


@DeveloperAPI
@register_torchvision_model_variants(
    [
        TVModelVariant("t", tvm.swin_t, tvm.Swin_T_Weights),
        TVModelVariant("s", tvm.swin_s, tvm.Swin_S_Weights),
        TVModelVariant("b", tvm.swin_b, tvm.Swin_B_Weights),
    ]
)
@register_encoder("swin_transformer", IMAGE)
class TVSwinTransformerEncoder(TVBaseEncoder):
    # specify base torchvision model
    torchvision_model_type: str = "swin_transformer"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")
        super().__init__(**kwargs)

    def _remove_softmax_layer(self) -> None:
        self.model.head = torch.nn.Identity()

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TVSwinTransformerEncoderConfig


@DeveloperAPI
@register_torchvision_model_variants(
    [
        TVModelVariant(11, tvm.vgg11, tvm.VGG11_Weights),
        TVModelVariant("11_bn", tvm.vgg11_bn, tvm.VGG11_BN_Weights),
        TVModelVariant(13, tvm.vgg13, tvm.VGG13_Weights),
        TVModelVariant("13_bn", tvm.vgg13_bn, tvm.VGG13_BN_Weights),
        TVModelVariant(16, tvm.vgg16, tvm.VGG16_Weights),
        TVModelVariant("16_bn", tvm.vgg16_bn, tvm.VGG16_BN_Weights),
        TVModelVariant(19, tvm.vgg19, tvm.VGG19_Weights),
        TVModelVariant("19_bn", tvm.vgg19_bn, tvm.VGG19_BN_Weights),
    ]
)
@register_encoder("vgg", IMAGE)
class TVVGGEncoder(TVBaseEncoder):
    # specify base torchvison model
    torchvision_model_type: str = "vgg"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")
        super().__init__(**kwargs)

    def _remove_softmax_layer(self) -> None:
        self.model.classifier[-1] = torch.nn.Identity()

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TVVGGEncoderConfig


@DeveloperAPI
@register_torchvision_model_variants(
    [
        TVModelVariant("b_16", tvm.vit_b_16, tvm.ViT_B_16_Weights),
        TVModelVariant("b_32", tvm.vit_b_32, tvm.ViT_B_32_Weights),
        TVModelVariant("l_16", tvm.vit_l_16, tvm.ViT_L_16_Weights),
        TVModelVariant("l_32", tvm.vit_l_32, tvm.ViT_L_32_Weights),
        TVModelVariant("h_14", tvm.vit_h_14, tvm.ViT_H_14_Weights),
    ]
)
@register_encoder("vit", IMAGE)
class TVViTEncoder(TVBaseEncoder):
    # specify base torchvision model
    torchvision_model_type: str = "vit"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")

        # Depending on model variant and weight specification, the expected image size
        # will vary.  This code determines at run time what the expected image size will be
        # and adds to the kwargs dictionary the parameter that specifies the image size.
        # this is needed only if not using pretrained weights.  If pre-trained weights are
        # specified, then the correct image size is set.
        if not kwargs["use_pretrained"]:
            weights_specification = torchvision_model_registry[self.torchvision_model_type][
                kwargs["model_variant"]
            ].model_weights.DEFAULT
            kwargs["image_size"] = weights_specification.transforms.keywords["crop_size"]

        super().__init__(**kwargs)

    def _remove_softmax_layer(self) -> None:
        self.model.heads[-1] = torch.nn.Identity()

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TVViTEncoderConfig


@DeveloperAPI
@register_torchvision_model_variants(
    [
        TVModelVariant("50_2", tvm.wide_resnet50_2, tvm.Wide_ResNet50_2_Weights),
        TVModelVariant("101_2", tvm.wide_resnet101_2, tvm.Wide_ResNet101_2_Weights),
    ]
)
@register_encoder("wide_resnet", IMAGE)
class TVWideResNetEncoder(TVBaseEncoder):
    # specify base torchvision model
    torchvision_model_type: str = "wide_resnet"

    def __init__(
        self,
        **kwargs,
    ):
        logger.debug(f" {self.name}")
        super().__init__(**kwargs)

    def _remove_softmax_layer(self) -> None:
        self.model.fc = torch.nn.Identity()

    @staticmethod
    def get_schema_cls() -> Type[BaseEncoderConfig]:
        return TVWideResNetEncoderConfig
