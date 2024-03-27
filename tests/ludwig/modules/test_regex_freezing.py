import re

import pytest

from ludwig.encoders.image.torchvision import TVEfficientNetEncoder
from ludwig.schema.trainer import BaseTrainerConfig
from ludwig.utils.misc_utils import set_random_seed
from ludwig.utils.trainer_utils import freeze_layers_regex

RANDOM_SEED = 130


@pytest.mark.parametrize("trainable", [True])
@pytest.mark.parametrize(
    "use_pretrained",
    [
        False,
    ],
)
@pytest.mark.parametrize(
    "regex",
    [
        r"(features\.1.*|features\.2.*|features\.3.*|model\.features\.4\.1\.block\.3\.0\.weight)",
        r"(features\.1.*|features\.2\.*|features\.3.*)",
        r"(features\.4\.0\.block|features\.4\.\d+\.block)",
        r"(features\.5\.*|features\.6\.*|features\.7\.*)",
        r"(features\.8\.\d+\.weight|features\.8\.\d+\.bias)",
    ],
)
def test_tv_efficientnet_freezing(trainable: bool, use_pretrained: bool, regex):
    set_random_seed(RANDOM_SEED)

    pretrained_model = TVEfficientNetEncoder(
        model_variant="b0", use_pretrained=use_pretrained, saved_weights_in_checkpoint=True, trainable=trainable
    )

    config = BaseTrainerConfig(layers_to_freeze_regex=regex)
    freeze_layers_regex(config, pretrained_model)
    for name, param in pretrained_model.named_parameters():
        if re.search(re.compile(regex), name):
            assert not param.requires_grad
        else:
            assert param.requires_grad
