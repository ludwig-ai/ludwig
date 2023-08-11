import os
import tempfile
from typing import Dict

import torch
from marshmallow_dataclass import dataclass
from torch import nn, Tensor

from ludwig.api import LudwigModel
from ludwig.combiners.combiners import Combiner, register_combiner
from ludwig.constants import BATCH_SIZE, ENCODER_OUTPUT, LOGITS, MINIMIZE, NUMBER, TRAINER
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.modules.loss_modules import LogitsInputsMixin, register_loss
from ludwig.modules.metric_modules import LossMetric, register_metric
from ludwig.schema import utils as schema_utils
from ludwig.schema.combiners.base import BaseCombinerConfig
from ludwig.schema.combiners.utils import register_combiner_config
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import register_decoder_config
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import register_encoder_config
from ludwig.schema.features.loss.loss import BaseLossConfig
from ludwig.schema.features.loss.loss import register_loss as register_loss_schema
from tests.integration_tests.utils import (
    category_feature,
    generate_data,
    LocalTestBackend,
    number_feature,
    sequence_feature,
)


@register_encoder_config("custom_number_encoder", NUMBER)
@dataclass
class CustomNumberEncoderConfig(BaseEncoderConfig):
    type: str = "custom_number_encoder"

    input_size: int = schema_utils.PositiveInteger(default=1, description="")


@register_decoder_config("custom_number_decoder", NUMBER)
@dataclass
class CustomNumberDecoderConfig(BaseDecoderConfig):
    type: str = "custom_number_decoder"

    input_size: int = schema_utils.PositiveInteger(default=1, description="")


@register_loss_schema([NUMBER])
@dataclass
class CustomLossConfig(BaseLossConfig):
    type: str = "custom_loss"


@register_combiner_config("custom_combiner")
@dataclass
class CustomTestCombinerConfig(BaseCombinerConfig):
    type: str = "custom_combiner"

    foo: bool = schema_utils.Boolean(default=False, description="")


@register_combiner(CustomTestCombinerConfig)
class CustomTestCombiner(Combiner):
    def __init__(self, input_features: Dict = None, config: CustomTestCombinerConfig = None, **kwargs):
        super().__init__(input_features)
        self.foo = config.foo

    def forward(self, inputs: Dict) -> Dict:  # encoder outputs
        if not self.foo:
            raise ValueError("expected foo to be True")

        # minimal transformation from inputs to outputs
        encoder_outputs = [inputs[k][ENCODER_OUTPUT] for k in inputs]
        hidden = torch.cat(encoder_outputs, 1)
        return_data = {"combiner_output": hidden}

        return return_data


@register_encoder("custom_number_encoder", NUMBER)
class CustomNumberEncoder(Encoder):
    def __init__(self, input_size, **kwargs):
        super().__init__()
        self.input_size = input_size

    def forward(self, inputs, **kwargs):
        return {ENCODER_OUTPUT: inputs}

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        return self.input_shape

    @staticmethod
    def get_schema_cls():
        return CustomNumberEncoderConfig


@register_decoder("custom_number_decoder", NUMBER)
class CustomNumberDecoder(Decoder):
    def __init__(self, input_size, **kwargs):
        super().__init__()
        self.input_size = input_size

    @property
    def input_shape(self):
        return torch.Size([self.input_size])

    def forward(self, inputs, **kwargs):
        return torch.mean(inputs, 1)

    @staticmethod
    def get_schema_cls():
        return CustomNumberDecoderConfig


@register_loss(CustomLossConfig)
class CustomLoss(nn.Module, LogitsInputsMixin):
    def __init__(self, config: CustomLossConfig):
        super().__init__()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        return torch.mean(torch.square(preds - target))

    @staticmethod
    def get_schema_cls():
        return CustomLossConfig


@register_metric("custom_loss", [NUMBER], MINIMIZE, LOGITS)
class CustomLossMetric(LossMetric):
    def __init__(self, config: CustomLossConfig, **kwargs):
        super().__init__()
        self.loss_fn = CustomLoss(config)

    def get_current_value(self, preds: Tensor, target: Tensor):
        return self.loss_fn(preds, target)


def test_custom_combiner():
    _run_test(combiner={"type": "custom_combiner", "foo": True})


def test_custom_encoder_decoder():
    input_features = [
        sequence_feature(encoder={"reduce_output": "sum"}),
        number_feature(encoder={"type": "custom_number_encoder"}),
    ]
    output_features = [
        number_feature(decoder={"type": "custom_number_decoder"}),
    ]
    _run_test(input_features=input_features, output_features=output_features)


def test_custom_loss_metric():
    output_features = [
        number_feature(loss={"type": "custom_loss"}),
    ]
    _run_test(output_features=output_features)


def _run_test(input_features=None, output_features=None, combiner=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_features = input_features or [
            sequence_feature(encoder={"reduce_output": "sum"}),
            number_feature(),
        ]
        output_features = output_features or [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]
        combiner = combiner or {"type": "concat"}

        csv_filename = os.path.join(tmpdir, "training.csv")
        data_csv = generate_data(input_features, output_features, csv_filename)

        config = {
            "input_features": input_features,
            "output_features": output_features,
            "combiner": combiner,
            TRAINER: {"epochs": 2, BATCH_SIZE: 128},
        }

        model = LudwigModel(config, backend=LocalTestBackend())
        _, _, output_directory = model.train(
            dataset=data_csv,
            output_directory=tmpdir,
        )
        model.predict(dataset=data_csv, output_directory=output_directory)
