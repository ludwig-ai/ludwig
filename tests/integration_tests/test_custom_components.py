import os
import tempfile
from typing import Dict

import torch
from marshmallow_dataclass import dataclass
from torch import nn, Tensor

from ludwig.api import LudwigModel
from ludwig.combiners.combiners import Combiner, register_combiner
from ludwig.constants import NUMBER, TRAINER
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.modules.loss_modules import LogitsInputsMixin, register_loss
from ludwig.modules.metric_modules import LossMetric, register_metric
from ludwig.schema.combiners.base import BaseCombinerConfig
from tests.integration_tests.utils import (
    category_feature,
    generate_data,
    LocalTestBackend,
    number_feature,
    sequence_feature,
)


@dataclass
class CustomTestCombinerConfig(BaseCombinerConfig):
    foo: bool = False


@register_combiner(name="custom_test")
class CustomTestCombiner(Combiner):
    def __init__(self, input_features: Dict = None, config: CustomTestCombinerConfig = None, **kwargs):
        super().__init__(input_features)
        self.foo = config.foo

    def forward(self, inputs: Dict) -> Dict:  # encoder outputs
        if not self.foo:
            raise ValueError("expected foo to be True")

        # minimal transformation from inputs to outputs
        encoder_outputs = [inputs[k]["encoder_output"] for k in inputs]
        hidden = torch.cat(encoder_outputs, 1)
        return_data = {"combiner_output": hidden}

        return return_data

    @staticmethod
    def get_schema_cls():
        return CustomTestCombinerConfig


@register_encoder("custom_number_encoder", NUMBER)
class CustomNumberEncoder(Encoder):
    def __init__(self, input_size, **kwargs):
        super().__init__()
        self.input_size = input_size

    def forward(self, inputs, **kwargs):
        return {"encoder_output": inputs}

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])

    @property
    def output_shape(self) -> torch.Size:
        return self.input_shape


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


@register_loss("custom_loss", [NUMBER])
class CustomLoss(nn.Module, LogitsInputsMixin):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        return torch.mean(torch.square(preds - target))


@register_metric("custom_loss", [NUMBER])
class CustomLossMetric(LossMetric):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = CustomLoss()

    def get_current_value(self, preds: Tensor, target: Tensor):
        return self.loss_fn(preds, target)


def test_custom_combiner():
    _run_test(combiner={"type": "custom_test", "foo": True})


def test_custom_encoder_decoder():
    input_features = [
        sequence_feature(reduce_output="sum"),
        number_feature(encoder="custom_number_encoder"),
    ]
    output_features = [
        number_feature(decoder="custom_number_decoder"),
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
            sequence_feature(reduce_output="sum"),
            number_feature(),
        ]
        output_features = output_features or [category_feature(vocab_size=2, reduce_input="sum")]
        combiner = combiner or {"type": "concat"}

        csv_filename = os.path.join(tmpdir, "training.csv")
        data_csv = generate_data(input_features, output_features, csv_filename)

        config = {
            "input_features": input_features,
            "output_features": output_features,
            "combiner": combiner,
            TRAINER: {"epochs": 2},
        }

        model = LudwigModel(config, backend=LocalTestBackend())
        _, _, output_directory = model.train(
            dataset=data_csv,
            output_directory=tmpdir,
        )
        model.predict(dataset=data_csv, output_directory=output_directory)
