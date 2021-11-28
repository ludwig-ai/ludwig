import os
import tempfile
from dataclasses import dataclass
from typing import Dict

import torch
from marshmallow import INCLUDE

from ludwig.api import LudwigModel
from ludwig.combiners.combiners import CombinerClass, register_combiner
from tests.integration_tests.utils import sequence_feature, numerical_feature, category_feature, generate_data, \
    LocalTestBackend


@dataclass
class CustomTestCombinerConfig:
    foo: bool = False

    class Meta:
        unknown = INCLUDE


@register_combiner(name='custom_test')
class CustomTestCombiner(CombinerClass):
    def __init__(
            self,
            input_features: Dict = None,
            config: CustomTestCombinerConfig = None,
            **kwargs
    ):
        super().__init__()
        self.input_features = input_features
        self.foo = config.foo

    def forward(
            self,
            inputs: Dict  # encoder outputs
    ) -> Dict:
        if not self.foo:
            raise ValueError("expected foo to be True")

        # minimal transformation from inputs to outputs
        encoder_outputs = [inputs[k]['encoder_output'] for k in inputs]
        hidden = torch.cat(encoder_outputs, 1)
        return_data = {'combiner_output': hidden}

        return return_data

    @staticmethod
    def get_schema_cls():
        return CustomTestCombinerConfig


def test_custom_combiner():
    _run_test(combiner={
        'type': 'custom_test',
        'foo': True
    })


def test_custom_encoder_decoder():
    pass


def test_custom_loss_metric():
    pass


def _run_test(input_features=None, output_features=None, combiner=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_features = input_features or [
            sequence_feature(reduce_output='sum'),
            numerical_feature(),
        ]
        output_features = output_features or [
            category_feature(vocab_size=2, reduce_input='sum')
        ]
        combiner = combiner or {
            'type': 'concat'
        }

        csv_filename = os.path.join(tmpdir, 'training.csv')
        data_csv = generate_data(input_features, output_features, csv_filename)

        config = {
            'input_features': input_features,
            'output_features': output_features,
            'combiner': combiner,
            'training': {'epochs': 2},
        }

        model = LudwigModel(config, backend=LocalTestBackend())
        _, _, output_directory = model.train(
            dataset=data_csv,
            output_directory=tmpdir,
        )
        model.predict(dataset=data_csv,
                      output_directory=output_directory)
