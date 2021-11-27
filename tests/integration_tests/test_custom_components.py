import os
from dataclasses import dataclass
from typing import Dict

import torch
from marshmallow import INCLUDE

from ludwig.api import LudwigModel
from ludwig.combiners.combiners import CombinerClass, combiner_registry
from ludwig.utils.registry import register
from tests.integration_tests.utils import sequence_feature, numerical_feature, category_feature, generate_data, \
    LocalTestBackend


@dataclass
class CustomTestCombinerConfig:
    foo: bool = False

    class Meta:
        unknown = INCLUDE


@register(name='custom_test')
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


def test_custom_combiner(tmpdir):
    input_features = [
        sequence_feature(reduce_output='sum'),
        numerical_feature(),
    ]
    output_features = [
        category_feature(vocab_size=2, reduce_input='sum')
    ]

    csv_filename = os.path.join(tmpdir, 'training.csv')
    data_csv = generate_data(input_features, output_features, csv_filename)

    config = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {
            'type': 'custom_test',
            'foo': True
        },
        'training': {'epochs': 2},
    }

    model = LudwigModel(config, backend=LocalTestBackend())
    _, _, output_directory = model.train(
        dataset=data_csv,
        output_directory=tmpdir,
    )
    model.predict(dataset=data_csv,
                  output_directory=output_directory)
