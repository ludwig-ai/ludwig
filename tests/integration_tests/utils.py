import pandas as pd
import os
import yaml

from string import Template
from ludwig.data.dataset_synthesyzer import build_synthetic_dataset

ENCODERS = ['embed', 'rnn', 'parallel_cnn', 'cnnrnn', 'stacked_parallel_cnn',
            'stacked_cnn']

model_definition_template = Template(
    '{input_features: ${input_name}, output_features: ${output_name}, '
    'training: {epochs: 2}, combiner: {type: concat, fc_size: 56}}')


def generate_data(input_features,
                  output_features,
                  filename='test_csv.csv',
                  num_examples=25):
    """
    Helper method to generate synthetic data based on input, output feature
    specs
    :param num_examples: number of examples to generate
    :param input_features: schema
    :param output_features: schema
    :param filename: path to the file where data is stored
    :return:
    """
    features = yaml.load(input_features) + yaml.load(output_features)
    df = build_synthetic_dataset(num_examples, features)
    data = [next(df) for _ in range(num_examples)]

    dataframe = pd.DataFrame(data[1:], columns=data[0])
    dataframe.to_csv(filename, index=False)

    return filename
