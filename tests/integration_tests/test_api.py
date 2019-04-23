import pytest
import uuid
import yaml

from string import Template
from ludwig.api import LudwigModel
from ludwig.utils.data_utils import read_csv

from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import model_definition_template
from tests.integration_tests.utils import ENCODERS
from tests.fixtures.csv_filename import csv_filename


def run_api_experiment(input_features, output_features, data_csv):
    """
    Helper method to avoid code repetition in running an experiment
    :param input_features: input schema
    :param output_features: output schema
    :param data_csv: path to data
    :return: None
    """
    model_definition = model_definition_template.substitute(
        input_name=input_features,
        output_name=output_features
    )

    model = LudwigModel(yaml.load(model_definition))

    # Training with csv
    model.train(
        data_csv=data_csv,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True
    )
    model.predict(data_csv=data_csv)

    # Training with dataframe
    data_df = read_csv(data_csv)
    model.train(
        data_df=data_df,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True
    )
    model.predict(data_df=data_df)


def test_api_intent_classification(csv_filename):
    # Single sequence input, single category output
    input_features = Template('[{name: utterance, type: sequence,'
                              'vocab_size: 10, max_len: 10, '
                              'encoder: ${encoder}, reduce_output: sum}]')
    output_features = "[{name: intent, type: category, vocab_size: 2," \
                      " reduce_input: sum}] "

    # Generate test data
    rel_path = generate_data(
        input_features.substitute(encoder='rnn'), output_features, csv_filename
    )
    for encoder in ENCODERS:
        run_api_experiment(
            input_features.substitute(encoder=encoder),
            output_features,
            data_csv=rel_path
        )
