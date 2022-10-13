from collections import namedtuple

import pytest

from ludwig.models.base import BaseModel
from ludwig.schema.model_config import ModelConfig
from tests.integration_tests.utils import (
    category_feature,
    generate_data,
    number_feature,
    run_experiment,
    sequence_feature,
    text_feature,
)

# InputFeatureOptions namedtuple structure:
# feature_type: input feature type, e.g., number, category, etc.
# feature_options: None or dictionary of required input feature specification
# tie_features: boolean, True to tie features, False not to tie features
InputFeatureOptions = namedtuple("InputFeatureOptions", "feature_type feature_options tie_features")


# micro level test confirms the encoders for tied input features are sharing
# the same encoder.  Include negative tests to confirm untied input features
# do not share the same encoder.
# note: vocab parameter, below, is made up to facilitate creating input encoders
@pytest.mark.parametrize(
    "input_feature_options",
    [
        # tie input features, encoders should be the same
        InputFeatureOptions("number", {"encoder": {"type": "passthrough"}}, True),
        InputFeatureOptions(
            "number", {"encoder": {"type": "passthrough"}, "preprocessing": {"normalization": "zscore"}}, True
        ),
        InputFeatureOptions("binary", {"encoder": {"type": "passthrough"}}, True),
        InputFeatureOptions("category", {"encoder": {"type": "dense", "vocab": ["a", "b", "c"]}}, True),
        InputFeatureOptions("set", {"encoder": {"type": "embed", "vocab": ["a", "b", "c"]}}, True),
        InputFeatureOptions(
            "sequence", {"encoder": {"type": "parallel_cnn", "max_sequence_length": 10, "vocab": ["x", "y", "z"]}}, True
        ),
        InputFeatureOptions(
            "text", {"encoder": {"type": "parallel_cnn", "max_sequence_length": 10, "vocab": ["a", "b", "c"]}}, True
        ),
        InputFeatureOptions(
            "timeseries", {"encoder": {"type": "parallel_cnn", "max_sequence_length": 10, "should_embed": False}}, True
        ),
        InputFeatureOptions(
            "audio",
            {
                "encoder": {
                    "type": "parallel_cnn",
                    "embedding_size": 64,
                    "max_sequence_length": 16,
                    "should_embed": False,
                }
            },
            True,
        ),
        # do not tie input features, encoders should be different
        InputFeatureOptions("number", {"encoder": {"type": "passthrough"}}, False),
        InputFeatureOptions(
            "number", {"encoder": {"type": "passthrough"}, "preprocessing": {"normalization": "zscore"}}, False
        ),
        InputFeatureOptions("binary", {"encoder": {"type": "passthrough"}}, False),
        InputFeatureOptions("category", {"encoder": {"type": "dense", "vocab": ["a", "b", "c"]}}, False),
        InputFeatureOptions("set", {"encoder": {"type": "embed", "vocab": ["a", "b", "c"]}}, False),
        InputFeatureOptions(
            "sequence",
            {"encoder": {"type": "parallel_cnn", "max_sequence_length": 10, "vocab": ["x", "y", "z"]}},
            False,
        ),
        InputFeatureOptions(
            "text", {"encoder": {"type": "parallel_cnn", "max_sequence_length": 10, "vocab": ["a", "b", "c"]}}, False
        ),
        InputFeatureOptions(
            "timeseries", {"encoder": {"type": "parallel_cnn", "max_sequence_length": 10, "should_embed": False}}, False
        ),
        InputFeatureOptions(
            "audio",
            {
                "encoder": {
                    "type": "parallel_cnn",
                    "embedding_size": 64,
                    "max_sequence_length": 16,
                    "should_embed": False,
                }
            },
            False,
        ),
    ],
)
def test_tied_micro_level(input_feature_options):
    # build input feature config
    input_feature_configs = list()

    input_feature_configs.append({"name": "input_feature_1", "type": input_feature_options.feature_type})
    input_feature_configs[0].update(input_feature_options.feature_options)

    input_feature_configs.append({"name": "input_feature_2", "type": input_feature_options.feature_type})
    input_feature_configs[1].update(input_feature_options.feature_options)

    # add tied option to the second feature
    if input_feature_options.tie_features:
        input_feature_configs[1]["tied"] = "input_feature_1"

    config_obj = ModelConfig.from_dict(
        {"input_features": input_feature_configs, "output_features": [{"name": "dummy_feature", "type": "binary"}]}
    )

    input_features = BaseModel.build_inputs(input_feature_configs=config_obj.input_features)

    if input_feature_options.tie_features:
        # should be same encoder
        assert input_features["input_feature_1"].encoder_obj is input_features["input_feature_2"].encoder_obj
    else:
        # no tied parameter, encoders should be different
        assert input_features["input_feature_1"].encoder_obj is not input_features["input_feature_2"].encoder_obj


# TiedUseCase namedtuple structure:
# input_feature: Ludwig synthetic data creation function.
# output_feature: Ludwig synthetic data creation function
TiedUseCase = namedtuple("TiedUseCase", "input_feature output_feature")


# Macro level test ensures no exceptions are raised during a full_experiment()
@pytest.mark.parametrize(
    "tied_use_case",
    [
        TiedUseCase(number_feature, number_feature),
        TiedUseCase(text_feature, category_feature),
        TiedUseCase(sequence_feature, sequence_feature),
    ],
)
def test_tied_macro_level(tied_use_case: TiedUseCase, csv_filename: str):
    input_features = [
        number_feature(),  # Other feature
        tied_use_case.input_feature(),  # first feature to be tied
        tied_use_case.input_feature(),  # second feature to be tied
        category_feature(),  # other feature
    ]
    # tie second feature to first feature
    input_features[2]["tied"] = input_features[1]["name"]

    # setup output feature
    output_features = [tied_use_case.output_feature(output_feature=True)]

    # Generate test data and run full_experiment
    rel_path = generate_data(input_features, output_features, csv_filename)
    run_experiment(input_features, output_features, dataset=rel_path)
