import logging
import os

import numpy as np
import pandas as pd
import pytest

from ludwig.api import LudwigModel
from ludwig.constants import BINARY, CATEGORY, MODEL_ECD, MODEL_GBM
from ludwig.explain.captum import IntegratedGradientsExplainer
from ludwig.explain.explainer import Explainer
from ludwig.explain.explanation import Explanation
from ludwig.explain.gbm import GBMExplainer
from tests.integration_tests.utils import (
    binary_feature,
    category_feature,
    generate_data,
    LocalTestBackend,
    number_feature,
)


def test_explanation_dataclass():
    explanation = Explanation(target="target")

    feature_attributions_for_label_1 = np.array([1, 2, 3])
    feature_attributions_for_label_2 = np.array([4, 5, 6])

    # test add()
    explanation.add(feature_attributions_for_label_1)

    with pytest.raises(AssertionError, match="Expected feature attributions of shape"):
        # test add() with wrong shape
        explanation.add(np.array([1, 2, 3, 4]))

    explanation.add(feature_attributions_for_label_2)

    # test to_array()
    explanation_array = explanation.to_array()
    assert np.array_equal(explanation_array, [[1, 2, 3], [4, 5, 6]])


def test_abstract_explainer_instantiation(tmpdir):
    with pytest.raises(TypeError, match="Can't instantiate abstract class Explainer with abstract method"):
        Explainer(None, inputs_df=None, sample_df=None, target=None)


@pytest.mark.parametrize("use_global", [True, False])
@pytest.mark.parametrize(
    "explainer_class, model_type",
    [
        (IntegratedGradientsExplainer, MODEL_ECD),
        (GBMExplainer, MODEL_GBM),
    ],
)
@pytest.mark.parametrize(
    "additional_config",
    [
        pytest.param({}, id="default"),
        pytest.param({"preprocessing": {"split": {"type": "fixed", "column": "split"}}}, id="fixed_split"),
    ],
)
def test_explainer_api(explainer_class, model_type, additional_config, use_global, tmpdir):
    output_features = [category_feature(decoder={"vocab_size": 3})]
    run_test_explainer_api(explainer_class, model_type, output_features, additional_config, use_global, tmpdir)


@pytest.mark.distributed
@pytest.mark.parametrize(
    "output_feature",
    [binary_feature(), number_feature(), category_feature(decoder={"vocab_size": 3})],
    ids=["binary", "number", "category"],
)
@pytest.mark.parametrize("use_global", [True, False])
def test_explainer_api_ray(use_global, output_feature, tmpdir, ray_cluster_2cpu):
    from ludwig.explain.captum_ray import RayIntegratedGradientsExplainer

    run_test_explainer_api(
        RayIntegratedGradientsExplainer,
        "ecd",
        [output_feature],
        {},
        use_global,
        tmpdir,
        resources_per_task={"num_cpus": 1},
        num_workers=1,
    )


def run_test_explainer_api(
    explainer_class, model_type, output_features, additional_config, use_global, tmpdir, **kwargs
):
    input_features = [number_feature(), category_feature(encoder={"reduce_output": "sum"})]

    # Generate data
    csv_filename = os.path.join(tmpdir, "training.csv")
    generate_data(input_features, output_features, csv_filename, num_examples=100)
    df = pd.read_csv(csv_filename)
    if "split" in additional_config.get("preprocessing", {}):
        df["split"] = np.random.randint(0, 3, df.shape[0])

    # Train model
    config = {"input_features": input_features, "output_features": output_features, "model_type": model_type}
    if model_type == MODEL_ECD:
        config["trainer"] = {"epochs": 2}
    config.update(additional_config)

    model = LudwigModel(config, logging_level=logging.WARNING, backend=LocalTestBackend())
    model.train(df)

    # Explain model
    explainer = explainer_class(
        model, inputs_df=df, sample_df=df, target=output_features[0]["name"], use_global=use_global, **kwargs
    )

    is_binary = output_features[0].get("type") == BINARY
    is_category = output_features[0].get("type") == CATEGORY

    vocab_size = 1
    if is_binary:
        vocab_size = 2
    elif is_category:
        vocab_size = output_features[0].get("decoder", {}).get("vocab_size")

    assert explainer.is_binary_target == is_binary
    assert explainer.is_category_target == is_category
    assert explainer.vocab_size == vocab_size

    explanations, expected_values = explainer.explain()

    # Verify shapes. One explanation per row, or 1 averaged explanation if `use_global=True`
    expected_explanations = len(df) if not use_global else 1
    assert len(explanations) == expected_explanations
    for e in explanations:
        assert e.to_array().shape == (vocab_size, len(input_features))

    assert len(expected_values) == vocab_size
