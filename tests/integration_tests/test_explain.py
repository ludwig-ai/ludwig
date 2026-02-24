import os

import numpy as np
import pandas as pd
import pytest

from ludwig.api import LudwigModel
from ludwig.constants import MODEL_ECD
from ludwig.explain.captum import IntegratedGradientsExplainer
from ludwig.explain.explainer import Explainer
from ludwig.explain.explanation import Explanation
from tests.integration_tests.utils import category_feature, generate_data, number_feature


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
    with pytest.raises(TypeError, match="Can't instantiate abstract class Explainer"):
        Explainer(None, inputs_df=None, sample_df=None, target=None)


def _train_model(tmpdir, input_features, output_features, model_type):
    csv_filename = os.path.join(tmpdir, "training.csv")
    generate_data(input_features, output_features, csv_filename, num_examples=100)
    df = pd.read_csv(csv_filename)

    config = {"input_features": input_features, "output_features": output_features, "model_type": model_type}

    model = LudwigModel(config)
    model.train(df)

    return df, model


@pytest.mark.parametrize(
    "explainer_class, model_type",
    [
        (IntegratedGradientsExplainer, MODEL_ECD),
    ],
)
def test_explainer_api(explainer_class, model_type, tmpdir):
    input_features = [number_feature(), category_feature(encoder={"reduce_output": "sum"})]
    vocab_size = 3
    output_features = [category_feature(decoder={"vocab_size": vocab_size})]

    df, model = _train_model(tmpdir, input_features, output_features, model_type)

    # Explain model
    explainer = explainer_class(model, inputs_df=df, sample_df=df, target=output_features[0]["name"])

    assert not explainer.is_binary_target
    assert explainer.is_category_target
    assert explainer.vocab_size == vocab_size

    explanations, expected_values = explainer.explain()

    # Verify shapes
    assert len(explanations) == len(df)
    for e in explanations:
        assert e.to_array().shape == (vocab_size, len(input_features))

    assert len(expected_values) == vocab_size
