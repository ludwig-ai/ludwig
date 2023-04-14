import os

import pandas as pd
import pytest

from ludwig.api import LudwigModel
from ludwig.constants import INPUT_FEATURES, MODEL_LLM, MODEL_NAME, MODEL_TYPE, OUTPUT_FEATURES
from ludwig.utils.types import DataFrame
from tests.integration_tests.utils import category_feature, generate_data, text_feature

LOCAL_BACKEND = {"type": "local"}
RAY_BACKEND = {
    "type": "ray",
    "processor": {
        "parallelism": 1,
    },
    "trainer": {
        "use_gpu": False,
        "num_workers": 2,
        "resources_per_worker": {
            "CPU": 1,
            "GPU": 0,
        },
    },
}
TEST_MODEL_NAME = "hf-internal-testing/tiny-random-GPTJForCausalLM"


@pytest.fixture(scope="module")
def local_backend():
    return LOCAL_BACKEND


@pytest.fixture(scope="module")
def ray_backend():
    return RAY_BACKEND


def get_generation_config():
    return {
        "temperature": 0.1,
        "top_p": 0.75,
        "top_k": 40,
        "num_beams": 4,
        "max_new_tokens": 5,
    }


def convert_preds(backend: dict, preds: DataFrame):
    if backend["type"] == "ray":
        return preds.compute().to_dict()
    return preds.to_dict()


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(LOCAL_BACKEND, id="local"),
        # pytest.param(RAY_BACKEND, id="ray", marks=pytest.mark.distributed),
    ],
)
def test_llm_text_to_text(tmpdir, backend):  # , ray_cluster_4cpu)
    """Test that the LLM model can train and predict with text inputs and text outputs."""
    input_features = [{"name": "Question", "type": "text"}]
    output_features = [text_feature(output_feature=True, name="Answer", decoder={"type": "text_parser"})]

    csv_filename = os.path.join(tmpdir, "training.csv")
    dataset_filename = generate_data(input_features, output_features, csv_filename, num_examples=100)

    config = {
        MODEL_TYPE: MODEL_LLM,
        MODEL_NAME: TEST_MODEL_NAME,
        "generation_config": get_generation_config(),
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
    }

    model = LudwigModel(config, backend=backend)
    model.train(dataset=dataset_filename, output_directory=str(tmpdir), skip_save_processed_input=True)

    preds, _ = model.predict(dataset=dataset_filename, output_directory=str(tmpdir), split="test")
    preds = convert_preds(backend, preds)

    assert "Answer_predictions" in preds
    assert "Answer_probabilities" in preds
    assert "Answer_probability" in preds

    assert preds["Answer_predictions"]
    assert preds["Answer_probabilities"]
    assert preds["Answer_probability"]


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(LOCAL_BACKEND, id="local"),
        pytest.param(RAY_BACKEND, id="ray", marks=pytest.mark.distributed),
    ],
)
def test_llm_zero_shot_classification(tmpdir, backend, ray_cluster_4cpu):
    input_features = [{"name": "review", "type": "text"}]
    output_features = [
        category_feature(
            name="label",
            preprocessing={
                "vocab": ["positive", "neutral", "negative"],
                "fallback_label": "neutral",
                "prompt_template": """
                    Context information is below.
                    ###
                    {review}
                    ###
                    Given the context information and not prior knowledge, classify the context as one of: {vocab}
                """,
            },
            # How can we avoid using r here for regex, since it is technically an implementation detail?
            decoder={
                "type": "category_parser",
                "match": {
                    "positive": {"type": "contains", "value": "positive"},
                    "neutral": {"type": "regex", "value": r"\bneutral\b"},
                    "negative": {"type": "contains", "value": "negative"},
                },
            },
        )
    ]

    reviews = [
        "I loved this movie!",
        "The food was okay, but the service was terrible.",
        "I can't believe how rude the staff was.",
        "This book was a real page-turner.",
        "The hotel room was dirty and smelled bad.",
        "I had a great experience at this restaurant.",
        "The concert was amazing!",
        "The traffic was terrible on my way to work this morning.",
        "The customer service was excellent.",
        "I was disappointed with the quality of the product.",
    ]

    labels = [
        "positive",
        "negative",
        "negative",
        "positive",
        "negative",
        "positive",
        "positive",
        "negative",
        "positive",
        "negative",
    ]

    df = pd.DataFrame({"review": reviews, "label": labels})

    config = {
        MODEL_TYPE: MODEL_LLM,
        MODEL_NAME: TEST_MODEL_NAME,
        "generation_config": get_generation_config(),
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
    }

    model = LudwigModel(config, backend=backend)
    model.train(dataset=df, output_directory=str(tmpdir), skip_save_processed_input=True)

    prediction_df = pd.DataFrame(
        {
            "review": ["The food was amazing!", "The service was terrible.", "The food was okay."],
            "label": [
                "positive",
                "negative",
                "neutral",
            ],
        }
    )

    preds, _ = model.predict(dataset=prediction_df, output_directory=str(tmpdir))
    preds = convert_preds(backend, preds)

    assert "label_predictions" in preds
    assert "label_probabilities" in preds
    assert "label_probability" in preds
    assert "label_probabilities_positive" in preds
    assert "label_probabilities_neutral" in preds
    assert "label_probabilities_negative" in preds

    assert preds["label_predictions"]
    assert preds["label_probabilities"]
    assert preds["label_probability"]
    assert preds["label_probabilities_positive"]
    assert preds["label_probabilities_neutral"]
    assert preds["label_probabilities_negative"]
