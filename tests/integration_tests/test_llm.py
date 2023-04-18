import os

import pandas as pd
import pytest
import yaml

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
GENERATION_CONFIG = "generation"


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
        GENERATION_CONFIG: get_generation_config(),
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

    data = [
        {"review": "I loved this movie!", "label": "positive"},
        {"review": "The food was okay, but the service was terrible.", "label": "negative"},
        {"review": "I can't believe how rude the staff was.", "label": "negative"},
        {"review": "This book was a real page-turner.", "label": "positive"},
        {"review": "The hotel room was dirty and smelled bad.", "label": "negative"},
        {"review": "I had a great experience at this restaurant.", "label": "positive"},
        {"review": "The concert was amazing!", "label": "positive"},
        {"review": "The traffic was terrible on my way to work this morning.", "label": "negative"},
        {"review": "The customer service was excellent.", "label": "positive"},
        {"review": "I was disappointed with the quality of the product.", "label": "negative"},
    ]

    df = pd.DataFrame(data)

    config = {
        MODEL_TYPE: MODEL_LLM,
        MODEL_NAME: TEST_MODEL_NAME,
        GENERATION_CONFIG: get_generation_config(),
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
    }

    model = LudwigModel(config, backend=backend)
    model.train(dataset=df, output_directory=str(tmpdir), skip_save_processed_input=True)

    prediction_df = pd.DataFrame(
        [
            {"review": "The food was amazing!", "label": "positive"},
            {"review": "The service was terrible.", "label": "negative"},
            {"review": "The food was okay.", "label": "neutral"},
        ]
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


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(LOCAL_BACKEND, id="local"),
        # pytest.param(RAY_BACKEND, id="ray", marks=pytest.mark.distributed),
    ],
)
def test_llm_few_shot_classification(tmpdir, backend):
    df = pd.DataFrame(
        [
            {
                "reviews_text": "I liked the look and location of the",
                "reviews_rating_floor": 3,
            },
            {
                "reviews_text": "My wife and I have not stayed in",
                "reviews_rating_floor": 5,
            },
            {
                "reviews_text": "This was the hotel our son and daughter",
                "reviews_rating_floor": 5,
            },
            {
                "reviews_text": "great hotel right on the ocean with a",
                "reviews_rating_floor": 4,
            },
            {
                "reviews_text": "The hotel was great from the start.",
                "reviews_rating_floor": 5,
            },
            {
                "reviews_text": "We stayed here pre and post cruise.",
                "reviews_rating_floor": 4,
            },
            {
                "reviews_text": "This place was just ok, sketchy looking",
                "reviews_rating_floor": 2,
            },
            {
                "reviews_text": "Stayed here for a business trip. Very",
                "reviews_rating_floor": 4,
            },
            {
                "reviews_text": "We went on a trip to new orleans it was fun",
                "reviews_rating_floor": 3,
            },
            {
                "reviews_text": "My wife and I have stayed at Ascend brand ",
                "reviews_rating_floor": 5,
            },
        ]
    )

    prediction_df = pd.DataFrame(
        [
            {
                "reviews_text": "I really liked this hotel. I stayed in",
                "reviews_rating_floor": 4,
            },
            {
                "reviews_text": "Stayed for One night but really enjoyed our stay",
                "reviews_rating_floor": 4,
            },
            {
                "reviews_text": "Very conveniently located near the airport and Ft.",
                "reviews_rating_floor": 5,
            },
            {
                "reviews_text": "We stayed at the Del Sol Inn again over",
                "reviews_rating_floor": 5,
            },
            {
                "reviews_text": "We had a wonderful time staying at Eden Roc",
                "reviews_rating_floor": 5,
            },
        ]
    )

    config = """
model_type: llm
model_name: hf-internal-testing/tiny-random-GPTJForCausalLM
generation:
    temperature: 0.1
    top_p: 0.75
    top_k: 40
    num_beams: 4
    max_new_tokens: 5
input_features:
-
    name: reviews_text
    type: text
    preprocessing:
        prompt:
            retrieval:
                type: "semantic"
                index_name: null
                model_name: multi-qa-MiniLM-L6-cos-v1
                k: 5
            task: "Given the sample input, complete this sentence by
                replacing XXXX: The review rating is XXXX. Choose one value
                in this list: [1, 2, 3, 4, 5]."
output_features:
-
    name: reviews_rating_floor
    type: category
    preprocessing:
        vocab:
            - "1"
            - "2"
            - "3"
            - "4"
            - "5"
        fallback_label: "3"
    decoder:
        type: category_parser
        match:
            "1":
                type: contains
                value: "1"
            "2":
                type: contains
                value: "2"
            "3":
                type: contains
                value: "3"
            "4":
                type: contains
                value: "4"
            "5":
                type: contains
                value: "5"
"""
    config = yaml.safe_load(config)

    model = LudwigModel(config, backend={"type": "local", "cache_dir": str(tmpdir)})
    model.train(dataset=df, output_directory=str(tmpdir), skip_save_processed_input=True)

    preds, _ = model.predict(dataset=prediction_df, output_directory=str(tmpdir))
    preds = convert_preds(backend, preds)
