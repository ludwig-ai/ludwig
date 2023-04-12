import os

import pytest

from ludwig.api import LudwigModel
from ludwig.constants import INPUT_FEATURES, MODEL_LLM, MODEL_NAME, MODEL_TYPE, OUTPUT_FEATURES
from tests.integration_tests.utils import category_feature, generate_data, text_feature

BOOSTING_TYPES = ["gbdt", "goss", "dart"]
TREE_LEARNERS = ["serial", "feature", "data", "voting"]
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


@pytest.fixture(scope="module")
def local_backend():
    return LOCAL_BACKEND


@pytest.fixture(scope="module")
def ray_backend():
    return RAY_BACKEND


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(LOCAL_BACKEND, id="local"),
        # pytest.param(RAY_BACKEND, id="ray", marks=pytest.mark.distributed),
    ],
)
def test_llm_text_to_text(tmpdir, backend):  # , ray_cluster_4cpu):
    """Test that the LLM model can train and predict with text inputs and text outputs."""
    input_features = [text_feature(name="Question")]
    output_features = [text_feature(output_feature=True, name="Answer")]

    csv_filename = os.path.join(tmpdir, "training.csv")
    dataset_filename = generate_data(input_features, output_features, csv_filename, num_examples=100)

    config = {
        MODEL_TYPE: MODEL_LLM,
        MODEL_NAME: "hf-internal-testing/tiny-random-GPTJForCausalLM",
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
    }

    model = LudwigModel(config, backend=backend)
    # (TODO): Need to debug issue when skip_save_processed_input is False
    model.train(dataset=dataset_filename, output_directory=str(tmpdir), skip_save_processed_input=True)
    preds, _ = model.predict(dataset=dataset_filename, output_directory=str(tmpdir), split="test")
    # model.experiment(dataset_filename, output_directory=str(tmpdir), skip_save_processed_input=True)

    import pprint

    pprint.pprint(preds.to_dict())


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(LOCAL_BACKEND, id="local"),
        # pytest.param(RAY_BACKEND, id="ray", marks=pytest.mark.distributed),
    ],
)
def test_llm_zero_shot_classification(tmpdir, backend):  # , ray_cluster_4cpu):
    input_features = [text_feature(name="review")]
    output_features = [
        category_feature(
            name="label",
            # (TODO): Figure out why preprocessing is not getting reflected in the config
            preprocessing={
                "labels": ["positive", "neutral", "negative"],
                "fallback_label": "neutral",
                "prompt_template": """
                Context information is below.
                ###
                {review}
                ###
                Given the context information and not prior knowledge, classify the context as one of: {labels}
                """,
            },
            # How can we avoid using r here for regex, since it is technically an implementation detail?
            decoder={
                "type": "parser",
                "match": {
                    "positive": {"type": "contains", "value": "positive"},
                    "neutral": {"type": "regex", "value": r"\bneutral\b"},
                    "negative": {"type": "contains", "value": "negative"},
                },
            },
        )
    ]

    import pandas as pd

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
        MODEL_NAME: "hf-internal-testing/tiny-random-GPTJForCausalLM",
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
    }

    model = LudwigModel(config, backend=backend)

    # (TODO): Need to debug issue when skip_save_processed_input is False
    model.train(dataset=df, output_directory=str(tmpdir))

    prediction_df = pd.DataFrame(
        {
            "review": ["The food was amazing!", "The service was terrible.", "The food was okay. "],
            "label": [
                "positive",
                "negative",
                "neutral",
            ],
        }
    )

    preds, _ = model.predict(dataset=prediction_df, output_directory=str(tmpdir))
    # model.experiment(dataset_filename, output_directory=str(tmpdir), skip_save_processed_input=True)

    import pprint

    pprint.pprint(preds.to_dict())
