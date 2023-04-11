import os

import pytest

from ludwig.api import LudwigModel
from ludwig.constants import INPUT_FEATURES, MODEL_TYPE, OUTPUT_FEATURES
from tests.integration_tests.utils import generate_data, text_feature

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
    """Test that the GBM model can train and predict with non-number inputs."""
    input_features = [text_feature(name="Question")]
    output_features = [text_feature(output_feature=True, name="Answer")]

    csv_filename = os.path.join(tmpdir, "training.csv")
    dataset_filename = generate_data(input_features, output_features, csv_filename, num_examples=100)

    # import pandas as pd

    # qa_pairs = {
    #     "Question": [
    #         "What is the capital of Uzbekistan?",
    #         "Who is the founder of Microsoft?",
    #         "What is the tallest building in the world?",
    #         "What is the currency of Brazil?",
    #         "What is the boiling point of mercury in Celsius?",
    #         "What is the most commonly spoken language in the world?",
    #         "What is the diameter of the Earth?",
    #         'Who wrote the novel "1984"?',
    #         "What is the name of the largest moon of Neptune?",
    #         "What is the speed of light in meters per second?",
    #         "What is the smallest country in Africa by land area?",
    #         "What is the largest organ in the human body?",
    #         'Who directed the film "The Godfather"?',
    #         "What is the name of the smallest planet in our solar system?",
    #         "What is the largest lake in Africa?",
    #         "What is the smallest country in Asia by land area?",
    #         "Who is the current president of Russia?",
    #         "What is the chemical symbol for gold?",
    #         "What is the name of the famous Swiss mountain known for skiing?",
    #         "What is the largest flower in the world?",
    #     ],
    #     "Answer": [
    #         "Tashkent",
    #         "Bill Gates",
    #         "Burj Khalifa",
    #         "Real",
    #         "-38.83",
    #         "Mandarin",
    #         "12,742 km",
    #         "George Orwell",
    #         "Triton",
    #         "299,792,458 m/s",
    #         "Seychelles",
    #         "Skin",
    #         "Francis Ford Coppola",
    #         "Mercury",
    #         "Lake Victoria",
    #         "Maldives",
    #         "Vladimir Putin",
    #         "Au",
    #         "The Matterhorn",
    #         "Rafflesia arnoldii",
    #     ],
    # }

    # df = pd.DataFrame(qa_pairs)

    config = {
        MODEL_TYPE: "llm",
        "model_name": "hf-internal-testing/tiny-random-GPTJForCausalLM",  # "facebook/opt-350m"
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        "trainer": {
            "train_steps": 1,
        },
    }

    model = LudwigModel(config, backend=backend)
    # (TODO): Need to debug issue when skip_save_processed_input is False
    model.train(dataset=dataset_filename, output_directory=str(tmpdir), skip_save_processed_input=True)
    preds, _ = model.predict(dataset=dataset_filename, output_directory=str(tmpdir), split="test")
    # model.experiment(dataset_filename, output_directory=str(tmpdir), skip_save_processed_input=True)

    import pprint

    pprint.pprint(preds.to_dict())
