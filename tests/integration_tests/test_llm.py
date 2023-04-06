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
    input_features = [text_feature()]
    output_features = [text_feature()]

    csv_filename = os.path.join(tmpdir, "training.csv")
    dataset_filename = generate_data(input_features, output_features, csv_filename, num_examples=100)

    config = {
        MODEL_TYPE: "llm",
        "model_name": "hf-internal-testing/tiny-random-GPTJForCausalLM",
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
    }

    model = LudwigModel(config, backend=backend)
    model.train(dataset=dataset_filename, output_directory=str(tmpdir))
    preds, _ = model.predict(dataset=dataset_filename, output_directory=str(tmpdir), split="test")
    print(preds)
