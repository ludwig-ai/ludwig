import logging
import tempfile
from typing import Dict, Tuple

import pytest

from ludwig.constants import (
    CATEGORY,
    COMBINER,
    DECODER,
    DEFAULTS,
    ENCODER,
    FILL_WITH_CONST,
    INPUT_FEATURES,
    LOSS,
    OUTPUT_FEATURES,
    PREPROCESSING,
    SOFTMAX_CROSS_ENTROPY,
    TRAINER,
    TYPE,
)
from tests.integration_tests.utils import category_feature, generate_data, run_experiment, text_feature

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)


def _prepare_data(csv_filename: str) -> Tuple[Dict, str]:
    input_features = [
        text_feature(name="title", cell_type="rnn", reduce_output="sum"),
        text_feature(name="summary", cell_type="rnn"),
        category_feature(vocab_size=2, reduce_input="sum"),
        category_feature(vocab_size=3),
    ]

    output_features = [category_feature(vocab_size=2, reduce_input="sum")]

    dataset = generate_data(input_features, output_features, csv_filename)

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        COMBINER: {TYPE: "concat", "num_fc_layers": 2},
        TRAINER: {"epochs": 2, "learning_rate": 0.001},
    }

    return config, dataset


@pytest.mark.distributed
@pytest.mark.parametrize("backend", ["local", "ray"])
def test_run_global_default_parameters(backend, csv_filename):
    config, dataset = _prepare_data(csv_filename)

    config[DEFAULTS] = {
        CATEGORY: {
            PREPROCESSING: {"missing_value_strategy": FILL_WITH_CONST, "fill_value": "<UNK>"},
            ENCODER: {TYPE: "sparse"},
            DECODER: {"norm_params": None, "dropout": 0, "use_bias": True},
            LOSS: {TYPE: SOFTMAX_CROSS_ENTROPY, "confidence_penalty": 0},
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        run_experiment(config=config, backend=backend, dataset=dataset, output_dir=tmpdir)
