from typing import Dict, Tuple

from ludwig.constants import COMBINER, INPUT_FEATURES, OUTPUT_FEATURES, TRAINER, TYPE
from tests.integration_tests.utils import category_feature, generate_data, text_feature


def _prepared_data(csv_filename: str) -> Tuple[Dict, str]:
    input_features = [
        text_feature(name="title", cell_type="rnn", reduce_output="sum"),
        text_feature(name="summary", cell_type="rnn"),
        category_feature(vocab_size=2, reduce_input="sum"),
        category_feature(vocab_size=3),
    ]

    output_features = [category_feature(vocab_size=2, reduce_input="sum")]

    rel_path = generate_data(input_features, output_features, csv_filename)

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        COMBINER: {TYPE: "concat", "num_fc_layers": 2},
        TRAINER: {"epochs": 2, "learning_rate": 0.001},
    }

    return config, rel_path
