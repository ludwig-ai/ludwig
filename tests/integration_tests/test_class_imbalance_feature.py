import numpy as np
import pandas as pd
import pytest

from ludwig.api import LudwigModel
from ludwig.backend import LocalBackend
from ludwig.constants import PROC_COLUMN


def run_test_imbalance(
    input_df,
    config,
    balance,
):
    model = LudwigModel(config)
    _, output_dataset, output_dir = model.train(
        input_df,
        skip_save_model=True,
        skip_save_log=True,
        skip_save_progress=True,
        skip_save_processed_input=True,
        skip_save_training_description=True,
        skip_save_training_statistics=True,
    )

    target = model.config["output_features"][0][PROC_COLUMN]
    input_train_set = input_df.sample(frac=0.7, replace=False)
    processed_len = output_dataset[0].size
    processed_target_pos = sum(output_dataset[0].dataset[target])
    processed_target_neg = len(output_dataset[0].dataset[target]) - processed_target_pos
    assert len(input_train_set) == 140
    assert 0.05 <= len(input_train_set[input_train_set["Label"] == 1]) / len(input_train_set) <= 0.15
    assert round(processed_target_pos / processed_target_neg, 1) == 0.5
    assert isinstance(model.backend, LocalBackend)

    if balance == "oversample_minority":
        assert len(input_train_set) < processed_len
        assert 55 <= processed_target_pos <= 75
        assert 110 <= processed_target_neg <= 150

    if balance == "undersample_majority":
        assert len(input_train_set) > processed_len
        assert 7 <= processed_target_pos <= 20
        assert 14 <= processed_target_neg <= 40


@pytest.mark.parametrize(
    "balance",
    ["oversample_minority", "undersample_majority"],
)
def test_imbalance(balance):
    config = {
        "input_features": [
            {"name": "Index", "column": "Index", "type": "number"},
            {"name": "random_1", "column": "random_1", "type": "number"},
            {"name": "random_2", "column": "random_2", "type": "number"},
        ],
        "output_features": [{"name": "Label", "column": "Label", "proc_column": "Label_mZFLky", "type": "binary"}],
        "trainer": {"epochs": 2, "batch_size": 8},
        "preprocessing": {},
    }
    df = pd.DataFrame(
        {
            "Index": np.arange(0, 200, 1),
            "random_1": np.random.randint(0, 50, 200),
            "random_2": np.random.choice(["Type A", "Type B", "Type C", "Type D"], 200),
            "Label": np.concatenate((np.zeros(180), np.ones(20))),
        }
    )

    config["preprocessing"][balance] = 0.5
    run_test_imbalance(df, config, balance)
