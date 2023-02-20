import numpy as np
import pandas as pd
import pytest

from ludwig.backend.base import LocalBackend
from ludwig.constants import BALANCE_PERCENTAGE_TOLERANCE, NAME
from ludwig.data.preprocessing import balance_data


@pytest.mark.parametrize(
    "method, balance",
    [
        ("oversample_minority", 0.25),
        ("oversample_minority", 0.5),
        ("oversample_minority", 0.75),
        ("undersample_majority", 0.25),
        ("undersample_majority", 0.5),
        ("undersample_majority", 0.75),
        ("undersample_majority", 0.9),
    ],
)
def test_balance(method, balance):
    config = {
        "input_features": [
            {"name": "Index", "proc_column": "Index", "type": "number"},
            {"name": "random_1", "proc_column": "random_1", "type": "number"},
            {"name": "random_2", "proc_column": "random_2", "type": "number"},
        ],
        "output_features": [{"name": "Label", "proc_column": "Label", "type": "binary"}],
        "preprocessing": {"oversample_minority": None, "undersample_majority": None},
    }
    input_df = pd.DataFrame(
        {
            "Index": np.arange(0, 200, 1),
            "random_1": np.random.randint(0, 50, 200),
            "random_2": np.random.choice(["Type A", "Type B", "Type C", "Type D"], 200),
            "Label": np.concatenate((np.zeros(180), np.ones(20))),
            "split": np.zeros(200),
        }
    )

    config["preprocessing"][method] = balance
    backend = LocalBackend()

    test_df = balance_data(input_df, config["output_features"], config["preprocessing"], backend)
    target = config["output_features"][0][NAME]
    majority_class = test_df[target].value_counts()[test_df[target].value_counts().idxmax()]
    minority_class = test_df[target].value_counts()[test_df[target].value_counts().idxmin()]
    new_class_balance = round(minority_class / majority_class, 2)

    assert abs(balance - new_class_balance) < BALANCE_PERCENTAGE_TOLERANCE
