import numpy as np
import pandas as pd
import pytest


from ludwig.backend.base import LocalBackend
from ludwig.constants import NAME, BALANCE_PERCENTAGE_TOLERANCE
from ludwig.data.preprocessing import balance_data


def run_test_balance_data(
    input_df,
    config,
    target,
    target_balance,
    backend,
):
    test_df = balance_data(input_df, config["output_features"], config["preprocessing"], backend)

    majority_class = test_df[target].value_counts()[test_df[target].value_counts().idxmax()]
    minority_class = test_df[target].value_counts()[test_df[target].value_counts().idxmin()]
    new_class_balance = round(minority_class / majority_class, 2)

    assert (target_balance - BALANCE_PERCENTAGE_TOLERANCE) <= new_class_balance <= (target_balance + BALANCE_PERCENTAGE_TOLERANCE)
    assert isinstance(backend, LocalBackend)


@pytest.mark.parametrize(
    "method, balance",
    [
        ("oversample_minority", 0.25),
        ("oversample_minority", 0.5),
        ("oversample_minority", 0.75),
        ("undersample_majority", 0.25),
        ("undersample_majority", 0.5),
        ("undersample_majority", 0.75),
    ],
)
def test_balance_data(method, balance):
    config = {
        "input_features": [
            {"name": "Index", "proc_column": "Index", "type": "numerical"},
            {"name": "random_1", "proc_column": "random_1", "type": "numerical"},
            {"name": "random_2", "proc_column": "random_2", "type": "numerical"},
        ],
        "output_features": [{"name": "Label", "proc_column": "Label", "type": "binary"}],
        "preprocessing": {"oversample_minority": None, "undersample_majority": None},
    }
    df = pd.DataFrame(
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
    target = config["output_features"][0][NAME]

    run_test_balance_data(df, config, target, balance, backend)


def test_non_binary_failure():
    config = {
        "input_features": [
            {"name": "Index", "proc_column": "Index", "type": "numerical"},
            {"name": "random_1", "proc_column": "random_1", "type": "numerical"},
            {"name": "random_2", "proc_column": "random_2", "type": "numerical"},
        ],
        "output_features": [{"name": "Label", "proc_column": "Label", "type": "number"}],
        "preprocessing": {},
    }
    df = pd.DataFrame(
        {
            "Index": np.arange(0, 200, 1),
            "random_1": np.random.randint(0, 50, 200),
            "random_2": np.random.choice(["Type A", "Type B", "Type C", "Type D"], 200),
            "Label": np.concatenate((np.zeros(180), np.ones(20))),
            "split": np.zeros(200),
        }
    )

    backend = LocalBackend()
    target = config["output_features"][0][NAME]

    with pytest.raises(ValueError):
        run_test_balance_data(df, config, target, 0.5, backend)


def test_multiple_class_failure():
    config = {
        "input_features": [
            {"name": "Index", "proc_column": "Index", "type": "numerical"},
            {"name": "random_1", "proc_column": "random_1", "type": "numerical"},
            {"name": "random_2", "proc_column": "random_2", "type": "numerical"},
        ],
        "output_features": [
            {"name": "Label", "proc_column": "Label", "type": "binary"},
            {"name": "Label2", "proc_column": "Label2", "type": "binary"},
        ],
        "preprocessing": {},
    }
    df = pd.DataFrame(
        {
            "Index": np.arange(0, 200, 1),
            "random_1": np.random.randint(0, 50, 200),
            "random_2": np.random.choice(["Type A", "Type B", "Type C", "Type D"], 200),
            "Label": np.concatenate((np.zeros(180), np.ones(20))),
            "Label2": np.concatenate((np.zeros(180), np.ones(20))),
            "split": np.zeros(200),
        }
    )

    backend = LocalBackend()
    target = config["output_features"][0][NAME]

    with pytest.raises(ValueError):
        run_test_balance_data(df, config, target, 0.5, backend)
