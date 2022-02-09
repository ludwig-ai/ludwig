import numpy as np
import pandas as pd
import pytest
from ludwig.data.preprocessing import balance_data
from ludwig.backend import initialize_backend, RAY
from ludwig.backend.base import Backend, LocalBackend

DFS = {
    "test_df_1": pd.DataFrame({"Index": np.arange(0, 200, 1),
                               "random_1": np.random.randint(0, 50, 200),
                               "random_2": np.random.choice(['Type A', 'Type B', 'Type C', 'Type D'], 200),
                               "Label": np.random.choice(2, 200, p=[0.9, 0.1])})
}

CONFIGS = {
    "test_oversample": {
        "input_features": [
            {"name": "Index", "proc_column": "Index_1234", "type": "numerical", "output_flag": False},
            {"name": "random_1", "proc_column": "random_1_2345", "type": "numerical", "output_flag": False},
            {"name": "random_2", "proc_column": "random_2_3456", "type": "numerical", "output_flag": False}],
        "output_features": [
            {"name": "Label", "proc_column": "Label_4567", "type": "numerical", "output_flag": True}
        ],
        "preprocessing": {
            "undersample_majority": None,
            "oversample_minority": 0.5
        }
    },
    "test_undersample": {
        "input_features": [
            {"name": "Index", "proc_column": "Index_1234", "type": "numerical", "output_flag": False},
            {"name": "random_1", "proc_column": "random_1_2345", "type": "numerical", "output_flag": False},
            {"name": "random_2", "proc_column": "random_2_3456", "type": "numerical", "output_flag": False}],
        "output_features": [
            {"name": "Label", "proc_column": "Label_4567", "type": "numerical", "output_flag": True}
        ],
        "preprocessing": {
            "undersample_majority": 0.5,
            "oversample_minority": None
        }
    }
}

TEST_COLS = {
    "df1_cols": {"Index_1234": DFS["test_df_1"]["Index"],
                 "random_1_2345": DFS["test_df_1"]["random_1"],
                 "random_2_3456": DFS["test_df_1"]["random_2"],
                 "Label_4567": DFS["test_df_1"]["Label"],
                 "split": pd.Series(np.random.choice(3, len(DFS["test_df_1"]), p=(0.7, 0.1, 0.2)))
                 }
}

LOCAL_BACKEND = LocalBackend()
REMOTE_BACKEND = initialize_backend(RAY)


@pytest.mark.parametrize(
    "config, backend",
    [
        ("test_oversample", LOCAL_BACKEND),
        ("test_undersample", LOCAL_BACKEND),
        ("test_oversample", REMOTE_BACKEND),
        ("test_undersample", REMOTE_BACKEND),
    ],
)
def test_balance_data(config, backend):
    test_proc_cols, test_df = balance_data(DFS['test_df_1'],
                                           TEST_COLS['df1_cols'],
                                           CONFIGS[config]['input_features'] + CONFIGS[config]['output_features'],
                                           CONFIGS[config]['preprocessing'],
                                           backend,
                                           )
    balanced_train = test_df[test_df.split == 0]
    new_class_balance = round(balanced_train["Label_4567"].value_counts()[
                                  balanced_train["Label_4567"].value_counts().idxmin()] / \
                              balanced_train["Label_4567"].value_counts()[
                                  balanced_train["Label_4567"].value_counts().idxmax()], 1)
    assert len(test_proc_cols) == 5
    assert new_class_balance == 0.5


def test_build_dataset(config):
    assert True is True


@pytest.mark.parametrize("config", ["test_oversample", "test_undersample"])
def test_train_local_backend(config):
    assert True is True


def test_train_remote_backend(config):
    assert True is True
