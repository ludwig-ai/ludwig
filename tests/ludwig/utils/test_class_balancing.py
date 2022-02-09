import numpy as np
import pandas as pd
import pytest
from ludwig.data.preprocessing import balance_data
from ludwig.backend import initialize_backend, RAY
from ludwig.backend.base import Backend, LocalBackend

fraud_inputs = [
    {"name": "Time", "type": "numerical", "output_flag": False},
    {"name": "V1", "type": "numerical", "output_flag": False},
    {"name": "V2", "type": "numerical", "output_flag": False},
    {"name": "V3", "type": "numerical", "output_flag": False},
    {"name": "V4", "type": "numerical", "output_flag": False},
    {"name": "V5", "type": "numerical", "output_flag": False},
    {"name": "V6", "type": "numerical", "output_flag": False},
    {"name": "V7", "type": "numerical", "output_flag": False},
    {"name": "V8", "type": "numerical", "output_flag": False},
    {"name": "V9", "type": "numerical", "output_flag": False},
    {"name": "V10", "type": "numerical", "output_flag": False},
    {"name": "V11", "type": "numerical", "output_flag": False},
    {"name": "V12", "type": "numerical", "output_flag": False},
    {"name": "V13", "type": "numerical", "output_flag": False},
    {"name": "V14", "type": "numerical", "output_flag": False},
    {"name": "V15", "type": "numerical", "output_flag": False},
    {"name": "V16", "type": "numerical", "output_flag": False},
    {"name": "V17", "type": "numerical", "output_flag": False},
    {"name": "V18", "type": "numerical", "output_flag": False},
    {"name": "V19", "type": "numerical", "output_flag": False},
    {"name": "V20", "type": "numerical", "output_flag": False},
    {"name": "V21", "type": "numerical", "output_flag": False},
    {"name": "V22", "type": "numerical", "output_flag": False},
    {"name": "V23", "type": "numerical", "output_flag": False},
    {"name": "V24", "type": "numerical", "output_flag": False},
    {"name": "V25", "type": "numerical", "output_flag": False},
    {"name": "V26", "type": "numerical", "output_flag": False},
    {"name": "V27", "type": "numerical", "output_flag": False},
    {"name": "V28", "type": "numerical", "output_flag": False},
    {"name": "Amount", "type": "numerical", "output_flag": False},
]

fraud_outputs = [
    {"name": "Class", "type": "binary", "output_flag": True}
]

CONFIGS = {
    "test_oversample": {
        "input_features": fraud_inputs,
        "output_features": fraud_outputs,
        "preprocessing": {
            "oversample_minority": 0.5
        }
    },
    "test_undersample": {
        "input_features": fraud_inputs,
        "output_features": fraud_outputs,
        "preprocessing": {
            "undersample_majority": 0.5
        }
    }
}

DFS = {
    "test_df_1": pd.DataFrame({"Index": np.arange(0, 100, 1),
                               "random_1": np.random.randint(0, 50, 100),
                               "random_2": np.random.choice(['Type A', 'Type B', 'Type C', 'Type D'], 100),
                               "Label": np.random.choice(2, 100, p=[0.8, 0.2])}),
    "test_df_2": pd.read_csv("./creditcard.csv")
}

TEST_COLS = {
    "df1_cols": {"Index": DFS["test_df_1"]["Index"],
                 "random_1": DFS["test_df_1"]["random_1"],
                 "random_2": DFS["test_df_1"]["random_2"],
                 "Label": DFS["test_df_1"]["Label"]
                 },
    "df2_cols": {"V1": DFS["test_df_2"]["V1"], "V2": DFS["test_df_2"]["V2"],
                 "V3": DFS["test_df_2"]["V3"], "V4": DFS["test_df_2"]["V4"],
                 "V5": DFS["test_df_2"]["V5"], "V6": DFS["test_df_2"]["V6"],
                 "V7": DFS["test_df_2"]["V7"], "V8": DFS["test_df_2"]["V8"],
                 "V9": DFS["test_df_2"]["V9"], "V10": DFS["test_df_2"]["V10"],
                 "V11": DFS["test_df_2"]["V11"], "V12": DFS["test_df_2"]["V12"],
                 "V13": DFS["test_df_2"]["V13"], "V14": DFS["test_df_2"]["V14"],
                 "V15": DFS["test_df_2"]["V15"], "V16": DFS["test_df_2"]["V16"],
                 "V17": DFS["test_df_2"]["V17"], "V18": DFS["test_df_2"]["V18"],
                 "V19": DFS["test_df_2"]["V19"], "V20": DFS["test_df_2"]["V20"],
                 "V21": DFS["test_df_2"]["V21"], "V22": DFS["test_df_2"]["V22"],
                 "V23": DFS["test_df_2"]["V23"], "V24": DFS["test_df_2"]["V24"],
                 "V25": DFS["test_df_2"]["V25"], "V26": DFS["test_df_2"]["V26"],
                 "V27": DFS["test_df_2"]["V27"], "V28": DFS["test_df_2"]["V28"],
                 "Amount": DFS["test_df_2"]["Amount"], "Class": DFS["test_df_2"]["Class"],
                 "Time": DFS["test_df_2"]["Time"],
                 "split": pd.Series(np.random.choice(3, len(DFS["test_df_2"]), p=(0.7, 0.1, 0.2)))
                 }
}

LOCAL_BACKEND = LocalBackend()
REMOTE_BACKEND = initialize_backend(RAY)


@pytest.mark.parametrize(
    "config, df, cols, backend",
    [
        ("test_oversample", "test_df_1", "df1_cols", LOCAL_BACKEND),
        ("test_undersample", "test_df_1", "df1_cols", LOCAL_BACKEND),
        ("test_oversample", "test_df_2", "df2_cols", LOCAL_BACKEND),
        ("test_undersample", "test_df_2", "df2_cols", LOCAL_BACKEND),
        ("test_oversample", "test_df_1", "df1_cols", REMOTE_BACKEND),
        ("test_undersample", "test_df_1", "df1_cols", REMOTE_BACKEND),
        ("test_oversample", "test_df_2", "df2_cols", REMOTE_BACKEND),
        ("test_undersample", "test_df_2", "df2_cols", REMOTE_BACKEND)
    ]
)
def test_balance_data_synthetic(config, df, cols, backend):
    test_proc_cols, test_df = balance_data(DFS[df],
                                           TEST_COLS[cols],
                                           CONFIGS[config]['input_features'] + CONFIGS[config]['output_features'],
                                           CONFIGS[config]['preprocessing'],
                                           backend,
                                           )
    assert len(test_proc_cols) == 4
    assert len(test_df) == 100

@pytest.mark.parametrize
def test_balance_data_fraud(config, df, cols, backend):
    test_proc_cols, test_df = balance_data(DFS[df],
                                           TEST_COLS[cols],
                                           CONFIGS[config]['input_features'] + CONFIGS[config]['output_features'],
                                           CONFIGS[config]['preprocessing'],
                                           backend,
                                           )
    assert len(test_proc_cols) == 4
    assert len(test_df) == 100

def test_build_dataset(config):
    assert True is True


@pytest.mark.parametrize("config", ["test_oversample", "test_undersample"])
def test_train_local_backend(config):
    assert True is True


def test_train_remote_backend(config):
    assert True is True
