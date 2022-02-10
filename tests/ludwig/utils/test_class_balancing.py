import numpy as np
import pandas as pd
import pytest

from ludwig.constants import OUTPUT_FLAG, PROC_COLUMN, SPLIT
from ludwig.data.preprocessing import balance_data, build_dataset, cast_columns, get_split
from ludwig.backend import initialize_backend, RAY
from ludwig.backend.base import LocalBackend
from ludwig.utils.defaults import default_random_seed, merge_with_defaults
from ludwig.api import LudwigModel

DFS = {
    "test_df_1": pd.DataFrame({"Index": np.arange(0, 200, 1),
                               "random_1": np.random.randint(0, 50, 200),
                               "random_2": np.random.choice(['Type A', 'Type B', 'Type C', 'Type D'], 200),
                               "Label": np.random.choice(2, 200, p=[0.9, 0.1])})
}

CONFIGS = {
    "test_config_1": {
        "input_features": [
            {"name": "Index", "column": "Index", "type": "numerical", "output_flag": False},
            {"name": "random_1", "column": "random_1", "type": "numerical", "output_flag": False},
            {"name": "random_2", "column": "random_2", "type": "numerical", "output_flag": False}],
        "output_features": [
            {"name": "Label", "column": "Label", "type": "numerical", "output_flag": True}
        ],
        "preprocessing": {
        }
    }
}

LOCAL_BACKEND = LocalBackend()
REMOTE_BACKEND = initialize_backend(RAY)
TEST_BALANCE_CONFIG = merge_with_defaults(CONFIGS['test_config_1'].copy())
TEST_BUILD_DATA_CONFIG = merge_with_defaults(CONFIGS['test_config_1'].copy())
TEST_TRAIN_CONFIG = CONFIGS['test_config_1'].copy()



@pytest.mark.parametrize(
    "balance, backend",
    [
        ("oversample_minority", LOCAL_BACKEND),
        ("undersample_majority", LOCAL_BACKEND),
        ("oversample_minority", REMOTE_BACKEND),
        ("undersample_majority", REMOTE_BACKEND),
    ],
)
def test_balance_data(balance, backend, config=TEST_BALANCE_CONFIG):
    config[balance] = 0.5
    features = config['input_features'] + config['output_features']
    columns = cast_columns(DFS['test_df_1'], features, backend)
    split = get_split(DFS['test_df_1'])
    columns[SPLIT] = split

    test_proc_cols, test_df = balance_data(DFS['test_df_1'],
                                           columns,
                                           features,
                                           config['preprocessing'],
                                           backend,
                                           )

    for feature in features:
        if feature[OUTPUT_FLAG]:
            target = feature[PROC_COLUMN]
    balanced_train = test_df[test_df.split == 0]
    new_class_balance = round(balanced_train[target].value_counts()[
                                  balanced_train[target].value_counts().idxmin()] / \
                              balanced_train[target].value_counts()[
                                  balanced_train[target].value_counts().idxmax()], 1)

    assert len(test_proc_cols) == 5
    assert new_class_balance == 0.5


@pytest.mark.parametrize(
    "balance, backend",
    [
        ("test_oversample", LOCAL_BACKEND),
        ("test_undersample", LOCAL_BACKEND),
        ("test_oversample", REMOTE_BACKEND),
        ("test_undersample", REMOTE_BACKEND),
    ],
)
def test_build_dataset(balance, backend, config=TEST_BUILD_DATA_CONFIG):
    if balance == "test_oversample":
        config['preprocessing']['oversample_minority'] = 0.5
    else:
        config['preprocessing']['undersample_majority'] = 0.5
    test_dataset, test_metadata = build_dataset(DFS['test_df_1'],
                                                config['input_features'] + config['output_features'],
                                                config['preprocessing'],
                                                metadata=None,
                                                backend=backend,
                                                random_seed=default_random_seed,
                                                skip_save_processed_input=False,
                                                callbacks=None,
                                                mode=None,
                                                )
    assert True is True


@pytest.mark.parametrize(
    "balance, backend",
    [
        ("oversample_minority", LOCAL_BACKEND),
        ("undersample_majority", LOCAL_BACKEND),
        ("oversample_minority", REMOTE_BACKEND),
        ("undersample_majority", REMOTE_BACKEND),
    ],
)
def test_train_full_stack(balance, backend, config=TEST_TRAIN_CONFIG):
    config[balance] = 0.5
    model = LudwigModel(config)
    train_stats, processed_df, url = model.train(DFS['test_df_1'])

    assert len(DFS['test_df_1']) < len(processed_df)

