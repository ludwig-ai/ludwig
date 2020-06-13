import os.path
import json

import pandas as pd
import numpy as np

import pytest

from ludwig.experiment import full_experiment

@pytest.fixture
def train_df():
    # function generates simple training data tha guarantee convergence
    # within 30 epochs for suitable model definition
    NUMBER_OBSERVATIONS = 200

    # generate training data
    np.random.seed(42)
    x = np.array(range(NUMBER_OBSERVATIONS)).reshape(-1, 1)
    y = 2*x + 1 + np.random.normal(size=x.shape[0]).reshape(-1, 1)

    train_df = pd.DataFrame(np.concatenate((x, y), axis=1), columns=['x', 'y'])

    return train_df

@pytest.mark.parametrize('early_stop', [3, 5])
def test_early_stopping(early_stop, train_df, tmp_path):
    # model definition guarantee convergence in under 30 epochs
    input_features = [
            {'name': 'x', 'type': 'numerical'},
        ]
    output_features = [
        {'name': 'y', 'type': 'numerical', 'loss': {'type': 'mean_squared_error'},
         'num_fc_layers': 5, 'fc_size': 64}
    ]
    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {
            'type': 'concat'
        },
        'training': {
            'epochs': 100,
            'early_stop': early_stop,
            'batch_size': 16
        }
    }

    # create sub-directory to store results
    results_dir = tmp_path / 'results'
    results_dir.mkdir()

    # specify model training options
    kwargs = {
        'output_directory':results_dir,
        'model_definition': model_definition,
        'skip_save_processed_input': True,
        'skip_save_progress': True,
        'skip_save_unprocessed_output': True,
        'skip_save_model': True,
        'skip_save_log': True

    }

    # run experiment
    exp_dir_name = full_experiment(data_df=train_df, **kwargs)

    # test existence of required files
    train_stats_fp = os.path.join(exp_dir_name, 'training_statistics.json')
    metadata_fp = os.path.join(exp_dir_name, 'description.json')
    assert os.path.isfile(train_stats_fp)
    assert os.path.isfile(metadata_fp)

    # retrieve results so we can validate early stopping
    with open(train_stats_fp,'r') as f:
        train_stats = json.load(f)
    with open(metadata_fp, 'r') as f:
        metadata = json.load(f)

    # get early stopping value
    early_stop_value = metadata['model_definition']['training']['early_stop']

    # retrieve validation losses
    vald_losses = np.array(train_stats['validation']['combined']['loss'])
    last_epoch = vald_losses.shape[0]
    best_epoch = np.argmin(vald_losses)

    # confirm early stopping
    assert (last_epoch - best_epoch - 1) == early_stop_value


