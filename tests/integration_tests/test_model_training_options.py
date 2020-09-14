import json
import os.path
import re
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from sklearn.model_selection import train_test_split

from ludwig.experiment import experiment_cli
from ludwig.modules.optimization_modules import optimizers_registry
from ludwig.utils.data_utils import load_json

RANDOM_SEED = 42
NUMBER_OBSERVATIONS = 500

GeneratedData = namedtuple('GeneratedData',
                           'train_df validation_df test_df')


def get_feature_definitions():
    input_features = [
        {'name': 'x', 'type': 'numerical'},
    ]
    output_features = [
        {'name': 'y', 'type': 'numerical',
         'loss': {'type': 'mean_squared_error'},
         'num_fc_layers': 5, 'fc_size': 64}
    ]

    return input_features, output_features


@pytest.fixture(scope='module')
def generated_data():
    # function generates simple training data that guarantee convergence
    # within 30 epochs for suitable model definition


    # generate data
    np.random.seed(43)
    x = np.array(range(NUMBER_OBSERVATIONS)).reshape(-1, 1)
    y = 2 * x + 1 + np.random.normal(size=x.shape[0]).reshape(-1, 1)
    raw_df = pd.DataFrame(np.concatenate((x, y), axis=1), columns=['x', 'y'])

    # create training data
    train, valid_test = train_test_split(raw_df, train_size=0.7)

    # create validation and test data
    validation, test = train_test_split(valid_test, train_size=0.5)

    return GeneratedData(train, validation, test)


@pytest.mark.parametrize('early_stop', [3, 5])
def test_early_stopping(early_stop, generated_data, tmp_path):
    input_features, output_features = get_feature_definitions()

    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {
            'type': 'concat'
        },
        'training': {
            'epochs': 30,
            'early_stop': early_stop,
            'batch_size': 16
        }
    }

    # create sub-directory to store results
    results_dir = tmp_path / 'results'
    results_dir.mkdir()

    # run experiment
    model, _, _, _ = experiment_cli(
        training_set=generated_data.train_df,
        validation_set=generated_data.validation_df,
        test_set=generated_data.test_df,
        output_directory=str(results_dir),
        model_definition=model_definition,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True,
        skip_save_model=True,
        skip_save_log=True
    )

    # test existence of required files
    train_stats_fp = os.path.join(model.exp_dir_name, 'training_statistics.json')
    metadata_fp = os.path.join(model.exp_dir_name, 'description.json')
    assert os.path.isfile(train_stats_fp)
    assert os.path.isfile(metadata_fp)

    # retrieve results so we can validate early stopping
    with open(train_stats_fp, 'r') as f:
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


@pytest.mark.parametrize('skip_save_progress', [False, True])
@pytest.mark.parametrize('skip_save_model', [False, True])
def test_model_progress_save(
        skip_save_progress,
        skip_save_model,
        generated_data,
        tmp_path
):
    input_features, output_features = get_feature_definitions()

    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat'},
        'training': {'epochs': 5}
    }

    # create sub-directory to store results
    results_dir = tmp_path / 'results'
    results_dir.mkdir()

    # run experiment
    model, _, _, _ = experiment_cli(
        training_set=generated_data.train_df,
        validation_set=generated_data.validation_df,
        test_set=generated_data.test_df,
        output_directory=str(results_dir),
        model_definition=model_definition,
        skip_save_processed_input=True,
        skip_save_progress=skip_save_progress,
        skip_save_unprocessed_output=True,
        skip_save_model=skip_save_model,
        skip_save_log=True
    )

    # ========== Check for required result data sets =============
    if skip_save_model:
        model_dir = os.path.join(model.exp_dir_name, 'model')
        files = [f for f in os.listdir(model_dir) if
                 re.match(r'model_weights', f)]
        assert len(files) == 0
    else:
        model_dir = os.path.join(model.exp_dir_name, 'model')
        files = [f for f in os.listdir(model_dir) if
                 re.match(r'model_weights', f)]
        # at least one .index and one .data file, but .data may be more
        assert len(files) >= 2
        assert os.path.isfile(
            os.path.join(model.exp_dir_name, 'model', 'checkpoint'))

    if skip_save_progress:
        assert not os.path.isdir(
            os.path.join(model.exp_dir_name, 'model', 'training_checkpoints')
        )
    else:
        assert os.path.isdir(
            os.path.join(model.exp_dir_name, 'model', 'training_checkpoints')
        )


@pytest.mark.parametrize('optimizer', ['sgd', 'adam'])
def test_resume_training(optimizer, generated_data, tmp_path):
    input_features, output_features = get_feature_definitions()
    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat'},
        'training': {
            'epochs': 2,
            'early_stop': 1000,
            'batch_size': 16,
            'optimizer': {'type': optimizer}
        }
    }

    # create sub-directory to store results
    results_dir = tmp_path / 'results'
    results_dir.mkdir()

    model1, _, _, _ = experiment_cli(
        model_definition,
        training_set=generated_data.train_df,
        validation_set=generated_data.validation_df,
        test_set=generated_data.test_df,
        output_directory='results'  # results_dir
    )

    model_definition['training']['epochs'] = 4

    experiment_cli(
        model_definition,
        training_set=generated_data.train_df,
        validation_set=generated_data.validation_df,
        test_set=generated_data.test_df,
        model_resume_path=model1.exp_dir_name
    )

    model2, _, _, _ = experiment_cli(
        model_definition,
        training_set=generated_data.train_df,
        validation_set=generated_data.validation_df,
        test_set=generated_data.test_df,
    )

    # compare learning curves with and without resuming
    ts1 = load_json(os.path.join(model1.exp_dir_name, 'training_statistics.json'))
    ts2 = load_json(os.path.join(model2.exp_dir_name, 'training_statistics.json'))
    print('ts1', ts1)
    print('ts2', ts2)
    assert ts1['training']['combined']['loss'] == ts2['training']['combined'][
        'loss']

    # compare predictions with and without resuming
    y_pred1 = np.load(os.path.join(model1.exp_dir_name, 'y_predictions.npy'))
    y_pred2 = np.load(os.path.join(model2.exp_dir_name, 'y_predictions.npy'))
    print('y_pred1', y_pred1)
    print('y_pred2', y_pred2)
    assert np.all(np.isclose(y_pred1, y_pred2))


# work-in-progress
# def test_model_save_resume(generated_data, tmp_path):
#
#     input_features, output_features = get_feature_definitions()
#     model_definition = {
#         'input_features': input_features,
#         'output_features': output_features,
#         'combiner': {'type': 'concat'},
#         'training': {'epochs': 3, 'batch_size': 16}
#     }
#
#     # create sub-directory to store results
#     results_dir = tmp_path / 'results'
#     results_dir.mkdir()
#
#     # perform inital model training
#     ludwig_model = LudwigModel(model_definition)
#     train_stats = ludwig_model.train(
#         training_set=generated_data.train_df,
#         validation_set=generated_data.validation_df,
#         test_set=generated_data.test_df,
#         output_directory='results' #results_dir
#     )
#
#     # load saved model definition
#     ludwig_model2 = LudwigModel.load(
#         os.path.join(ludwig_model.exp_dir_name, 'model')
#     )
#
#     for _, i_feature in ludwig_model2.model.ecd.input_features.items():
#         i_feature.encoder_obj(None, training=False)
#
#     ludwig_model2.model.ecd.combiner({'x': {'encoder_output': [None]}}, training=False)
#
#     for _, o_feature in ludwig_model2.model.ecd.output_features.items():
#         o_feature.decoder_obj(None, training=False)
#
#     pass


@pytest.mark.parametrize('optimizer_type', optimizers_registry)
def test_optimizers(optimizer_type, generated_data, tmp_path):
    input_features, output_features = get_feature_definitions()

    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {
            'type': 'concat'
        },
        'training': {
            'epochs': 5,
            'batch_size': 16,
            'optimizer': {'type': optimizer_type}
        }
    }

    # create sub-directory to store results
    results_dir = tmp_path / 'results'
    results_dir.mkdir()

    # run experiment
    model, _, _, _ = experiment_cli(
        training_set=generated_data.train_df,
        validation_set=generated_data.validation_df,
        test_set=generated_data.test_df,
        output_directory=str(results_dir),
        model_definition=model_definition,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True,
        skip_save_model=True,
        skip_save_log=True
    )

    # test existence of required files
    train_stats_fp = os.path.join(model.exp_dir_name, 'training_statistics.json')
    metadata_fp = os.path.join(model.exp_dir_name, 'description.json')
    assert os.path.isfile(train_stats_fp)
    assert os.path.isfile(metadata_fp)

    # retrieve results so we can validate early stopping
    with open(train_stats_fp, 'r') as f:
        train_stats = json.load(f)

    # retrieve training losses for first and last epochs
    train_losses = np.array(train_stats['training']['combined']['loss'])
    last_epoch = train_losses.shape[0]

    # ensure train loss for last epoch is less than first epoch
    assert train_losses[last_epoch - 1] < train_losses[0]


def test_regularization(generated_data, tmp_path):
    input_features, output_features = get_feature_definitions()

    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {
            'type': 'concat'
        },
        'training': {
            'epochs': 1,
            'batch_size': 16,
            'regularization_lambda': 1
        }
    }

    # create sub-directory to store results
    results_dir = tmp_path / 'results'
    results_dir.mkdir()

    regularization_losses = []
    for regularizer in [None, 'l1', 'l2', 'l1_l2']:
        tf.keras.backend.clear_session()
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)

        # setup regularization parameters
        model_definition['output_features'][0][
            'weights_regularizer'] = regularizer
        model_definition['output_features'][0][
            'bias_regularizer'] = regularizer
        model_definition['output_features'][0][
            'activity_regularizer'] = regularizer

        # run experiment
        model, _, _, _ = experiment_cli(
            training_set=generated_data.train_df,
            validation_set=generated_data.validation_df,
            test_set=generated_data.test_df,
            output_directory=str(results_dir),
            model_definition=model_definition,
            experiment_name='regularization',
            model_name=str(regularizer),
            skip_save_processed_input=True,
            skip_save_progress=True,
            skip_save_unprocessed_output=True,
            skip_save_model=True,
            skip_save_log=True
        )

        # test existence of required files
        train_stats_fp = os.path.join(model.exp_dir_name, 'training_statistics.json')
        metadata_fp = os.path.join(model.exp_dir_name, 'description.json')
        assert os.path.isfile(train_stats_fp)
        assert os.path.isfile(metadata_fp)

        # retrieve results so we can compare training loss with regularization
        with open(train_stats_fp, 'r') as f:
            train_stats = json.load(f)

        # retrieve training losses for all epochs
        train_losses = np.array(train_stats['training']['combined']['loss'])
        regularization_losses.append(train_losses[0])

    # create a set of losses
    regularization_losses_set = set(regularization_losses)

    # ensure all losses obtained with the different methods are different
    assert len(regularization_losses) == len(regularization_losses_set)
