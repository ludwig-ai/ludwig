import json
import os.path
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal
from sklearn.model_selection import train_test_split

from ludwig.api import LudwigModel
from ludwig.experiment import full_experiment

GeneratedData = namedtuple('GeneratedData',
                           'train_df validation_df test_df')


def get_feature_definitions():
    input_features = [
        {'name': 'x', 'type': 'numerical'},
        ]
    output_features = [
        {'name': 'y', 'type': 'numerical', 'loss': {'type': 'mean_squared_error'},
         'num_fc_layers': 5, 'fc_size': 64}
    ]

    return input_features, output_features


@pytest.fixture(scope='module')
def generated_data():
    # function generates simple training data that guarantee convergence
    # within 30 epochs for suitable model definition
    NUMBER_OBSERVATIONS = 500

    # generate data
    np.random.seed(43)
    x = np.array(range(NUMBER_OBSERVATIONS)).reshape(-1, 1)
    y = 2*x + 1 + np.random.normal(size=x.shape[0]).reshape(-1, 1)
    raw_df = pd.DataFrame(np.concatenate((x, y), axis=1), columns=['x', 'y'])

    # create training data
    train, valid_test  = train_test_split(raw_df, train_size=0.7)

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
    exp_dir_name = full_experiment(
        data_train_df=generated_data.train_df,
        data_validation_df=generated_data.validation_df,
        data_test_df=generated_data.test_df,
        output_directory=str(results_dir),
        model_definition=model_definition,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True,
        skip_save_model=True,
        skip_save_log=True
    )

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
    exp_dir_name = full_experiment(
        data_train_df=generated_data.train_df,
        data_validation_df=generated_data.validation_df,
        data_test_df=generated_data.test_df,
        output_directory=str(results_dir),
        model_definition=model_definition,
        skip_save_processed_input=True,
        skip_save_progress=skip_save_progress,
        skip_save_unprocessed_output=True,
        skip_save_model=skip_save_model,
        skip_save_log=True
    )

    #========== Check for required result data sets =============
    if skip_save_model:
        assert not os.path.isdir(
            os.path.join(exp_dir_name, 'model', 'model_weights')
        )
    else:
        assert os.path.isdir(
            os.path.join(exp_dir_name, 'model', 'model_weights')
        )

    if skip_save_progress:
        assert not os.path.isdir(
            os.path.join(exp_dir_name, 'model', 'model_weights_progress')
        )
    else:
        assert os.path.isdir(
            os.path.join(exp_dir_name, 'model', 'model_weights_progress')
        )


# work-in-progress
def test_model_save_resume(generated_data, tmp_path):

    input_features, output_features = get_feature_definitions()
    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat'},
        'training': {
            'epochs': 7,
            'early_stop': 1000,
            'batch_size': 16,
            'optimizer': {'type': 'adam'}
        }
    }

    # create sub-directory to store results
    results_dir = tmp_path / 'results'
    results_dir.mkdir()

    exp_dir_name = full_experiment(
        model_definition,
        data_train_df=generated_data.train_df,
        data_validation_df=generated_data.validation_df,
        data_test_df=generated_data.test_df,
        output_directory='results' #results_dir
    )

    y_pred1 = np.load(os.path.join(exp_dir_name, 'y_predictions.npy'))

    model_definition['training']['epochs'] = 15

    full_experiment(
        model_definition,
        data_train_df=generated_data.train_df,
        data_validation_df=generated_data.validation_df,
        data_test_df=generated_data.test_df,
        model_resume_path=exp_dir_name
    )

    y_pred2 = np.load(os.path.join(exp_dir_name, 'y_predictions.npy'))

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
#         data_train_df=generated_data.train_df,
#         data_validation_df=generated_data.validation_df,
#         data_test_df=generated_data.test_df,
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


def test_model_save_reload_API(generated_data, tmp_path):
    input_features, output_features = get_feature_definitions()
    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat'},
        'training': {'epochs': 3, 'batch_size': 16}
    }

    # create sub-directory to store results
    results_dir = tmp_path / 'results'
    results_dir.mkdir()

    # perform initial model training
    ludwig_model1 = LudwigModel(model_definition)
    train_stats = ludwig_model1.train(
        data_train_df=generated_data.train_df,
        data_validation_df=generated_data.validation_df,
        data_test_df=generated_data.test_df,
        output_directory='results'  # results_dir
    )

    preds_1 = ludwig_model1.predict(data_df=generated_data.validation_df)

    # load saved model
    ludwig_model2 = LudwigModel.load(
        os.path.join(ludwig_model1.exp_dir_name, 'model')
    )

    preds_2 = ludwig_model2.predict(data_df=generated_data.validation_df)

    assert_frame_equal(preds_1, preds_2)

    for if_name in ludwig_model1.model.ecd.input_features:
        if1 = ludwig_model1.model.ecd.input_features[if_name]
        if2 = ludwig_model2.model.ecd.input_features[if_name]
        for if1_w, if2_w in zip(if1.encoder_obj.weights,
                                if2.encoder_obj.weights):
            assert np.allclose(if1_w.numpy(), if2_w.numpy())

    c1 = ludwig_model1.model.ecd.combiner
    c2 = ludwig_model2.model.ecd.combiner
    for c1_w, c2_w in zip(c1.weights, c2.weights):
        assert np.allclose(c1_w.numpy(), c2_w.numpy())

    for of_name in ludwig_model1.model.ecd.output_features:
        of1 = ludwig_model1.model.ecd.output_features[of_name]
        of2 = ludwig_model2.model.ecd.output_features[of_name]
        for of1_w, of2_w in zip(of1.decoder_obj.weights,
                                of2.decoder_obj.weights):
            assert np.allclose(of1_w.numpy(), of2_w.numpy())
