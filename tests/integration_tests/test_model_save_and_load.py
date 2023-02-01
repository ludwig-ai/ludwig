import os
import os.path
import random

import numpy as np
import pandas as pd
import pytest
import torch

from ludwig.api import LudwigModel
from ludwig.constants import BATCH_SIZE, ENCODER, LOSS, NAME, PREPROCESSING, TRAINER, TRAINING, TYPE
from ludwig.data.split import get_splitter
from ludwig.modules.loss_modules import MSELoss
from ludwig.utils.data_utils import read_csv
from tests.integration_tests.utils import (
    audio_feature,
    bag_feature,
    binary_feature,
    category_feature,
    date_feature,
    generate_data,
    h3_feature,
    image_feature,
    LocalTestBackend,
    number_feature,
    sequence_feature,
    set_feature,
    text_feature,
    timeseries_feature,
    vector_feature,
)


def test_model_save_reload_api(tmpdir, csv_filename, tmp_path):
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    image_dest_folder = os.path.join(os.getcwd(), "generated_images")
    audio_dest_folder = os.path.join(os.getcwd(), "generated_audio")

    input_features = [
        binary_feature(),
        number_feature(),
        category_feature(encoder={"vocab_size": 3}),
        sequence_feature(encoder={"vocab_size": 3}),
        text_feature(
            encoder={"vocab_size": 3, "type": "rnn", "cell_type": "lstm", "num_layers": 2, "bidirectional": False}
        ),
        vector_feature(),
        image_feature(image_dest_folder, encoder={"type": "mlp_mixer", "patch_size": 12}),
        audio_feature(audio_dest_folder, encoder={"type": "stacked_cnn"}),
        timeseries_feature(encoder={"type": "parallel_cnn"}),
        sequence_feature(encoder={"vocab_size": 3, "type": "stacked_parallel_cnn"}),
        date_feature(),
        h3_feature(),
        set_feature(encoder={"vocab_size": 3}),
        bag_feature(encoder={"vocab_size": 3}),
    ]

    output_features = [
        binary_feature(),
        number_feature(),
        category_feature(decoder={"vocab_size": 3}, output_feature=True),
        sequence_feature(decoder={"vocab_size": 3}, output_feature=True),
        text_feature(decoder={"vocab_size": 3}, output_feature=True),
        set_feature(decoder={"vocab_size": 3}, output_feature=True),
        vector_feature(),
    ]

    # Generate test data
    data_csv_path = generate_data(input_features, output_features, csv_filename, num_examples=50)

    #############
    # Train model
    #############
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }

    data_df = read_csv(data_csv_path)
    splitter = get_splitter("random")
    training_set, validation_set, test_set = splitter.split(data_df, LocalTestBackend())

    # create sub-directory to store results
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # perform initial model training
    backend = LocalTestBackend()
    ludwig_model1 = LudwigModel(config, backend=backend)
    _, _, output_dir = ludwig_model1.train(
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        output_directory="results",  # results_dir
    )

    preds_1, _ = ludwig_model1.predict(dataset=validation_set)

    def check_model_equal(ludwig_model2):
        # Compare model predictions
        preds_2, _ = ludwig_model2.predict(dataset=validation_set)
        assert set(preds_1.keys()) == set(preds_2.keys())
        for key in preds_1:
            assert preds_1[key].dtype == preds_2[key].dtype, key
            assert np.all(a == b for a, b in zip(preds_1[key], preds_2[key])), key
            # assert preds_2[key].dtype == preds_3[key].dtype, key
            # assert list(preds_2[key]) == list(preds_3[key]), key

        # Compare model weights
        for if_name in ludwig_model1.model.input_features:
            if1 = ludwig_model1.model.input_features[if_name]
            if2 = ludwig_model2.model.input_features[if_name]
            for if1_w, if2_w in zip(if1.encoder_obj.parameters(), if2.encoder_obj.parameters()):
                assert torch.allclose(if1_w, if2_w)

        c1 = ludwig_model1.model.combiner
        c2 = ludwig_model2.model.combiner
        for c1_w, c2_w in zip(c1.parameters(), c2.parameters()):
            assert torch.allclose(c1_w, c2_w)

        for of_name in ludwig_model1.model.output_features:
            of1 = ludwig_model1.model.output_features[of_name]
            of2 = ludwig_model2.model.output_features[of_name]
            for of1_w, of2_w in zip(of1.decoder_obj.parameters(), of2.decoder_obj.parameters()):
                assert torch.allclose(of1_w, of2_w)

    ludwig_model1.save(tmpdir)
    ludwig_model_loaded = LudwigModel.load(tmpdir, backend=backend)
    check_model_equal(ludwig_model_loaded)

    # Test loading the model from the experiment directory
    ludwig_model_exp = LudwigModel.load(os.path.join(output_dir, "model"), backend=backend)
    check_model_equal(ludwig_model_exp)


def test_gbm_model_save_reload_api(tmpdir, csv_filename, tmp_path):
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    input_features = [
        binary_feature(),
        number_feature(),
        category_feature(encoder={"type": "passthrough", "vocab_size": 3}),
    ]
    output_features = [category_feature(decoder={"vocab_size": 3}, output_feature=True)]

    # Generate test data
    data_csv_path = generate_data(input_features, output_features, csv_filename)

    #############
    # Train tree model
    #############
    config = {
        "model_type": "gbm",
        "input_features": input_features,
        "output_features": output_features,
        # Disable feature filtering to avoid having no features due to small test dataset,
        # see https://stackoverflow.com/a/66405983/5222402
        TRAINER: {"num_boost_round": 2, "feature_pre_filter": False},
    }

    data_df = read_csv(data_csv_path)
    splitter = get_splitter("random")
    training_set, validation_set, test_set = splitter.split(data_df, LocalTestBackend())

    # create sub-directory to store results
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # perform initial model training
    backend = LocalTestBackend()
    ludwig_model1 = LudwigModel(config, backend=backend)
    _, _, output_dir = ludwig_model1.train(
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        output_directory="results",  # results_dir
    )

    preds_1, _ = ludwig_model1.predict(dataset=validation_set)

    def check_model_equal(ludwig_model2):
        # Compare model predictions
        preds_2, _ = ludwig_model2.predict(dataset=validation_set)
        assert set(preds_1.keys()) == set(preds_2.keys())
        for key in preds_1:
            assert preds_1[key].dtype == preds_2[key].dtype, key
            assert np.all(a == b for a, b in zip(preds_1[key], preds_2[key])), key

        # Compare model weights
        for if_name in ludwig_model1.model.input_features:
            if1 = ludwig_model1.model.input_features[if_name]
            if2 = ludwig_model2.model.input_features[if_name]
            for if1_w, if2_w in zip(if1.encoder_obj.parameters(), if2.encoder_obj.parameters()):
                assert torch.allclose(if1_w, if2_w)

        tree1 = ludwig_model1.model
        tree2 = ludwig_model2.model

        with tree1.compile():
            tree1_params = tree1.compiled_model.parameters()

        with tree2.compile():
            tree2_params = tree2.compiled_model.parameters()

        for t1_w, t2_w in zip(tree1_params, tree2_params):
            assert torch.allclose(t1_w, t2_w)

        for of_name in ludwig_model1.model.output_features:
            of1 = ludwig_model1.model.output_features[of_name]
            of2 = ludwig_model2.model.output_features[of_name]
            for of1_w, of2_w in zip(of1.decoder_obj.parameters(), of2.decoder_obj.parameters()):
                assert torch.allclose(of1_w, of2_w)

    # Test saving and loading the model explicitly
    ludwig_model1.save(tmpdir)
    ludwig_model_loaded = LudwigModel.load(tmpdir, backend=backend)
    check_model_equal(ludwig_model_loaded)

    # Test loading the model from the experiment directory
    ludwig_model_exp = LudwigModel.load(os.path.join(output_dir, "model"), backend=backend)
    check_model_equal(ludwig_model_exp)


def test_model_weights_match_training(tmpdir, csv_filename):
    np.random.seed(1)

    input_features = [number_feature()]
    output_features = [number_feature()]
    output_feature_name = output_features[0][NAME]

    # Generate test data
    data_csv_path = generate_data(input_features, output_features, os.path.join(tmpdir, csv_filename), num_examples=100)

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "trainer": {
            "epochs": 5,
            "batch_size": 32,
            "evaluate_training_set": True,  # needed to ensure exact training metrics computed
        },
    }

    model = LudwigModel(
        config=config,
    )

    training_stats, _, _ = model.train(training_set=data_csv_path, random_seed=1919)

    # generate predicitons from training data
    df = pd.read_csv(data_csv_path)
    predictions = model.predict(df)

    # compute loss on predictions from training data
    loss_function = MSELoss()
    loss = loss_function(
        torch.tensor(predictions[0][output_feature_name + "_predictions"].values),  # predictions
        torch.tensor(df[output_feature_name].values),  # target
    ).type(torch.float32)

    # get last loss value from training
    last_training_loss = torch.tensor(training_stats[TRAINING][output_feature_name][LOSS][-1])

    # loss from predictions should match last loss value recorded during training
    assert torch.isclose(loss, last_training_loss), (
        "Model predictions on training set did not generate same loss value as in training. "
        "Need to confirm that weights were correctly captured in model."
    )


@pytest.mark.parametrize("torch_encoder, variant", [("resnet", 18), ("googlenet", "base")])
def test_model_save_reload_tv_model(torch_encoder, variant, tmpdir, csv_filename, tmp_path):
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    image_dest_folder = os.path.join(os.getcwd(), "generated_images")

    input_features = [
        image_feature(image_dest_folder),
    ]
    input_features[0][ENCODER] = {
        TYPE: torch_encoder,
        "model_variant": variant,
    }
    input_features[0][PREPROCESSING]["height"] = 128
    input_features[0][PREPROCESSING]["width"] = 128

    output_features = [
        category_feature(decoder={"vocab_size": 3}),
    ]

    # Generate test data
    data_csv_path = generate_data(input_features, output_features, csv_filename, num_examples=50)

    #############
    # Train model
    #############
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }

    data_df = read_csv(data_csv_path)
    splitter = get_splitter("random")
    training_set, validation_set, test_set = splitter.split(data_df, LocalTestBackend())

    # create sub-directory to store results
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # perform initial model training
    backend = LocalTestBackend()
    ludwig_model1 = LudwigModel(config, backend=backend)
    _, _, output_dir = ludwig_model1.train(
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        output_directory="results",  # results_dir
    )

    preds_1, _ = ludwig_model1.predict(dataset=validation_set)

    def check_model_equal(ludwig_model2):
        # Compare model predictions
        preds_2, _ = ludwig_model2.predict(dataset=validation_set)
        assert set(preds_1.keys()) == set(preds_2.keys())
        for key in preds_1:
            assert preds_1[key].dtype == preds_2[key].dtype, key
            assert np.all(a == b for a, b in zip(preds_1[key], preds_2[key])), key
            # assert preds_2[key].dtype == preds_3[key].dtype, key
            # assert list(preds_2[key]) == list(preds_3[key]), key

        # Compare model weights
        for if_name in ludwig_model1.model.input_features:
            if1 = ludwig_model1.model.input_features[if_name]
            if2 = ludwig_model2.model.input_features[if_name]
            for if1_w, if2_w in zip(if1.encoder_obj.parameters(), if2.encoder_obj.parameters()):
                assert torch.allclose(if1_w, if2_w)

        c1 = ludwig_model1.model.combiner
        c2 = ludwig_model2.model.combiner
        for c1_w, c2_w in zip(c1.parameters(), c2.parameters()):
            assert torch.allclose(c1_w, c2_w)

        for of_name in ludwig_model1.model.output_features:
            of1 = ludwig_model1.model.output_features[of_name]
            of2 = ludwig_model2.model.output_features[of_name]
            for of1_w, of2_w in zip(of1.decoder_obj.parameters(), of2.decoder_obj.parameters()):
                assert torch.allclose(of1_w, of2_w)

    ludwig_model1.save(tmpdir)
    ludwig_model_loaded = LudwigModel.load(tmpdir, backend=backend)

    # confirm model structure and weights are the same
    check_model_equal(ludwig_model_loaded)

    # Test loading the model from the experiment directory
    ludwig_model_exp = LudwigModel.load(os.path.join(output_dir, "model"), backend=backend)

    # confirm model structure and weights are the same
    check_model_equal(ludwig_model_exp)


@pytest.mark.slow
def test_model_save_reload_hf_model(tmpdir, csv_filename, tmp_path):
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    input_features = [
        text_feature(
            encoder={
                "vocab_size": 3,
                "type": "bert",
            }
        ),
    ]

    output_features = [
        category_feature(decoder={"vocab_size": 3}),
    ]

    # Generate test data
    data_csv_path = generate_data(input_features, output_features, csv_filename, num_examples=50)

    #############
    # Train model
    #############
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }

    data_df = read_csv(data_csv_path)
    splitter = get_splitter("random")
    training_set, validation_set, test_set = splitter.split(data_df, LocalTestBackend())

    # create sub-directory to store results
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # perform initial model training
    backend = LocalTestBackend()
    ludwig_model1 = LudwigModel(config, backend=backend)
    _, _, output_dir = ludwig_model1.train(
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        output_directory="results",  # results_dir
    )

    preds_1, _ = ludwig_model1.predict(dataset=validation_set)

    def check_model_equal(ludwig_model2):
        # Compare model predictions
        preds_2, _ = ludwig_model2.predict(dataset=validation_set)
        assert set(preds_1.keys()) == set(preds_2.keys())
        for key in preds_1:
            assert preds_1[key].dtype == preds_2[key].dtype, key
            assert np.all(a == b for a, b in zip(preds_1[key], preds_2[key])), key
            # assert preds_2[key].dtype == preds_3[key].dtype, key
            # assert list(preds_2[key]) == list(preds_3[key]), key

        # Compare model weights
        for if_name in ludwig_model1.model.input_features:
            if1 = ludwig_model1.model.input_features[if_name]
            if2 = ludwig_model2.model.input_features[if_name]
            for if1_w, if2_w in zip(if1.encoder_obj.parameters(), if2.encoder_obj.parameters()):
                assert torch.allclose(if1_w, if2_w)

        c1 = ludwig_model1.model.combiner
        c2 = ludwig_model2.model.combiner
        for c1_w, c2_w in zip(c1.parameters(), c2.parameters()):
            assert torch.allclose(c1_w, c2_w)

        for of_name in ludwig_model1.model.output_features:
            of1 = ludwig_model1.model.output_features[of_name]
            of2 = ludwig_model2.model.output_features[of_name]
            for of1_w, of2_w in zip(of1.decoder_obj.parameters(), of2.decoder_obj.parameters()):
                assert torch.allclose(of1_w, of2_w)

    ludwig_model1.save(tmpdir)
    ludwig_model_loaded = LudwigModel.load(tmpdir, backend=backend)

    # confirm model structure and weights are the same
    check_model_equal(ludwig_model_loaded)

    # Test loading the model from the experiment directory
    ludwig_model_exp = LudwigModel.load(os.path.join(output_dir, "model"), backend=backend)

    # confirm model structure and weights are the same
    check_model_equal(ludwig_model_exp)
