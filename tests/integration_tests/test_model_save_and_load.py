import os.path
import random
import tempfile

import numpy as np
import pandas as pd
import torch

from ludwig.api import LudwigModel
from ludwig.constants import SPLIT, TRAINER
from ludwig.data.preprocessing import get_split
from ludwig.utils.data_utils import read_csv, split_dataset_ttv
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


def test_model_save_reload_api(csv_filename, tmp_path):
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)

    image_dest_folder = os.path.join(os.getcwd(), "generated_images")
    audio_dest_folder = os.path.join(os.getcwd(), "generated_audio")

    input_features = [
        binary_feature(),
        number_feature(),
        category_feature(vocab_size=3),
        sequence_feature(vocab_size=3),
        text_feature(vocab_size=3, encoder="rnn", cell_type="lstm", num_layers=2, bidirections=True),
        vector_feature(),
        image_feature(image_dest_folder),
        audio_feature(audio_dest_folder, encoder="stacked_cnn"),
        timeseries_feature(encoder="parallel_cnn"),
        sequence_feature(vocab_size=3, encoder="stacked_parallel_cnn"),
        date_feature(),
        h3_feature(),
        set_feature(vocab_size=3),
        bag_feature(vocab_size=3),
    ]

    output_features = [
        binary_feature(),
        number_feature(),
        category_feature(vocab_size=3),
        sequence_feature(vocab_size=3),
        text_feature(vocab_size=3),
        set_feature(vocab_size=3),
        vector_feature(),
    ]

    # Generate test data
    data_csv_path = generate_data(input_features, output_features, csv_filename)

    #############
    # Train model
    #############
    config = {"input_features": input_features, "output_features": output_features, TRAINER: {"epochs": 2}}

    data_df = read_csv(data_csv_path)
    data_df[SPLIT] = get_split(data_df)
    training_set, test_set, validation_set = split_dataset_ttv(data_df, SPLIT)
    training_set = pd.DataFrame(training_set)
    validation_set = pd.DataFrame(validation_set)
    test_set = pd.DataFrame(test_set)

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

    # Test saving and loading the model explicitly
    with tempfile.TemporaryDirectory() as tmpdir:
        ludwig_model1.save(tmpdir)
        ludwig_model_loaded = LudwigModel.load(tmpdir, backend=backend)
        check_model_equal(ludwig_model_loaded)

    # Test loading the model from the experiment directory
    ludwig_model_exp = LudwigModel.load(os.path.join(output_dir, "model"), backend=backend)
    check_model_equal(ludwig_model_exp)
