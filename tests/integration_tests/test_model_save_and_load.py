import os.path
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf

from ludwig.api import LudwigModel
from ludwig.constants import SPLIT
from ludwig.data.preprocessing import get_split
from ludwig.utils.data_utils import split_dataset_ttv, read_csv
from tests.integration_tests.utils import binary_feature, numerical_feature, \
    category_feature, sequence_feature, date_feature, h3_feature, \
    set_feature, generate_data, text_feature, vector_feature, bag_feature, \
    image_feature, audio_feature, timeseries_feature


def test_model_save_reload_api(csv_filename, tmp_path):
    tf.random.set_seed(1234)

    image_dest_folder = os.path.join(os.getcwd(), 'generated_images')
    audio_dest_folder = os.path.join(os.getcwd(), 'generated_audio')

    input_features = [
        binary_feature(),
        numerical_feature(),
        category_feature(vocab_size=3),
        sequence_feature(vocab_size=3),
        text_feature(vocab_size=3, encoder='rnn', cell_type='lstm',
                     num_layers=2, bidirections=True),
        vector_feature(),
        image_feature(image_dest_folder),
        audio_feature(audio_dest_folder, encoder='stacked_cnn'),
        timeseries_feature(encoder='parallel_cnn'),
        sequence_feature(vocab_size=3, encoder='stacked_parallel_cnn'),
        date_feature(),
        h3_feature(),
        set_feature(vocab_size=3),
        bag_feature(vocab_size=3),
    ]

    output_features = [
        binary_feature(),
        numerical_feature(),
        category_feature(vocab_size=3),
        sequence_feature(vocab_size=3),
        text_feature(vocab_size=3),
        set_feature(vocab_size=3),
        vector_feature(),
    ]

    # Generate test data
    data_csv_path = generate_data(input_features, output_features,
                                  csv_filename)

    #############
    # Train model
    #############
    config = {
        'input_features': input_features,
        'output_features': output_features,
        'training': {'epochs': 2}
    }

    data_df = read_csv(data_csv_path)
    data_df[SPLIT] = get_split(data_df)
    training_set, test_set, validation_set = split_dataset_ttv(
        data_df,
        SPLIT
    )
    training_set = pd.DataFrame(training_set)
    validation_set = pd.DataFrame(validation_set)
    test_set = pd.DataFrame(test_set)

    # create sub-directory to store results
    results_dir = tmp_path / 'results'
    results_dir.mkdir()

    # perform initial model training
    ludwig_model1 = LudwigModel(config)
    _, _, output_dir = ludwig_model1.train(
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        output_directory='results'  # results_dir
    )

    preds_1, _ = ludwig_model1.predict(dataset=validation_set)

    def check_model_equal(ludwig_model2):
        # Compare model predictions
        preds_2, _ = ludwig_model2.predict(dataset=validation_set)
        assert set(preds_1.keys()) == set(preds_2.keys())
        for key in preds_1:
            assert preds_1[key].dtype == preds_2[key].dtype, key
            assert list(preds_1[key]) == list(preds_2[key]), key
            # assert preds_2[key].dtype == preds_3[key].dtype, key
            # assert list(preds_2[key]) == list(preds_3[key]), key

        # Compare model weights
        # this has to be done after predicts because of TF2 lazy restoration
        for if_name in ludwig_model1.model.input_features:
            if1 = ludwig_model1.model.input_features[if_name]
            if2 = ludwig_model2.model.input_features[if_name]
            for if1_w, if2_w in zip(if1.encoder_obj.weights,
                                    if2.encoder_obj.weights):
                assert np.allclose(if1_w.numpy(), if2_w.numpy())

        c1 = ludwig_model1.model.combiner
        c2 = ludwig_model2.model.combiner
        for c1_w, c2_w in zip(c1.weights, c2.weights):
            assert np.allclose(c1_w.numpy(), c2_w.numpy())

        for of_name in ludwig_model1.model.output_features:
            of1 = ludwig_model1.model.output_features[of_name]
            of2 = ludwig_model2.model.output_features[of_name]
            for of1_w, of2_w in zip(of1.decoder_obj.weights,
                                    of2.decoder_obj.weights):
                assert np.allclose(of1_w.numpy(), of2_w.numpy())

    # Test saving and loading the model explicitly
    with tempfile.TemporaryDirectory() as tmpdir:
        ludwig_model1.save(tmpdir)
        ludwig_model_loaded = LudwigModel.load(tmpdir)
        check_model_equal(ludwig_model_loaded)

    # Test loading the model from the experiment directory
    ludwig_model_exp = LudwigModel.load(
        os.path.join(output_dir, 'model')
    )
    check_model_equal(ludwig_model_exp)
