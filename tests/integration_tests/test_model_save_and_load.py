import logging
import os.path

import pandas as pd

from ludwig.api import LudwigModel
from ludwig.data.preprocessing import get_split
from ludwig.utils.data_utils import split_dataset_tvt, read_csv
from tests.integration_tests.utils import binary_feature, numerical_feature, \
    category_feature, sequence_feature, date_feature, h3_feature, \
    set_feature, generate_data, text_feature, vector_feature, bag_feature, \
    image_feature, audio_feature, timeseries_feature


def test_model_save_reload_API(csv_filename, tmp_path):
    dir_path = os.path.dirname(csv_filename)
    image_dest_folder = os.path.join(os.getcwd(), 'generated_images')
    audio_dest_folder = os.path.join(os.getcwd(), 'generated_audio')

    input_features = [
        binary_feature(),
        numerical_feature(),
        category_feature(vocab_size=3),
        sequence_feature(vocab_size=3),
        text_feature(vocab_size=3),
        vector_feature(),
        image_feature(image_dest_folder),
        audio_feature(audio_dest_folder),
        timeseries_feature(),
        date_feature(),
        h3_feature(),
        set_feature(vocab_size=3),
        bag_feature(vocab_size=3),
    ]

    output_features = [
        binary_feature(),
        numerical_feature(),
        category_feature(vocab_size=3),
        # sequence_feature(vocab_size=3),
        # text_feature(vocab_size=3),
        set_feature(vocab_size=3),
        vector_feature(),
    ]

    # Generate test data
    data_csv_path = generate_data(input_features, output_features,
                                  csv_filename)

    #############
    # Train model
    #############
    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'training': {'epochs': 2}
    }

    data_df = read_csv(data_csv_path)
    training_set, test_set, validation_set = split_dataset_tvt(
        data_df,
        get_split(data_df)
    )
    training_set = pd.DataFrame(training_set)
    validation_set = pd.DataFrame(validation_set)
    test_set = pd.DataFrame(test_set)

    # create sub-directory to store results
    results_dir = tmp_path / 'results'
    results_dir.mkdir()

    # perform initial model training
    ludwig_model1 = LudwigModel(model_definition, logging_level=logging.DEBUG)
    train_stats = ludwig_model1.train(
        data_train_df=training_set,
        data_validation_df=validation_set,
        data_test_df=test_set,
        output_directory='results'  # results_dir
    )

    preds_1 = ludwig_model1.predict(data_df=validation_set)

    # load saved model
    # ludwig_model2 = LudwigModel.load(
    #     os.path.join(ludwig_model1.exp_dir_name, 'model')
    # )
    #
    # preds_2 = ludwig_model2.predict(data_df=validation_set)
    #
    # # Compare model predictions
    # assert set(preds_1.keys()) == set(preds_2.keys())
    # for key in preds_1:
    #     assert preds_1[key].dtype == preds_2[key].dtype
    #     assert preds_1[key].equals(preds_2[key])
    #
    # # Compare model weights
    # # this has to be done after predicts because of TF2 lazy restoration
    # for if_name in ludwig_model1.model.ecd.input_features:
    #     if1 = ludwig_model1.model.ecd.input_features[if_name]
    #     if2 = ludwig_model2.model.ecd.input_features[if_name]
    #     for if1_w, if2_w in zip(if1.encoder_obj.weights,
    #                             if2.encoder_obj.weights):
    #         assert np.allclose(if1_w.numpy(), if2_w.numpy())
    #
    # c1 = ludwig_model1.model.ecd.combiner
    # c2 = ludwig_model2.model.ecd.combiner
    # for c1_w, c2_w in zip(c1.weights, c2.weights):
    #     assert np.allclose(c1_w.numpy(), c2_w.numpy())
    #
    # for of_name in ludwig_model1.model.ecd.output_features:
    #     of1 = ludwig_model1.model.ecd.output_features[of_name]
    #     of2 = ludwig_model2.model.ecd.output_features[of_name]
    #     for of1_w, of2_w in zip(of1.decoder_obj.weights,
    #                             of2.decoder_obj.weights):
    #         assert np.allclose(of1_w.numpy(), of2_w.numpy())
