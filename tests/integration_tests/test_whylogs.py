import os
import shutil

import mlflow
import pandas as pd
import yaml

from ludwig.api import LudwigModel
from ludwig.contribs import WhyLogsCallback
from tests.integration_tests.utils import sequence_feature, category_feature, generate_data


def test_whylogs_callback(tmpdir):
    epochs = 2
    batch_size = 8
    num_examples = 32

    input_features = [sequence_feature(reduce_output='sum')]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    config = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat', 'fc_size': 14},
        'training': {'epochs': epochs, 'batch_size': batch_size},
    }

    data_csv = generate_data(input_features, output_features,
                             os.path.join(tmpdir, 'train.csv'),
                             num_examples=num_examples)
    val_csv = shutil.copyfile(data_csv,
                              os.path.join(tmpdir, 'validation.csv'))
    test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, 'test.csv'))

    exp_name = 'whylogs_test'
    callback = WhyLogsCallback()

    model = LudwigModel(config, callbacks=[callback])
    model.train(training_set=data_csv,
                validation_set=val_csv,
                test_set=test_csv,
                experiment_name=exp_name)
    expected_df, _ = model.predict(test_csv)

    # Check whylogs initialization
    assert callback.session is not None
    assert callback.session.is_active() is True

    local_training_output_dir = 'output/training'
    local_prediction_output_dir = 'output/prediction'

    assert os.path.isdir(local_training_output_dir) is True
    assert os.path.isdir(local_prediction_output_dir) is True
