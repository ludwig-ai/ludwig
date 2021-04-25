import os
import shutil

import pytest

from ludwig.api import LudwigModel
from tests.integration_tests.utils import sequence_feature, category_feature, generate_data


@pytest.mark.parametrize('fs_protocol', ['file'])
def test_remote_training_set(tmpdir, fs_protocol):
        input_features = [sequence_feature(reduce_output='sum')]
        output_features = [category_feature(vocab_size=2, reduce_input='sum')]

        csv_filename = os.path.join(tmpdir, 'training.csv')
        data_csv = generate_data(input_features, output_features, csv_filename)
        val_csv = shutil.copyfile(data_csv,
                                  os.path.join(tmpdir, 'validation.csv'))
        test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, 'test.csv'))

        data_csv = f'{fs_protocol}://{os.path.abspath(data_csv)}'
        val_csv = f'{fs_protocol}://{os.path.abspath(val_csv)}'
        test_csv = f'{fs_protocol}://{os.path.abspath(test_csv)}'

        config = {
            'input_features': input_features,
            'output_features': output_features,
            'combiner': {'type': 'concat', 'fc_size': 14},
        }
        model = LudwigModel(config)
        model.train(training_set=data_csv,
                    validation_set=val_csv,
                    test_set=test_csv)
        model.predict(dataset=test_csv)

        # Train again, this time the HDF5 cache will be used
        model.train(training_set=data_csv,
                    validation_set=val_csv,
                    test_set=test_csv)
