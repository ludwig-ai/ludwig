import os
import shutil
import tempfile

import yaml

from ludwig.api import LudwigModel
from ludwig.backend import initialize_backend
from ludwig.constants import TRAINING, BATCH_SIZE, EVAL_BATCH_SIZE
from tests.integration_tests.utils import sequence_feature, category_feature, generate_data, LocalTestBackend


def test_tune_batch_size(tmpdir):
    with tempfile.TemporaryDirectory() as outdir:
        input_features = [sequence_feature(reduce_output='sum')]
        output_features = [category_feature(vocab_size=2, reduce_input='sum')]

        csv_filename = os.path.join(tmpdir, 'training.csv')
        data_csv = generate_data(input_features, output_features, csv_filename)
        val_csv = shutil.copyfile(data_csv,
                                  os.path.join(tmpdir, 'validation.csv'))
        test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, 'test.csv'))

        config = {
            'input_features': input_features,
            'output_features': output_features,
            'combiner': {'type': 'concat', 'fc_size': 14},
            'training': {
                'epochs': 2,
                'batch_size': 'auto',
                'eval_batch_size': 'auto',
            },
        }

        model = LudwigModel(config, backend=LocalTestBackend())
        _, _, output_directory = model.train(
            training_set=data_csv,
            validation_set=val_csv,
            test_set=test_csv,
            output_directory=outdir
        )

        # Check batch size
        print(model.config)
        assert model.config[TRAINING][BATCH_SIZE] != 'auto'
        assert model.config[TRAINING][BATCH_SIZE] > 1

        assert model.config[TRAINING][EVAL_BATCH_SIZE] != 'auto'
        assert model.config[TRAINING][EVAL_BATCH_SIZE] > 1

        assert model.config[TRAINING][BATCH_SIZE] == model.config[TRAINING][EVAL_BATCH_SIZE]
