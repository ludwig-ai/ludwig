import os
import shutil
import tempfile

from ludwig.api import LudwigModel
from ludwig.constants import BATCH_SIZE, EVAL_BATCH_SIZE, LEARNING_RATE, TRAINING
from tests.integration_tests.utils import category_feature, generate_data, LocalTestBackend, sequence_feature


def test_tune_batch_size_and_lr(tmpdir):
    with tempfile.TemporaryDirectory() as outdir:
        input_features = [sequence_feature(reduce_output="sum")]
        output_features = [category_feature(vocab_size=2, reduce_input="sum")]

        csv_filename = os.path.join(tmpdir, "training.csv")
        data_csv = generate_data(input_features, output_features, csv_filename)
        val_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "validation.csv"))
        test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "test.csv"))

        config = {
            "input_features": input_features,
            "output_features": output_features,
            "combiner": {"type": "concat", "output_size": 14},
            "training": {
                "epochs": 2,
                "batch_size": "auto",
                "eval_batch_size": "auto",
                "learning_rate": "auto",
            },
        }

        model = LudwigModel(config, backend=LocalTestBackend())

        # check preconditions
        assert model.config[TRAINING][BATCH_SIZE] == "auto"
        assert model.config[TRAINING][EVAL_BATCH_SIZE] == "auto"
        assert model.config[TRAINING][LEARNING_RATE] == "auto"

        _, _, output_directory = model.train(
            training_set=data_csv, validation_set=val_csv, test_set=test_csv, output_directory=outdir
        )

        def check_postconditions(model):
            # check batch size
            assert model.config[TRAINING][BATCH_SIZE] != "auto"
            assert model.config[TRAINING][BATCH_SIZE] > 1

            assert model.config[TRAINING][EVAL_BATCH_SIZE] != "auto"
            assert model.config[TRAINING][EVAL_BATCH_SIZE] > 1

            assert model.config[TRAINING][BATCH_SIZE] == model.config[TRAINING][EVAL_BATCH_SIZE]

            # check learning rate
            assert model.config[TRAINING][LEARNING_RATE] != "auto"
            assert model.config[TRAINING][LEARNING_RATE] > 0

        check_postconditions(model)

        model = LudwigModel.load(os.path.join(output_directory, "model"))

        # loaded model should retain the tuned params
        check_postconditions(model)
