import os
import shutil
import tempfile

import pytest

from ludwig.api import LudwigModel
from tests.integration_tests.utils import binary_feature, category_feature, generate_data, number_feature, text_feature

# TODO: parametrize to include ray backend


def test_gbm_output_not_supported(tmpdir):
    """Test that an error is raised when the output feature is not supported by the model."""
    with tempfile.TemporaryDirectory() as outdir:
        input_features = [number_feature(), category_feature(reduce_output="sum")]
        output_features = [text_feature()]

        csv_filename = os.path.join(tmpdir, "training.csv")
        data_csv = generate_data(input_features, output_features, csv_filename)
        val_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "validation.csv"))
        test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "test.csv"))

        config = {"model_type": "gbm", "input_features": input_features, "output_features": output_features}

        model = LudwigModel(config)
        with pytest.raises(ValueError, match="Output feature must be numerical, categorical, or binary"):
            model.train(training_set=data_csv, validation_set=val_csv, test_set=test_csv, output_directory=outdir)


def test_gbm_multiple_outputs(tmpdir):
    """Test that an error is raised when the model is trained with multiple outputs."""
    with tempfile.TemporaryDirectory() as outdir:
        input_features = [number_feature(), category_feature(reduce_output="sum")]
        output_features = [
            category_feature(vocab_size=3),
            binary_feature(),
            category_feature(vocab_size=3),
        ]

        csv_filename = os.path.join(tmpdir, "training.csv")
        data_csv = generate_data(input_features, output_features, csv_filename)
        val_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "validation.csv"))
        test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "test.csv"))

        config = {
            "model_type": "gbm",
            "input_features": input_features,
            "output_features": output_features,
            "trainer": {"num_boosting_rounds": 1},
        }

        model = LudwigModel(config)
        with pytest.raises(ValueError, match="Only single task currently supported"):
            model.train(
                training_set=data_csv,
                validation_set=val_csv,
                test_set=test_csv,
                output_directory=outdir,
            )


def test_gbm_binary(tmpdir):
    """Test that the GBM model can train and predict a binary variable."""
    with tempfile.TemporaryDirectory() as outdir:
        input_features = [number_feature(), category_feature(reduce_output="sum")]
        output_feature = binary_feature()
        output_features = [output_feature]

        csv_filename = os.path.join(tmpdir, "training.csv")
        data_csv = generate_data(input_features, output_features, csv_filename)
        val_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "validation.csv"))
        test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "test.csv"))

        config = {
            "model_type": "gbm",
            "input_features": input_features,
            "output_features": output_features,
            "trainer": {"num_boosting_rounds": 1},
        }

        model = LudwigModel(config)
        _, _, output_directory = model.train(
            training_set=data_csv,
            validation_set=val_csv,
            test_set=test_csv,
            output_directory=outdir,
        )
        preds, _ = model.predict(dataset=test_csv, output_directory=output_directory)

    prob_col = preds[output_feature["name"] + "_probabilities"]
    assert len(prob_col[0]) == 2
    assert prob_col.apply(sum).mean() == pytest.approx(1.0)


def test_gbm_category(tmpdir):
    """Test that the GBM model can train and predict a categorical output."""
    with tempfile.TemporaryDirectory() as outdir:
        input_features = [number_feature(), category_feature(reduce_output="sum")]
        vocab_size = 3
        output_feature = category_feature(vocab_size=vocab_size)
        output_features = [output_feature]

        csv_filename = os.path.join(tmpdir, "training.csv")
        data_csv = generate_data(input_features, output_features, csv_filename)
        val_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "validation.csv"))
        test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "test.csv"))

        config = {
            "model_type": "gbm",
            "input_features": input_features,
            "output_features": output_features,
            "trainer": {"num_boosting_rounds": 1},
        }

        model = LudwigModel(config)
        _, _, output_directory = model.train(
            training_set=data_csv,
            validation_set=val_csv,
            test_set=test_csv,
            output_directory=outdir,
        )
        preds, _ = model.predict(dataset=test_csv, output_directory=output_directory)

    prob_col = preds[output_feature["name"] + "_probabilities"]
    assert len(prob_col[0]) == (vocab_size + 1)
    assert prob_col.apply(sum).mean() == pytest.approx(1.0)
