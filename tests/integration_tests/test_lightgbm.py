import os
import shutil
import tempfile

import pytest

from ludwig.api import LudwigModel
from tests.integration_tests.utils import binary_feature, category_feature, generate_data, number_feature, text_feature


def test_gbm_output_not_supported(tmpdir):
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


def test_gbm(tmpdir):
    with tempfile.TemporaryDirectory() as outdir:
        input_features = [number_feature(), category_feature(reduce_output="sum")]
        output_features = [binary_feature()]

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

    assert preds[output_features[0]["name"] + "_probabilities"].apply(sum).mean() == pytest.approx(1.0)


# TODO: test ray backend
