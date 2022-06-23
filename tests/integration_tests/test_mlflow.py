import os
import shutil
from unittest import mock

import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

from ludwig.api import LudwigModel
from ludwig.constants import TRAINER
from ludwig.contribs import MlflowCallback
from tests.integration_tests.utils import category_feature, FakeRemoteBackend, generate_data, sequence_feature


def test_mlflow_callback(tmpdir):
    epochs = 2
    batch_size = 8
    num_examples = 32

    input_features = [sequence_feature(reduce_output="sum")]
    output_features = [category_feature(vocab_size=2, reduce_input="sum")]

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {"epochs": epochs, "batch_size": batch_size},
    }

    data_csv = generate_data(
        input_features, output_features, os.path.join(tmpdir, "train.csv"), num_examples=num_examples
    )
    val_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "validation.csv"))
    test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, "test.csv"))

    mlflow_uri = f"file://{tmpdir}/mlruns"
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient(tracking_uri=mlflow_uri)

    exp_name = "mlflow_test"
    callback = MlflowCallback()
    wrapped_callback = mock.Mock(wraps=callback)

    model = LudwigModel(config, callbacks=[wrapped_callback], backend=FakeRemoteBackend())
    model.train(training_set=data_csv, validation_set=val_csv, test_set=test_csv, experiment_name=exp_name)
    expected_df, _ = model.predict(test_csv)

    # Check mlflow artifacts
    assert callback.experiment_id is not None
    assert callback.run is not None

    experiment = mlflow.get_experiment_by_name(exp_name)
    assert experiment.experiment_id == callback.experiment_id

    df = mlflow.search_runs([experiment.experiment_id])
    assert len(df) == 1

    run_id = df.run_id[0]
    assert run_id == callback.run.info.run_id

    run = mlflow.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert wrapped_callback.on_trainer_train_setup.call_count == 1
    assert wrapped_callback.on_trainer_train_teardown.call_count == 1

    artifacts = [f.path for f in client.list_artifacts(callback.run.info.run_id, "")]
    local_dir = f"{tmpdir}/local_artifacts"
    os.makedirs(local_dir)

    assert "config.yaml" in artifacts
    local_config_path = client.download_artifacts(callback.run.info.run_id, "config.yaml", local_dir)

    with open(local_config_path) as f:
        config_artifact = yaml.safe_load(f)
    assert config_artifact == config

    model_path = f"runs:/{callback.run.info.run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(model_path)

    assert "ludwig" in loaded_model.metadata.flavors
    flavor = loaded_model.metadata.flavors["ludwig"]

    def compare_features(key):
        assert len(model.config[key]) == len(flavor["ludwig_schema"][key])
        for feature, schema_feature in zip(model.config[key], flavor["ludwig_schema"][key]):
            assert feature["name"] == schema_feature["name"]
            assert feature["type"] == schema_feature["type"]

    compare_features("input_features")
    compare_features("output_features")

    test_df = pd.read_csv(test_csv)
    pred_df = loaded_model.predict(test_df)
    assert pred_df.equals(expected_df)
