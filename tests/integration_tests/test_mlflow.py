import os
import shutil
import uuid
from unittest import mock

import mlflow
import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient

from ludwig.api import LudwigModel
from ludwig.constants import TRAINER
from ludwig.contribs import MlflowCallback
from ludwig.export import export_mlflow
from ludwig.utils.backward_compatibility import upgrade_config_dict_to_latest_version
from tests.integration_tests.utils import category_feature, FakeRemoteBackend, generate_data, sequence_feature


def run_mlflow_callback_test(mlflow_client, config, training_data, val_data, test_data, tmpdir, exp_name=None):
    ludwig_exp_name = "mlflow_test"
    callback = MlflowCallback()
    wrapped_callback = mock.Mock(wraps=callback)

    model = LudwigModel(config, callbacks=[wrapped_callback], backend=FakeRemoteBackend())
    model.train(
        training_set=training_data, validation_set=val_data, test_set=test_data, experiment_name=ludwig_exp_name
    )
    expected_df, _ = model.predict(test_data)

    # Check mlflow artifacts
    assert callback.experiment_id is not None
    assert callback.run is not None

    mlflow_exp_name = exp_name or ludwig_exp_name
    experiment = mlflow.get_experiment_by_name(mlflow_exp_name)
    assert experiment.experiment_id == callback.experiment_id

    df = mlflow.search_runs([experiment.experiment_id])
    assert len(df) == 1

    run_id = df.run_id[0]
    assert run_id == callback.run.info.run_id

    run = mlflow.get_run(run_id)
    expected_status = "FINISHED" if exp_name is None else "RUNNING"
    assert run.info.status == expected_status
    assert wrapped_callback.on_trainer_train_setup.call_count == 1
    assert wrapped_callback.on_trainer_train_teardown.call_count == 1

    artifacts = [f.path for f in mlflow_client.list_artifacts(callback.run.info.run_id, "")]
    local_dir = f"{tmpdir}/local_artifacts"
    os.makedirs(local_dir)

    assert "config.yaml" in artifacts
    local_config_path = mlflow_client.download_artifacts(callback.run.info.run_id, "config.yaml", local_dir)

    with open(local_config_path) as f:
        config_artifact = yaml.safe_load(f)
    assert config_artifact == upgrade_config_dict_to_latest_version(config)

    model_path = f"runs:/{callback.run.info.run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(model_path)

    assert "ludwig" in loaded_model.metadata.flavors
    flavor = loaded_model.metadata.flavors["ludwig"]
    config = model.config

    def compare_features(key):
        assert len(config[key]) == len(flavor["ludwig_schema"][key])
        for feature, schema_feature in zip(config[key], flavor["ludwig_schema"][key]):
            assert feature["name"] == schema_feature["name"]
            assert feature["type"] == schema_feature["type"]

    compare_features("input_features")
    compare_features("output_features")

    test_df = pd.read_csv(test_data)
    pred_df = loaded_model.predict(test_df)
    assert pred_df.equals(expected_df)
    return run


def run_mlflow_callback_test_without_artifacts(mlflow_client, config, training_data, val_data, test_data):
    exp_name = "mlflow_test_without_artifacts"
    callback = MlflowCallback(log_artifacts=False)
    wrapped_callback = mock.Mock(wraps=callback)

    model = LudwigModel(config, callbacks=[wrapped_callback], backend=FakeRemoteBackend())
    model.train(training_set=training_data, validation_set=val_data, test_set=test_data, experiment_name=exp_name)
    expected_df, _ = model.predict(test_data)

    # Check mlflow artifacts
    artifacts = [f.path for f in mlflow_client.list_artifacts(callback.run.info.run_id, "")]
    assert len(artifacts) == 0


@pytest.mark.parametrize("external_run", [False, True], ids=["internal_run", "external_run"])
def test_mlflow(tmpdir, external_run):
    epochs = 2
    batch_size = 8
    num_examples = 32

    input_features = [sequence_feature(reduce_output="sum")]
    output_features = [category_feature(vocab_size=2, reduce_input="sum", output_feature=True)]

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

    exp_name = None
    run = None
    if external_run:
        # Start a run here and make sure it's still active when training completes
        exp_name = f"ext_experiment_{uuid.uuid4().hex}"
        exp_id = mlflow.create_experiment(name=exp_name)
        run = mlflow.start_run(experiment_id=exp_id, run_name=f"ext_run_{uuid.uuid4().hex}")

    callback_run = run_mlflow_callback_test(client, config, data_csv, val_csv, test_csv, tmpdir, exp_name=exp_name)

    if not external_run:
        run_mlflow_callback_test_without_artifacts(client, config, data_csv, val_csv, test_csv)
    else:
        assert run.info.run_id == callback_run.info.run_id

        active_run = mlflow.active_run()
        assert active_run is not None
        assert run.info.run_id == active_run.info.run_id

        mlflow.end_run()


def test_export_mlflow_local(tmpdir):
    epochs = 2
    batch_size = 8
    num_examples = 32

    input_features = [sequence_feature(reduce_output="sum")]
    output_features = [category_feature(vocab_size=2, reduce_input="sum", output_feature=True)]

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {"epochs": epochs, "batch_size": batch_size},
    }

    data_csv = generate_data(
        input_features, output_features, os.path.join(tmpdir, "train.csv"), num_examples=num_examples
    )

    exp_name = "mlflow_test"
    output_dir = os.path.join(tmpdir, "output")
    model = LudwigModel(config, backend=FakeRemoteBackend())
    _, _, output_directory = model.train(training_set=data_csv, experiment_name=exp_name, output_directory=output_dir)

    model_path = os.path.join(output_directory, "model")
    output_path = os.path.join(tmpdir, "data/results/mlflow")
    export_mlflow(model_path, output_path)
    assert set(os.listdir(output_path)) == {"MLmodel", "model", "conda.yaml"}


@pytest.mark.distributed
def test_mlflow_ray(tmpdir, ray_cluster_2cpu):
    epochs = 2
    batch_size = 8
    num_examples = 32

    input_features = [sequence_feature(reduce_output="sum")]
    output_features = [category_feature(vocab_size=2, reduce_input="sum", output_feature=True)]

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {"epochs": epochs, "batch_size": batch_size},
    }

    data_csv = generate_data(
        input_features, output_features, os.path.join(tmpdir, "train.csv"), num_examples=num_examples
    )

    exp_name = "mlflow_test"
    output_dir = os.path.join(tmpdir, "output")
    model = LudwigModel(config, callbacks=[MlflowCallback()], backend="ray")
    _, _, output_directory = model.train(training_set=data_csv, experiment_name=exp_name, output_directory=output_dir)
