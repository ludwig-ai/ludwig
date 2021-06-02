import os
import shutil

import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

from ludwig.api import LudwigModel
from ludwig.contribs import MlflowCallback
from ludwig.contribs.mlflow import LudwigMlflowModel
from tests.integration_tests.utils import sequence_feature, category_feature, generate_data


def test_mlflow_callback(tmpdir):
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

    mlflow_uri = f'file://{tmpdir}/mlruns'
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient(tracking_uri=mlflow_uri)

    exp_name = 'mlflow_test'
    callback = MlflowCallback()

    model = LudwigModel(config, callbacks=[callback])
    model.train(training_set=data_csv,
                validation_set=val_csv,
                test_set=test_csv,
                experiment_name=exp_name)
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

    artifacts = [f.path for f in client.list_artifacts(callback.run.info.run_id, "")]
    local_dir = f'{tmpdir}/local_artifacts'
    os.makedirs(local_dir)

    assert 'config.yaml' in artifacts
    local_config_path = client.download_artifacts(
        callback.run.info.run_id, "config.yaml", local_dir
    )

    with open(local_config_path, 'r') as f:
        config_artifact = yaml.safe_load(f)
    assert config_artifact == config

    model_path = f'runs:/{callback.run.info.run_id}/model'
    loaded_model = mlflow.pyfunc.load_model(model_path)

    test_df = pd.read_csv(test_csv)
    pred_df = loaded_model.predict(test_df)
    assert(pred_df.equals(expected_df))
