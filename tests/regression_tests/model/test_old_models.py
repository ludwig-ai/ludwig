import os

import boto3
import pandas as pd

from ludwig.api import LudwigModel


def download_directory_from_s3(bucket_name: str, remote_directory_name: str, tmpdir: str):
    """Downloads a directory from S3 to the same relative path, under tmpdir.

    Based on https://stackoverflow.com/questions/49772151/download-a-folder-from-s3-using-boto3.
    """
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=remote_directory_name):
        local_dir = os.path.join(tmpdir, os.path.dirname(obj.key))
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        bucket.download_file(obj.key, os.path.join(local_dir, os.path.basename(obj.key)))


def test_model_loaded_from_old_config_prediction_works(tmpdir):
    # Titanic model based on 0.5.3.
    # Model config:
    # https://predibase-public-us-west-2.s3.us-west-2.amazonaws.com/ludwig_unit_tests/simple_experiment_simple_model/
    # model/model_hyperparameters.json
    download_directory_from_s3(
        "predibase-public-us-west-2", "ludwig_unit_tests/simple_experiment_simple_model/model/", tmpdir
    )
    ludwig_model = LudwigModel.load(os.path.join(tmpdir, "ludwig_unit_tests/simple_experiment_simple_model/model"))
    example_data = {
        "PassengerId": 892,
        "Pclass": 3,
        "Name": "Kelly, Mr. James",
        "Sex": "male",
        "Age": 34.5,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "330911",
        "Fare": 7.8292,
        "Cabin": None,
        "Embarked": "Q",
    }
    test_set = pd.DataFrame(example_data, index=[0])

    predictions, _ = ludwig_model.predict(dataset=test_set)

    assert predictions.to_dict()["Survived_predictions"] == {0: False}
