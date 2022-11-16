import logging
import os
import shutil

import mlflow
import yaml
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import _save_example, ModelInputExample
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration

from ludwig.api_annotations import DeveloperAPI
from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME
from ludwig.utils.data_utils import load_json

FLAVOR_NAME = "ludwig"

_logger = logging.getLogger(__name__)


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    import ludwig

    # Ludwig is not yet available via the default conda channels, so we install it via pip
    return _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[f"ludwig=={ludwig.__version__}"],
        additional_conda_channels=None,
    )


def save_model(
    ludwig_model,
    path,
    conda_env=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
):
    """Save a Ludwig model to a path on the local file system.

    :param ludwig_model: Ludwig model (an instance of `ludwig.api.LudwigModel`_) to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this describes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If ``None``, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'pip': [
                                    'ludwig==0.4.0'
                                ]
                            ]
                        }

    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature

                        train = df.drop_column("target_label")
                        predictions = ...  # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    """
    import ludwig

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException(f"Path '{path}' already exists")
    model_data_subpath = "model"
    model_data_path = os.path.join(path, model_data_subpath)
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    # Save the Ludwig model
    ludwig_model.save(model_data_path)

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env) as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="ludwig.contribs.mlflow.model",
        data=model_data_subpath,
        env=conda_env_subpath,
    )

    schema_keys = {"name", "column", "type"}
    config = ludwig_model.config

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        ludwig_version=ludwig.__version__,
        ludwig_schema={
            "input_features": [
                {k: v for k, v in feature.items() if k in schema_keys} for feature in config["input_features"]
            ],
            "output_features": [
                {k: v for k, v in feature.items() if k in schema_keys} for feature in config["output_features"]
            ],
        },
        data=model_data_subpath,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def log_model(
    ludwig_model,
    artifact_path,
    conda_env=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
):
    """Log a Ludwig model as an MLflow artifact for the current run.

    :param ludwig_model: Ludwig model (an instance of `ludwig.api.LudwigModel`_) to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this describes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If ``None``, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'pip': [
                                    'ludwig==0.4.0'
                                ]
                            ]
                        }
    :param registered_model_name: (Experimental) If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature

                        train = df.drop_column("target_label")
                        predictions = ...  # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.
    """
    import ludwig

    Model.log(
        artifact_path=artifact_path,
        flavor=ludwig.contribs.mlflow.model,
        registered_model_name=registered_model_name,
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        ludwig_model=ludwig_model,
    )


def _load_model(path):
    from ludwig.api import LudwigModel

    return LudwigModel.load(path, backend="local")


def _load_pyfunc(path):
    """Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``ludwig`` flavor.
    """
    return _LudwigModelWrapper(_load_model(path))


def load_model(model_uri):
    """Load a Ludwig model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
                      artifact-locations>`_.

    :return: A Ludwig model (an instance of `ludwig.api.LudwigModel`_).
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    lgb_model_file_path = os.path.join(local_model_path, flavor_conf.get("data", "model.lgb"))
    return _load_model(path=lgb_model_file_path)


class _LudwigModelWrapper:
    def __init__(self, ludwig_model):
        self.ludwig_model = ludwig_model

    def predict(self, dataframe):
        pred_df, _ = self.ludwig_model.predict(dataframe)
        return pred_df


def export_model(model_path, output_path, registered_model_name=None):
    if registered_model_name:
        if not model_path.startswith("runs:/") or output_path is not None:
            # No run specified, so in order to register the model in mlflow, we need
            # to create a new run and upload the model as an artifact first
            output_path = output_path or "model"
            log_model(
                _CopyModel(model_path),
                artifact_path=output_path,
                registered_model_name=registered_model_name,
            )
        else:
            # Registering a model from an artifact of an existing run
            mlflow.register_model(
                model_path,
                registered_model_name,
            )
    else:
        # No model name means we only want to save the model locally
        save_model(
            _CopyModel(model_path),
            path=output_path,
        )


@DeveloperAPI
def log_saved_model(lpath):
    """Log a saved Ludwig model as an MLflow artifact.

    :param lpath: Path to saved Ludwig model.
    """
    log_model(
        _CopyModel(lpath),
        artifact_path="model",
    )


class _CopyModel:
    """Get model data without requiring us to read the model weights into memory."""

    def __init__(self, lpath):
        self.lpath = lpath

    def save(self, path):
        shutil.copytree(self.lpath, path)

    @property
    def config(self):
        return load_json(os.path.join(self.lpath, MODEL_HYPERPARAMETERS_FILE_NAME))
