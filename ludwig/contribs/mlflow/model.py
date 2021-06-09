from ludwig.api import LudwigModel
from ludwig.contribs.mlflow import mlflow


class LudwigMlflowModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        super().__init__()
        self._model = None

    def load_context(self, context):
        self._model = LudwigModel.load(context.artifacts['model'])

    def predict(self, context, model_input):
        pred_df, _ = self._model.predict(model_input)
        return pred_df


def export_model(model_path, output_path, registered_model_name=None):
    kwargs = _export_kwargs(model_path)
    if registered_model_name:
        if not model_path.startswith('runs:/') or output_path is not None:
            # No run specified, so in order to register the model in mlflow, we need
            # to create a new run and upload the model as an artifact first
            output_path = output_path or 'model'
            with mlflow.start_run():
                mlflow.pyfunc.log_model(
                    artifact_path=output_path,
                    registered_model_name=registered_model_name,
                    **kwargs
                )
        else:
            # Registering a model from an artifact of an existing run
            mlflow.register_model(
                model_path,
                registered_model_name,
            )
    else:
        # No model name means we only want to save the model locally
        mlflow.pyfunc.save_model(
            path=output_path,
            **kwargs
        )


def log_model(lpath):
    mlflow.pyfunc.log_model(
        artifact_path='model',
        **_export_kwargs(lpath)
    )


def _export_kwargs(model_path):
    return dict(
        python_model=LudwigMlflowModel(),
        artifacts={
            'model': model_path,
        },
    )