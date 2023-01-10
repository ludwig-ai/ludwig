from typing import Callable, Dict, Iterable, Tuple

import lightgbm as lgb
import numpy as np
import torch
from lightgbm.callback import CallbackEnv
from numpy import typing as npt

from ludwig.constants import NUMBER
from ludwig.features.base_feature import BaseFeatureMixin, OutputFeature
from ludwig.features.category_feature import CategoryOutputFeature
from ludwig.features.feature_utils import LudwigFeatureDict
from ludwig.models.base import BaseModel


def iter_feature_metrics(features: LudwigFeatureDict) -> Iterable[Tuple[str, str]]:
    """Helper for iterating feature names and metric names."""
    for feature_name, feature in features.items():
        for metric in feature.metric_functions:
            yield feature_name, metric


def get_single_output_feature(model: BaseModel) -> BaseFeatureMixin:
    """Helper for getting the single output feature of a model."""
    return next(iter(model.output_features.values()))


def sigmoid(x: npt.NDArray) -> npt.NDArray:
    """Compute sigmoid function.

    # Inputs

    :param x: 1D array of shape (n_samples,).

    # Returns

    :return: 1D array of shape (n_samples,).
    """
    return 1.0 / (1.0 + np.exp(-x))


def log_loss_objective(y_true: npt.NDArray, y_pred: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """Binary objective function for LightGBM. Computes the logistic loss.

    # Inputs

    :param y_true: 1D array of true labels of shape (n_samples,).
    :param y_pred: 1D array of raw predictions of shape (n_samples,).

    # Returns

    :return: 1D array of gradients of shape (n_samples,) and 1D array of hessians of shape (n_samples,).

    # References

    - https://github.com/microsoft/LightGBM/issues/3312
    - https://github.com/microsoft/LightGBM/issues/5373#issuecomment-1188595889
    """
    y_pred = sigmoid(y_pred)
    grad = y_pred - y_true
    hess = y_pred * (1.0 - y_pred)
    return grad, hess


def softmax(x: npt.NDArray) -> npt.NDArray:
    """Compute softmax values for each sets of scores in x.

    # Inputs

    :param x: 2D array of shape (n_samples, n_classes).

    # Returns

    :return: 2D array of shape (n_samples, n_classes).
    """
    row_wise_max = np.max(x, axis=1).reshape(-1, 1)
    exp_x = np.exp(x - row_wise_max)
    return exp_x / np.sum(exp_x, axis=1).reshape(-1, 1)


def multiclass_objective(
    y_true: npt.NDArray, y_pred: npt.NDArray, weight: npt.NDArray = None
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Multi-class objective function for LightGBM. Computes the softmax cross-entropy loss.

    # Inputs

    :param y_true: 1D array of true labels of shape (n_samples,).
    :param y_pred: 1D array of raw predictions of shape (n_samples * n_classes,).
    :param weight: 1D array of weights of shape (n_samples,).

    # Returns

    :return: 1D array of gradients of shape (n_samples * n_classes,) and 1D array of hessians
        of shape (n_samples * n_classes,).

    # References

    - https://github.com/microsoft/LightGBM/blob/9afd8b/tests/python_package_test/test_sklearn.py#L1296
    - https://github.com/microsoft/LightGBM/blob/9afd8b/tests/python_package_test/utils.py#L142
    """
    # TODO: remove reshaping once https://github.com/microsoft/LightGBM/pull/4925 is released
    y_pred = y_pred.reshape(y_true.shape[0], -1, order="F")

    num_rows, num_class = y_pred.shape
    prob = softmax(y_pred)
    grad_update = np.zeros_like(prob)
    grad_update[np.arange(num_rows), y_true.astype(np.int32)] = -1.0
    grad = prob + grad_update
    factor = num_class / (num_class - 1)
    hess = factor * prob * (1 - prob)
    if weight is not None:
        weight2d = weight.reshape(-1, 1)
        grad *= weight2d
        hess *= weight2d

    # TODO: remove ravel once https://github.com/microsoft/LightGBM/pull/4925 is released
    grad = grad.ravel(order="F")
    hess = hess.ravel(order="F")

    return grad, hess


def store_predictions(train_logits_buffer: torch.Tensor) -> Callable:
    """Create a callback that records the predictions of the model on the training data in ``train_logits_buffer``.

    # Inputs

    :param train_logits_buffer: 2D tensor of shape (n_samples, n_classes) that stores the predictions of the model.

    # Returns

    :return: a callback that records the predictions of the model in ``train_logits_buffer``.
    """

    def _callback(env: CallbackEnv) -> None:
        # NOTE: have to copy because the buffer is reused in each iteration
        # NOTE: buffer contains raw logits because we use custom objective functions for binary/multiclass.
        preds = env.model._Booster__inner_predict(data_idx=0).copy()

        batch_size = preds.size // env.model._Booster__num_class
        preds = preds.reshape(batch_size, env.model._Booster__num_class, order="F")

        # override the predictions with the new ones
        data_view = train_logits_buffer.view(-1)
        data_view[:] = torch.from_numpy(preds).reshape(-1)

    return _callback


def reshape_logits(output_feature: OutputFeature, logits: torch.Tensor) -> torch.Tensor:
    """Add logits for the oposite class if the output feature is category with two classes.
    This is needed because LightGBM classifier only returns logits for one class.
    """
    if isinstance(output_feature, CategoryOutputFeature) and output_feature.num_classes == 2:
        # add logits for the oposite class (LightGBM classifier only returns logits for one class)
        logits = logits.view(-1, 1)
        logits = torch.cat([-logits, logits], dim=1)

    return logits


def logits_to_predictions(model: BaseModel, train_logits: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
    """Convert the logits of the model to Ludwig predictions.

    # Inputs

    :param model: the Ludwig model.
    :param train_logits: 2D tensor of shape (n_samples, n_classes) that contains the predictions of the model.

    # Returns

    :return: a dictionary mapping the output feature name to the predictions.
    """
    output_feature = get_single_output_feature(model)
    train_logits = reshape_logits(output_feature, train_logits)
    return model.outputs_to_predictions({f"{output_feature.feature_name}::logits": train_logits})


def get_targets(lgb_train: lgb.Dataset, output_feature: BaseFeatureMixin, device: str) -> Dict[str, torch.Tensor]:
    """Get the targets of the training data.

    # Inputs

    :param lgb_train: the training data.
    :param output_feature: the output feature.

    # Returns

    :return: a dictionary mapping the output feature name to the targets.
    """
    is_regression = output_feature.type() == NUMBER
    targets = lgb_train.get_label().copy() if is_regression else lgb_train.get_label().copy().astype(int)
    return {output_feature.feature_name: torch.from_numpy(targets).to(device)}
