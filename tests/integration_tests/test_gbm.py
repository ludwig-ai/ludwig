import os

import numpy as np
import pytest
import torch
from marshmallow import ValidationError

from ludwig.api import LudwigModel
from ludwig.constants import MODEL_TYPE, TRAINER
from tests.integration_tests.utils import binary_feature, category_feature, generate_data, number_feature, text_feature


@pytest.fixture(scope="module")
def local_backend():
    return {"type": "local"}


@pytest.fixture(scope="module")
def ray_backend():
    num_workers = 2
    num_cpus_per_worker = 2
    return {
        "type": "ray",
        "processor": {
            "parallelism": num_cpus_per_worker * num_workers,
        },
        "trainer": {
            "use_gpu": False,
            "num_workers": num_workers,
            "resources_per_worker": {
                "CPU": num_cpus_per_worker,
                "GPU": 0,
            },
        },
    }


def _train_and_predict_gbm(input_features, output_features, tmpdir, backend_config):
    csv_filename = os.path.join(tmpdir, "training.csv")
    dataset_filename = generate_data(input_features, output_features, csv_filename, num_examples=100)

    config = {
        MODEL_TYPE: "gbm",
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"num_boost_round": 2},
    }

    model = LudwigModel(config, backend=backend_config)
    _, _, output_directory = model.train(
        dataset=dataset_filename,
        output_directory=tmpdir,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True,
        skip_save_log=True,
    )
    model.load(os.path.join(tmpdir, "api_experiment_run", "model"))
    preds, _ = model.predict(dataset=dataset_filename, output_directory=output_directory, split="test")

    return preds, model


def run_test_gbm_output_not_supported(tmpdir, backend_config):
    """Test that an error is raised when the output feature is not supported by the model."""
    input_features = [number_feature(), category_feature(encoder={"reduce_output": "sum"})]
    output_features = [text_feature()]

    with pytest.raises(ValueError, match="Model type GBM only supports numerical, categorical, or binary features"):
        _train_and_predict_gbm(input_features, output_features, tmpdir, backend_config)


def test_local_gbm_output_not_supported(tmpdir, local_backend):
    run_test_gbm_output_not_supported(tmpdir, local_backend)


@pytest.mark.distributed
def test_ray_gbm_output_not_supported(tmpdir, ray_backend, ray_cluster_4cpu):
    run_test_gbm_output_not_supported(tmpdir, ray_backend)


def run_test_gbm_multiple_outputs(tmpdir, backend_config):
    """Test that an error is raised when the model is trained with multiple outputs."""
    input_features = [number_feature(), category_feature(encoder={"reduce_output": "sum"})]
    output_features = [
        category_feature(decoder={"vocab_size": 3}),
        binary_feature(),
        category_feature(decoder={"vocab_size": 3}),
    ]

    with pytest.raises(ValueError, match="Only single task currently supported"):
        _train_and_predict_gbm(input_features, output_features, tmpdir, backend_config)


def test_local_gbm_multiple_outputs(tmpdir, local_backend):
    run_test_gbm_multiple_outputs(tmpdir, local_backend)


@pytest.mark.distributed
def test_ray_gbm_multiple_outputs(tmpdir, ray_backend, ray_cluster_4cpu):
    run_test_gbm_multiple_outputs(tmpdir, ray_backend)


def run_test_gbm_binary(tmpdir, backend_config):
    """Test that the GBM model can train and predict a binary variable (binary classification)."""
    input_features = [number_feature(), category_feature(encoder={"reduce_output": "sum"})]
    output_feature = binary_feature()
    output_features = [output_feature]

    preds, _ = _train_and_predict_gbm(input_features, output_features, tmpdir, backend_config)

    prob_col = preds[output_feature["name"] + "_probabilities"]
    if backend_config["type"] == "ray":
        prob_col = prob_col.compute()
    assert len(prob_col.iloc[0]) == 2
    assert prob_col.apply(sum).mean() == pytest.approx(1.0)


def test_local_gbm_binary(tmpdir, local_backend):
    run_test_gbm_binary(tmpdir, local_backend)


@pytest.mark.distributed
def test_ray_gbm_binary(tmpdir, ray_backend, ray_cluster_4cpu):
    run_test_gbm_binary(tmpdir, ray_backend)


def run_test_gbm_non_number_inputs(tmpdir, backend_config):
    """Test that the GBM model can train and predict with non-number inputs."""
    input_features = [binary_feature(), category_feature(encoder={"reduce_output": "sum"})]
    output_feature = binary_feature()
    output_features = [output_feature]

    preds, _ = _train_and_predict_gbm(input_features, output_features, tmpdir, backend_config)

    prob_col = preds[output_feature["name"] + "_probabilities"]
    if backend_config["type"] == "ray":
        prob_col = prob_col.compute()
    assert len(prob_col.iloc[0]) == 2
    assert prob_col.apply(sum).mean() == pytest.approx(1.0)


def test_local_gbm_non_number_inputs(tmpdir, local_backend):
    run_test_gbm_non_number_inputs(tmpdir, local_backend)


@pytest.mark.distributed
def test_ray_gbm_non_number_inputs(tmpdir, ray_backend, ray_cluster_4cpu):
    run_test_gbm_non_number_inputs(tmpdir, ray_backend)


def run_test_hummingbird_conversion_binary(tmpdir, backend_config):
    input_features = [number_feature(), category_feature(encoder={"reduce_output": "sum"})]
    output_features = [binary_feature()]

    preds_hb, model = _train_and_predict_gbm(input_features, output_features, tmpdir, backend_config)

    _, _, test, _ = model.preprocess(os.path.join(tmpdir, "training.csv"))
    test_inputs = test.to_df(model.model.input_features.values())

    output_feature = output_features[0]

    probs_hb = preds_hb[f'{output_feature["name"]}_probabilities']
    probs_lgbm = model.model.lgbm_model.predict_proba(test_inputs)

    # sanity check Hummingbird probabilities equal to LightGBM probabilities
    assert np.allclose(np.stack(probs_hb.values), probs_lgbm, atol=1e-8)


def run_test_hummingbird_conversion_regression(tmpdir, backend_config):
    input_features = [number_feature(), category_feature(encoder={"reduce_output": "sum"})]
    output_features = [number_feature()]

    preds_hb, model = _train_and_predict_gbm(input_features, output_features, tmpdir, backend_config)

    _, _, test, _ = model.preprocess(os.path.join(tmpdir, "training.csv"))
    test_inputs = test.to_df(model.model.input_features.values())

    preds_lgbm = model.model.lgbm_model.predict(test_inputs)

    assert np.allclose(preds_hb.values, preds_lgbm)


def run_test_hummingbird_conversion_category(vocab_size, tmpdir, backend_config):
    input_features = [number_feature(), category_feature(encoder={"reduce_output": "sum"})]
    output_features = [category_feature(decoder={"vocab_size": vocab_size})]

    preds_hb, model = _train_and_predict_gbm(input_features, output_features, tmpdir, backend_config)

    _, _, test, _ = model.preprocess(os.path.join(tmpdir, "training.csv"))
    test_inputs = test.to_df(model.model.input_features.values())

    output_feature = next(iter(model.model.output_features.values()))

    probs_hb = preds_hb[f"{output_feature.column}_probabilities"]

    if output_feature.num_classes == 2:
        # LightGBM binary classifier transforms logits using sigmoid, so we need to invert the
        # transformation to get the logits back, and then use softmax to get the probabilities to match
        # Ludwig's category feature prediction behavior.
        probs_lgbm = torch.softmax(torch.logit(torch.from_numpy(model.model.lgbm_model.predict_proba(test_inputs))), -1)
    else:
        # Otherwise, just compare to the raw probabilities from LightGBM
        probs_lgbm = model.model.lgbm_model.predict_proba(test_inputs)

    # sanity check Hummingbird probabilities equal to LightGBM probabilities
    assert np.allclose(np.stack(probs_hb.values), probs_lgbm, atol=1e-4)


def test_local_hummingbird_conversion_binary(tmpdir, local_backend):
    run_test_hummingbird_conversion_binary(tmpdir, local_backend)


def test_local_hummingbird_conversion_regression(tmpdir, local_backend):
    run_test_hummingbird_conversion_regression(tmpdir, local_backend)


@pytest.mark.parametrize("vocab_size", [2, 3])
def test_local_hummingbird_conversion_category(vocab_size, tmpdir, local_backend):
    run_test_hummingbird_conversion_category(vocab_size, tmpdir, local_backend)


# TODO: tests for loss going down

# TODO: saving, loading, continuing training


def run_test_gbm_category(vocab_size, tmpdir, backend_config):
    """Test that the GBM model can train and predict a categorical output (multiclass classification)."""
    input_features = [number_feature(), category_feature(encoder={"reduce_output": "sum"})]
    output_feature = category_feature(decoder={"vocab_size": vocab_size})
    output_features = [output_feature]

    preds, _ = _train_and_predict_gbm(input_features, output_features, tmpdir, backend_config)

    prob_col = preds[output_feature["name"] + "_probabilities"]
    if backend_config["type"] == "ray":
        prob_col = prob_col.compute()
    assert len(prob_col.iloc[0]) == vocab_size
    assert prob_col.apply(sum).mean() == pytest.approx(1.0)


@pytest.mark.parametrize("vocab_size", [2, 3])
def test_local_gbm_category(vocab_size, tmpdir, local_backend):
    run_test_gbm_category(vocab_size, tmpdir, local_backend)


@pytest.mark.distributed
@pytest.mark.parametrize("vocab_size", [2, 3])
def test_ray_gbm_category(vocab_size, tmpdir, ray_backend, ray_cluster_4cpu):
    run_test_gbm_category(vocab_size, tmpdir, ray_backend)


def run_test_gbm_number(tmpdir, backend_config):
    """Test that the GBM model can train and predict a numerical output (regression)."""
    # Given a dataset with a single input feature and a single output feature,
    input_features = [number_feature(), category_feature(encoder={"reduce_output": "sum"})]
    output_feature = number_feature()
    output_features = [output_feature]

    # When we train a GBM model on the dataset,
    preds, _ = _train_and_predict_gbm(input_features, output_features, tmpdir, backend_config)

    # Then the predictions should be included in the output
    pred_col = preds[output_feature["name"] + "_predictions"]
    if backend_config["type"] == "ray":
        pred_col = pred_col.compute()
    assert pred_col.dtype == float


def test_local_gbm_number(tmpdir, local_backend):
    run_test_gbm_number(tmpdir, local_backend)


@pytest.mark.distributed
def test_ray_gbm_number(tmpdir, ray_backend, ray_cluster_4cpu):
    run_test_gbm_number(tmpdir, ray_backend)


def run_test_gbm_schema(backend_config):
    input_features = [number_feature()]
    output_features = [binary_feature()]

    # When I pass an invalid trainer configuration,
    invalid_trainer = "trainer"
    config = {
        MODEL_TYPE: "gbm",
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {
            "num_boost_round": 2,
            "type": invalid_trainer,
        },
    }
    with pytest.raises(ValidationError):
        # Then I should get a schema validation error
        LudwigModel(config, backend=backend_config)


def test_local_gbm_schema(local_backend):
    run_test_gbm_schema(local_backend)


@pytest.mark.distributed
def test_ray_gbm_schema(ray_backend, ray_cluster_4cpu):
    run_test_gbm_schema(ray_backend)
