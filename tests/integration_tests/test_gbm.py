import os

import numpy as np
import pytest
import torch
from marshmallow import ValidationError

from ludwig.api import LudwigModel
from ludwig.constants import INPUT_FEATURES, MODEL_TYPE, OUTPUT_FEATURES, TRAINER
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
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
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
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
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


def test_hummingbird_conversion_binary(tmpdir, local_backend):
    input_features = [number_feature(), category_feature(encoder={"reduce_output": "sum"})]
    output_features = [binary_feature()]

    preds_hb, model = _train_and_predict_gbm(input_features, output_features, tmpdir, local_backend)

    _, _, test, _ = model.preprocess(os.path.join(tmpdir, "training.csv"))
    test_inputs = test.to_df(model.model.input_features.values())

    output_feature = output_features[0]

    probs_hb = preds_hb[f'{output_feature["name"]}_probabilities']
    probs_lgbm = model.model.lgbm_model.predict_proba(test_inputs)

    # sanity check Hummingbird probabilities equal to LightGBM probabilities
    assert np.allclose(np.stack(probs_hb.values), probs_lgbm, atol=1e-8)


def test_hummingbird_conversion_regression(tmpdir, local_backend):
    input_features = [number_feature(), category_feature(encoder={"reduce_output": "sum"})]
    output_features = [number_feature()]

    preds_hb, model = _train_and_predict_gbm(input_features, output_features, tmpdir, local_backend)

    _, _, test, _ = model.preprocess(os.path.join(tmpdir, "training.csv"))
    test_inputs = test.to_df(model.model.input_features.values())

    preds_lgbm = model.model.lgbm_model.predict(test_inputs)

    assert np.allclose(preds_hb.values, preds_lgbm)


@pytest.mark.parametrize("vocab_size", [2, 3])
def test_hummingbird_conversion_category(vocab_size, tmpdir, local_backend):
    input_features = [number_feature(), category_feature(encoder={"reduce_output": "sum"})]
    output_features = [category_feature(decoder={"vocab_size": vocab_size})]

    preds_hb, model = _train_and_predict_gbm(input_features, output_features, tmpdir, local_backend)

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


def test_loss_decreases(tmpdir, local_backend):
    from ludwig.datasets import adult_census_income

    config = {
        MODEL_TYPE: "gbm",
        INPUT_FEATURES: [
            {"name": "age", "type": "number"},
            {"name": "workclass", "type": "category"},
            {"name": "fnlwgt", "type": "number"},
        ],  # only use first 3 features
        OUTPUT_FEATURES: [
            {"name": "income", "type": "category"},
        ],
        TRAINER: {"num_boost_round": 2, "boosting_rounds_per_checkpoint": 1},
    }

    df = adult_census_income.load(split=False)
    # reduce dataset size to speed up test
    df = df.loc[:10, [f["name"] for f in config[INPUT_FEATURES] + config[OUTPUT_FEATURES]]]

    model = LudwigModel(config, backend=local_backend)
    train_stats, _, _ = model.train(
        dataset=df,
        output_directory=tmpdir,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True,
        skip_save_log=True,
    )

    # retrieve training losses for first and last entries.
    train_losses = train_stats["training"]["combined"]["loss"]
    last_entry = len(train_losses)

    # ensure train loss for last entry is less than or equal to the first entry.
    assert train_losses[last_entry - 1] <= train_losses[0]


def test_save_load(tmpdir, local_backend):
    input_features = [number_feature(), category_feature(encoder={"reduce_output": "sum"})]
    output_features = [binary_feature()]

    init_preds, model = _train_and_predict_gbm(input_features, output_features, tmpdir, local_backend)

    # save model
    model.save(tmpdir)

    # load model
    model = LudwigModel.load(tmpdir)
    preds, _ = model.predict(dataset=os.path.join(tmpdir, "training.csv"), split="test")

    assert init_preds.equals(preds)
