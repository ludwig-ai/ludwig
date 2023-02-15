import os
import re

import numpy as np
import pytest

try:
    import ray as _ray
except ImportError:
    _ray = None

from ludwig.api import LudwigModel
from ludwig.constants import COLUMN, INPUT_FEATURES, MODEL_TYPE, NAME, OUTPUT_FEATURES, TRAINER
from ludwig.error import ConfigValidationError
from tests.integration_tests import synthetic_test_data
from tests.integration_tests.utils import binary_feature
from tests.integration_tests.utils import category_feature as _category_feature
from tests.integration_tests.utils import generate_data, number_feature

BOOSTING_TYPES = ["gbdt", "goss", "dart"]
TREE_LEARNERS = ["serial", "feature", "data", "voting"]


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
            "parallelism": 1,
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


def category_feature(**kwargs):
    encoder = kwargs.get("encoder", {})
    encoder = {**{"type": "passthrough"}, **encoder}
    kwargs["encoder"] = encoder
    return _category_feature(**kwargs)


def _train_and_predict_gbm(input_features, output_features, tmpdir, backend_config, **trainer_config):
    csv_filename = os.path.join(tmpdir, "training.csv")
    dataset_filename = generate_data(input_features, output_features, csv_filename, num_examples=100)

    config = {
        MODEL_TYPE: "gbm",
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        # Disable feature filtering to avoid having no features due to small test dataset,
        # see https://stackoverflow.com/a/66405983/5222402
        TRAINER: {"num_boost_round": 2, "feature_pre_filter": False},
    }

    if trainer_config:
        config[TRAINER].update(trainer_config)

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
def test_ray_gbm_binary(tmpdir, ray_backend, ray_cluster_5cpu):
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
def test_ray_gbm_non_number_inputs(tmpdir, ray_backend, ray_cluster_5cpu):
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
def test_ray_gbm_category(vocab_size, tmpdir, ray_backend, ray_cluster_5cpu):
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
def test_ray_gbm_number(tmpdir, ray_backend, ray_cluster_5cpu):
    run_test_gbm_number(tmpdir, ray_backend)


def test_hummingbird_conversion_binary(tmpdir, local_backend):
    """Verify that Hummingbird conversion predictions match LightGBM predictions for binary outputs."""
    input_features = [number_feature(), category_feature(encoder={"reduce_output": "sum"})]
    output_features = [binary_feature()]
    output_feature = f'{output_features[0]["name"]}_probabilities'

    # Train a model and predict using the LightGBM interface
    preds_lgbm, model = _train_and_predict_gbm(input_features, output_features, tmpdir, local_backend)
    probs_lgbm = preds_lgbm[output_feature]

    # Predict using the Hummingbird compiled model
    with model.model.compile():
        preds_hb, _ = model.predict(dataset=os.path.join(tmpdir, "training.csv"), split="test")
        probs_hb = preds_hb[output_feature]

    # sanity check Hummingbird probabilities equal to LightGBM probabilities
    assert np.allclose(np.stack(probs_hb.values), np.stack(probs_lgbm.values), rtol=1e-6, atol=1e-6)


def test_hummingbird_conversion_regression(tmpdir, local_backend):
    """Verify that Hummingbird conversion predictions match LightGBM predictions for numeric outputs."""
    input_features = [number_feature(), category_feature(encoder={"reduce_output": "sum"})]
    output_features = [number_feature()]

    # Train a model and predict using the LightGBM interface
    preds_lgbm, model = _train_and_predict_gbm(input_features, output_features, tmpdir, local_backend)

    # Predict using the Hummingbird compiled model
    with model.model.compile():
        preds_hb, _ = model.predict(dataset=os.path.join(tmpdir, "training.csv"), split="test")

    # sanity check Hummingbird prediction equal to LightGBM prediction
    assert np.allclose(preds_hb, preds_lgbm, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("vocab_size", [2, 3])
def test_hummingbird_conversion_category(vocab_size, tmpdir, local_backend):
    """Verify that Hummingbird conversion predictions match LightGBM predictions for categorical outputs."""
    input_features = [number_feature(), category_feature(encoder={"reduce_output": "sum"})]
    output_features = [category_feature(decoder={"vocab_size": vocab_size})]

    # Train a model and predict using the LightGBM interface
    preds_lgbm, model = _train_and_predict_gbm(input_features, output_features, tmpdir, local_backend)
    output_feature = next(iter(model.model.output_features.values()))
    output_feature_name = f"{output_feature.column}_probabilities"
    probs_lgbm = np.stack(preds_lgbm[output_feature_name].to_numpy())

    # Predict using the Hummingbird compiled model
    with model.model.compile():
        preds_hb, _ = model.predict(dataset=os.path.join(tmpdir, "training.csv"), split="test")
        probs_hb = np.stack(preds_hb[output_feature_name].to_numpy())

    # sanity check Hummingbird probabilities equal to LightGBM probabilities
    assert np.allclose(probs_hb, probs_lgbm, rtol=1e-6, atol=1e-6)


def test_loss_decreases(tmpdir, local_backend):
    input_features, output_features = synthetic_test_data.get_feature_configs()

    config = {
        MODEL_TYPE: "gbm",
        "input_features": input_features,
        "output_features": output_features,
        # Disable feature filtering to avoid having no features due to small test dataset,
        # see https://stackoverflow.com/a/66405983/5222402
        TRAINER: {
            "num_boost_round": 2,
            "boosting_rounds_per_checkpoint": 1,
            "feature_pre_filter": False,
        },
    }

    generated_data = synthetic_test_data.get_generated_data_for_optimizer()
    model = LudwigModel(config, backend=local_backend)
    train_stats, _, _ = model.train(
        dataset=generated_data.train_df,
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


def test_lgbm_dataset_setup(tmpdir, local_backend):
    """Test that LGBM dataset column name errors are handled."""
    input_features = [number_feature()]
    output_features = [binary_feature()]

    # Overwrite the automatically generated feature/column name with an invalid string.
    input_features[0][NAME] = ",Unnamed: 0"
    input_features[0][COLUMN] = input_features[0][NAME]

    # Test that the custom error is raised (lightgbm.basic.LightGBMError -> ValueError)
    with pytest.raises(ValueError):
        try:
            _train_and_predict_gbm(input_features, output_features, tmpdir, local_backend)
        except ValueError as e:
            # Check that the intended ValueError was raised
            assert re.search("Some column names in the training set contain invalid characters", str(e))
            raise


def test_boosting_type_rf_invalid(tmpdir, local_backend):
    """Test that the Random Forest boosting type is not supported.

    LightGBM does not support model checkpointing for `boosting_type=rf`. This test ensures that a schema validation
    error is raised when trying to use random forests.
    """
    input_features = [number_feature()]
    output_features = [binary_feature()]

    with pytest.raises(ConfigValidationError):
        _train_and_predict_gbm(input_features, output_features, tmpdir, local_backend, boosting_type="rf")


@pytest.mark.skip(reason="LightGBMError: Number of class for initial score error")
def test_goss_deactivate_bagging(tmpdir, local_backend):
    """Test that bagging is disabled for the GOSS boosting type.

    TODO: Re-enable when GOSS is supported: https://github.com/ludwig-ai/ludwig/issues/2988
    """
    input_features = [number_feature()]
    output_features = [binary_feature()]

    _train_and_predict_gbm(input_features, output_features, tmpdir, local_backend, boosting_type="goss", bagging_freq=5)


@pytest.mark.parametrize("tree_learner", TREE_LEARNERS)
def test_boosting_type_null_invalid(tree_learner, tmpdir, local_backend):
    """Test that the null boosting type is disabled.

    `boosting_type: null` defaults to "gbdt", and it was removed to avoid confusing GBM trainer settings.
    """
    input_features = [number_feature()]
    output_features = [binary_feature()]

    with pytest.raises(ConfigValidationError):
        _train_and_predict_gbm(
            input_features, output_features, tmpdir, local_backend, boosting_type=None, tree_learner=tree_learner
        )


@pytest.mark.parametrize("boosting_type", BOOSTING_TYPES)
def test_tree_learner_null_invalid(boosting_type, tmpdir, local_backend):
    """Test that the null tree learner is disabled.

    `tree_learner: null` defaults to "serial", and it was removed to avoid confusing GBM trainer settings.
    """
    input_features = [number_feature()]
    output_features = [binary_feature()]

    with pytest.raises(ConfigValidationError):
        _train_and_predict_gbm(
            input_features, output_features, tmpdir, local_backend, boosting_type=boosting_type, tree_learner=None
        )


def test_dart_boosting_type(tmpdir, local_backend):
    """Test that DART does not error during eval due to progress tracking."""
    input_features = [number_feature()]
    output_features = [binary_feature()]

    _train_and_predict_gbm(input_features, output_features, tmpdir, local_backend, boosting_type="dart")
