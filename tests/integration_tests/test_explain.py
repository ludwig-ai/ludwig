import logging
import os

import numpy as np
import pandas as pd
import pytest

from ludwig.api import LudwigModel
from ludwig.constants import BATCH_SIZE, BINARY, CATEGORY, MINIMUM_BATCH_SIZE, MODEL_ECD, MODEL_GBM, TYPE
from ludwig.explain.captum import IntegratedGradientsExplainer
from ludwig.explain.explainer import Explainer
from ludwig.explain.explanation import Explanation
from ludwig.explain.gbm import GBMExplainer
from tests.integration_tests.utils import (
    binary_feature,
    category_feature,
    date_feature,
    generate_data,
    image_feature,
    LocalTestBackend,
    number_feature,
    sequence_feature,
    set_feature,
    text_feature,
    timeseries_feature,
    vector_feature,
)

try:
    from ludwig.explain.captum_ray import RayIntegratedGradientsExplainer
except ImportError:
    RayIntegratedGradientsExplainer = None

pytestmark = pytest.mark.integration_tests_d


def test_explanation_dataclass():
    explanation = Explanation(target="target")

    feature_attributions_for_label_1 = np.array([1, 2, 3])
    feature_attributions_for_label_2 = np.array([4, 5, 6])

    # test add()
    explanation.add(["f1", "f2", "f3"], feature_attributions_for_label_1)

    with pytest.raises(AssertionError, match="Expected feature attributions of shape"):
        # test add() with wrong shape
        explanation.add(["f1", "f2", "f3", "f4"], np.array([1, 2, 3, 4]))

    explanation.add(["f1", "f2", "f3"], feature_attributions_for_label_2)

    # test to_array()
    explanation_array = explanation.to_array()
    assert np.array_equal(explanation_array, [[1, 2, 3], [4, 5, 6]])


def test_abstract_explainer_instantiation():
    with pytest.raises(TypeError, match="Can't instantiate abstract class Explainer with abstract method"):
        Explainer(None, inputs_df=None, sample_df=None, target=None)


@pytest.mark.parametrize(
    "explainer_class, model_type",
    [
        (IntegratedGradientsExplainer, MODEL_ECD),
        (GBMExplainer, MODEL_GBM),
    ],
)
@pytest.mark.parametrize(
    "output_feature",
    [binary_feature(), number_feature(), category_feature(decoder={"vocab_size": 3})],
    ids=["binary", "number", "category"],
)
@pytest.mark.parametrize(
    "additional_config",
    [
        pytest.param({}, id="default"),
        pytest.param({"preprocessing": {"split": {"type": "fixed", "column": "split"}}}, id="fixed_split"),
    ],
)
def test_explainer_api(explainer_class, model_type, output_feature, additional_config, tmpdir):
    run_test_explainer_api(explainer_class, model_type, [output_feature], additional_config, tmpdir)


@pytest.mark.distributed
@pytest.mark.parametrize(
    "output_feature",
    [binary_feature(), number_feature(), category_feature(decoder={"vocab_size": 3})],
    ids=["binary", "number", "category"],
)
def test_explainer_api_ray(output_feature, tmpdir, ray_cluster_2cpu):
    from ludwig.explain.captum_ray import RayIntegratedGradientsExplainer

    run_test_explainer_api(
        RayIntegratedGradientsExplainer,
        "ecd",
        [output_feature],
        {},
        tmpdir,
        resources_per_task={"num_cpus": 1},
        num_workers=1,
    )


@pytest.mark.slow
@pytest.mark.distributed
def test_explainer_api_ray_minimum_batch_size(tmpdir, ray_cluster_2cpu):
    from ludwig.explain.captum_ray import RayIntegratedGradientsExplainer

    run_test_explainer_api(
        RayIntegratedGradientsExplainer,
        "ecd",
        [binary_feature()],
        {},
        tmpdir,
        resources_per_task={"num_cpus": 1},
        num_workers=1,
        batch_size=MINIMUM_BATCH_SIZE,
    )


@pytest.mark.parametrize("cache_encoder_embeddings", [True, False])
@pytest.mark.parametrize(
    "explainer_class,model_type",
    [
        pytest.param(IntegratedGradientsExplainer, MODEL_ECD, id="ecd_local"),
        pytest.param(RayIntegratedGradientsExplainer, MODEL_ECD, id="ecd_ray", marks=pytest.mark.distributed),
        # TODO(travis): once we support GBM text features
        # pytest.param((GBMExplainer, MODEL_GBM), id="gbm_local"),
    ],
)
def test_explainer_text_hf(explainer_class, model_type, cache_encoder_embeddings, tmpdir, ray_cluster_2cpu):
    input_features = [
        text_feature(
            encoder={
                "type": "auto_transformer",
                "pretrained_model_name_or_path": "hf-internal-testing/tiny-bert-for-token-classification",
            },
            preprocessing={"cache_encoder_embeddings": cache_encoder_embeddings},
        )
    ]
    run_test_explainer_api(explainer_class, model_type, [binary_feature()], {}, tmpdir, input_features=input_features)


@pytest.mark.parametrize(
    "explainer_class,model_type",
    [
        pytest.param(IntegratedGradientsExplainer, MODEL_ECD, id="ecd_local"),
        pytest.param(RayIntegratedGradientsExplainer, MODEL_ECD, id="ecd_ray", marks=pytest.mark.distributed),
        # TODO(travis): once we support GBM text features
        # pytest.param((GBMExplainer, MODEL_GBM), id="gbm_local"),
    ],
)
def test_explainer_text_tied_weights(explainer_class, model_type, tmpdir):
    text_feature_1 = text_feature()
    text_feature_2 = text_feature(tied=text_feature_1["name"])
    input_features = [text_feature_1, text_feature_2]
    run_test_explainer_api(explainer_class, model_type, [binary_feature()], {}, tmpdir, input_features=input_features)


def run_test_explainer_api(
    explainer_class,
    model_type,
    output_features,
    additional_config,
    tmpdir,
    input_features=None,
    batch_size=128,
    **kwargs
):
    image_dest_folder = os.path.join(tmpdir, "generated_images")

    if input_features is None:
        input_features = [
            # Include a non-canonical name that's not a valid key for a vanilla pytorch ModuleDict:
            # https://github.com/pytorch/pytorch/issues/71203
            {"name": "type", "type": "binary"},
            number_feature(),
            category_feature(encoder={TYPE: "onehot", "reduce_output": "sum"}),
            category_feature(encoder={TYPE: "passthrough", "reduce_output": "sum"}),
        ]
        if model_type == MODEL_ECD:
            # TODO(travis): need unit tests to test the get_embedding_layer() of every encoder to ensure it is
            #  compatible with the explainer
            input_features += [
                category_feature(encoder={"type": "dense", "reduce_output": "sum"}),
                text_feature(encoder={"vocab_size": 3}),
                vector_feature(),
                timeseries_feature(),
                image_feature(folder=image_dest_folder),
                # audio_feature(os.path.join(tmpdir, "generated_audio")), # NOTE: works but takes a long time
                # sequence_feature(encoder={"vocab_size": 3}),
                date_feature(),
                # h3_feature(),
                set_feature(encoder={"vocab_size": 3}),
                # bag_feature(encoder={"vocab_size": 3}),
            ]

    # Generate data
    csv_filename = os.path.join(tmpdir, "training.csv")
    generate_data(input_features, output_features, csv_filename, num_examples=200)
    df = pd.read_csv(csv_filename)
    if "split" in additional_config.get("preprocessing", {}):
        df["split"] = np.random.randint(0, 3, df.shape[0])

    # Train model
    config = {"input_features": input_features, "output_features": output_features, "model_type": model_type}
    if model_type == MODEL_ECD:
        config["trainer"] = {"epochs": 2, BATCH_SIZE: batch_size}
    else:
        # Disable feature filtering to avoid having no features due to small test dataset,
        # see https://stackoverflow.com/a/66405983/5222402
        config["trainer"] = {"feature_pre_filter": False}
    config.update(additional_config)

    model = LudwigModel(config, logging_level=logging.WARNING, backend=LocalTestBackend())
    model.train(df)

    # Explain model
    explainer = explainer_class(model, inputs_df=df, sample_df=df, target=output_features[0]["name"], **kwargs)

    is_binary = output_features[0].get("type") == BINARY
    is_category = output_features[0].get("type") == CATEGORY

    vocab_size = 1
    if is_binary:
        vocab_size = 2
    elif is_category:
        vocab_size = output_features[0].get("decoder", {}).get("vocab_size")

    assert explainer.is_binary_target == is_binary
    assert explainer.is_category_target == is_category
    assert explainer.vocab_size == vocab_size

    explanations_result = explainer.explain()

    # Verify shapes.
    assert explanations_result.global_explanation.to_array().shape == (vocab_size, len(input_features))

    assert len(explanations_result.row_explanations) == len(df)
    for e in explanations_result.row_explanations:
        assert e.to_array().shape == (vocab_size, len(input_features))

    assert len(explanations_result.expected_values) == vocab_size


@pytest.mark.parametrize(
    "output_feature",
    [set_feature(decoder={"vocab_size": 3}), vector_feature()],
    ids=["set", "vector"],
)
def test_explainer_api_nonscalar_outputs(output_feature, tmpdir):
    run_test_explainer_api(IntegratedGradientsExplainer, MODEL_ECD, [output_feature], {}, tmpdir)


def test_explainer_api_text_outputs(tmpdir):
    input_features = [text_feature(encoder={"type": "parallel_cnn", "reduce_output": None})]
    output_features = [text_feature(output_feature=True, decoder={"type": "tagger"})]
    run_test_explainer_api(
        IntegratedGradientsExplainer, MODEL_ECD, output_features, {}, tmpdir, input_features=input_features
    )


@pytest.mark.parametrize(
    "explainer_class,model_type",
    [
        pytest.param(IntegratedGradientsExplainer, MODEL_ECD, id="ecd_local"),
        pytest.param(RayIntegratedGradientsExplainer, MODEL_ECD, id="ecd_ray", marks=pytest.mark.distributed),
    ],
)
@pytest.mark.parametrize(
    "encoder_type", ["embed", "parallel_cnn", "stacked_cnn", "stacked_parallel_cnn", "rnn", "cnnrnn", "transformer"]
)
def test_explainer_sequence_feature(explainer_class, model_type, encoder_type, tmpdir):
    input_features = [sequence_feature()]
    input_features[0]["encoder"] = {"type": encoder_type}
    output_features = [binary_feature()]
    run_test_explainer_api(explainer_class, model_type, output_features, {}, tmpdir, input_features=input_features)
