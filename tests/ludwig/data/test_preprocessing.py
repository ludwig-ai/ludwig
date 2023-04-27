import pytest

from ludwig.backend import initialize_backend
from ludwig.constants import NAME
from ludwig.data.preprocessing import handle_features_with_prompt_config, is_input_feature
from ludwig.schema.prompt import PromptConfig
from tests.integration_tests.utils import category_feature, generate_data_as_dataframe, text_feature

LOCAL_BACKEND = {"type": "local"}
RAY_BACKEND = {
    "type": "ray",
    "processor": {
        "parallelism": 1,
    },
    "trainer": {
        "use_gpu": False,
        "num_workers": 2,
        "resources_per_worker": {
            "CPU": 1,
            "GPU": 0,
        },
    },
}


def test_is_input_feature():
    # Adds encoder when output_feature=False
    assert is_input_feature(text_feature(output_feature=False)) is True
    # Adds decoder when output_feature=True
    assert is_input_feature(text_feature(output_feature=True)) is False


@pytest.mark.parametrize(
    "backend_config",
    [
        pytest.param(LOCAL_BACKEND, id="local"),
        pytest.param(RAY_BACKEND, id="ray", marks=pytest.mark.distributed),
    ],
)
@pytest.mark.parametrize(
    "retrieval_kwargs",
    [
        # {"type": "random", "k": 2},
        {
            "type": "semantic",
            "model_name": "multi-qa-MiniLM-L6-cos-v1",
            "k": 2,
        },  # TODO: find a smaller model for testing
    ],
)
def test_handle_features_with_few_shot_prompt_config(backend_config, retrieval_kwargs, ray_cluster_2cpu):
    prompt_config = PromptConfig.from_dict(
        {
            "task": (
                "Given the sample input, complete this sentence by replacing XXXX: "
                "The label is XXXX. Choose one value in this list: [1, 2, 3]."
            ),
            "retrieval": retrieval_kwargs,
        }
    ).to_dict()  # convert back-and-forth to validate and add defaults

    input_features = [
        text_feature(
            encoder={"type": "passthrough"},
            preprocessing={"prompt": prompt_config},
        )
    ]
    output_features = [
        category_feature(
            output_feature=True,
            decoder={"type": "category_parser"},
        )
    ]
    input_feature_name = input_features[0][NAME]
    output_feature_name = output_features[0][NAME]

    df = generate_data_as_dataframe(input_features, output_features, 10, with_split=True)  # retrieval needs fixed split

    backend = initialize_backend(backend_config)
    if backend_config["type"] == "ray":
        import dask.dataframe as dd

        df = dd.from_pandas(df, npartitions=2)

    split_col = df["split"]

    dataset_cols = {k: df[k] for k in df.columns}
    feature_configs = input_features + output_features
    feature_names_to_preprocessing_parameters = {
        feature_config[NAME]: feature_config.get("preprocessing", {}) for feature_config in feature_configs
    }
    handle_features_with_prompt_config(
        dataset_cols,
        feature_names_to_preprocessing_parameters,
        feature_configs,
        backend=backend,
        split_col=split_col,
    )

    # Inspect the generated prompts
    for prompt in dataset_cols[input_feature_name]:
        # input_feature_name and output_feature_name should be in the prompt because
        # labeled samples are provided by the context
        assert input_feature_name in prompt
        assert output_feature_name in prompt
