from __future__ import annotations

import copy
import json
import os
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import ludwig.error as ludwig_error
from ludwig.api import LudwigModel
from ludwig.constants import (
    ADAPTER,
    BACKEND,
    BASE_MODEL,
    BATCH_SIZE,
    COMBINER,
    EPOCHS,
    EVAL_BATCH_SIZE,
    GENERATION,
    INPUT_FEATURES,
    MERGE_ADAPTER_INTO_BASE_MODEL,
    MODEL_ECD,
    MODEL_LLM,
    MODEL_TYPE,
    OUTPUT_FEATURES,
    POSTPROCESSOR,
    PREPROCESSING,
    PRETRAINED_ADAPTER_WEIGHTS,
    PROGRESSBAR,
    PROMPT,
    QUANTIZATION,
    TARGET_MODULES,
    TRAINER,
    TYPE,
)
from ludwig.globals import MODEL_FILE_NAME, MODEL_WEIGHTS_FILE_NAME
from ludwig.models.llm import LLM
from ludwig.schema.model_types.base import ModelConfig
from ludwig.utils.fs_utils import list_file_names_in_directory
from ludwig.utils.types import DataFrame
from tests.integration_tests.utils import category_feature, generate_data, text_feature

pytestmark = pytest.mark.llm


LOCAL_BACKEND = {"type": "local"}
TEST_MODEL_NAME = "hf-internal-testing/tiny-random-GPTJForCausalLM"
MAX_NEW_TOKENS_TEST_DEFAULT = 5

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


def get_num_non_empty_tokens(iterable):
    """Returns the number of non-empty tokens."""
    return len(list(filter(bool, iterable)))


@pytest.fixture(scope="module")
def local_backend():
    return LOCAL_BACKEND


@pytest.fixture(scope="module")
def ray_backend():
    return RAY_BACKEND


def get_dataset():
    data = [
        {"review": "I loved this movie!", "output": "positive"},
        {"review": "The food was okay, but the service was terrible.", "output": "negative"},
        {"review": "I can't believe how rude the staff was.", "output": "negative"},
        {"review": "This book was a real page-turner.", "output": "positive"},
        {"review": "The hotel room was dirty and smelled bad.", "output": "negative"},
        {"review": "I had a great experience at this restaurant.", "output": "positive"},
        {"review": "The concert was amazing!", "output": "positive"},
        {"review": "The traffic was terrible on my way to work this morning.", "output": "negative"},
        {"review": "The customer service was excellent.", "output": "positive"},
        {"review": "I was disappointed with the quality of the product.", "output": "negative"},
    ]
    df = pd.DataFrame(data)
    return df


def get_generation_config():
    return {
        "temperature": 0.1,
        "top_p": 0.75,
        "top_k": 40,
        "num_beams": 4,
        "max_new_tokens": MAX_NEW_TOKENS_TEST_DEFAULT,
    }


def convert_preds(preds: DataFrame):
    if isinstance(preds, pd.DataFrame):
        return preds.to_dict()
    return preds.compute().to_dict()


@pytest.mark.llm
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(LOCAL_BACKEND, id="local"),
        pytest.param(RAY_BACKEND, id="ray"),
    ],
)
def test_llm_text_to_text(tmpdir, backend, ray_cluster_4cpu):
    """Test that the LLM model can train and predict with text inputs and text outputs."""
    input_features = [
        {
            "name": "Question",
            "type": "text",
            "encoder": {"type": "passthrough"},
        }
    ]
    output_features = [text_feature(output_feature=True, name="Answer", decoder={"type": "text_extractor"})]

    csv_filename = os.path.join(tmpdir, "training.csv")
    dataset_filename = generate_data(input_features, output_features, csv_filename, num_examples=100)

    config = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: TEST_MODEL_NAME,
        GENERATION: get_generation_config(),
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        BACKEND: backend,
    }

    model = LudwigModel(config)
    model.train(dataset=dataset_filename, output_directory=str(tmpdir), skip_save_processed_input=True)

    preds, _ = model.predict(dataset=dataset_filename, output_directory=str(tmpdir), split="test")
    preds = convert_preds(preds)

    assert "Answer_predictions" in preds
    assert "Answer_probabilities" in preds
    assert "Answer_probability" in preds
    assert "Answer_response" in preds

    assert preds["Answer_predictions"]
    assert preds["Answer_probabilities"]
    assert preds["Answer_probability"]
    assert preds["Answer_response"]

    # Check that in-line generation parameters are used. Original prediction uses max_new_tokens = 5.
    assert get_num_non_empty_tokens(preds["Answer_predictions"][0]) <= MAX_NEW_TOKENS_TEST_DEFAULT
    original_max_new_tokens = model.model.generation.max_new_tokens

    # This prediction uses max_new_tokens = 2.
    preds, _ = model.predict(
        dataset=dataset_filename,
        output_directory=str(tmpdir),
        split="test",
        generation_config={"min_new_tokens": 2, "max_new_tokens": 3},
    )
    preds = convert_preds(preds)
    print(preds["Answer_predictions"][0])
    num_non_empty_tokens = get_num_non_empty_tokens(preds["Answer_predictions"][0])
    assert 2 <= num_non_empty_tokens <= 3

    # Check that the state of the model is unchanged.
    assert model.model.generation.max_new_tokens == original_max_new_tokens


@pytest.mark.llm
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(LOCAL_BACKEND, id="local"),
        pytest.param(RAY_BACKEND, id="ray"),
    ],
)
def test_llm_zero_shot_classification(tmpdir, backend, ray_cluster_4cpu):
    input_features = [
        {
            "name": "review",
            "type": "text",
        }
    ]
    output_features = [
        category_feature(
            name="output",
            preprocessing={
                "fallback_label": "neutral",
            },
            # How can we avoid using r here for regex, since it is technically an implementation detail?
            decoder={
                "type": "category_extractor",
                "match": {
                    "positive": {"type": "contains", "value": "positive"},
                    "neutral": {"type": "regex", "value": r"\bneutral\b"},
                    "negative": {"type": "contains", "value": "negative"},
                },
            },
        )
    ]

    df = get_dataset()

    config = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: TEST_MODEL_NAME,
        GENERATION: get_generation_config(),
        PROMPT: {"task": "This is a review of a restaurant. Classify the sentiment."},
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        BACKEND: backend,
    }

    model = LudwigModel(config)
    model.train(dataset=df, output_directory=str(tmpdir), skip_save_processed_input=True)

    prediction_df = pd.DataFrame(
        [
            {"review": "The food was amazing!", "output": "positive"},
            {"review": "The service was terrible.", "output": "negative"},
            {"review": "The food was okay.", "output": "neutral"},
        ]
    )

    preds, _ = model.predict(dataset=prediction_df, output_directory=str(tmpdir))
    preds = convert_preds(preds)

    assert preds


@pytest.mark.llm
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(LOCAL_BACKEND, id="local"),
        pytest.param(RAY_BACKEND, id="ray"),
    ],
)
def test_llm_few_shot_classification(tmpdir, backend, csv_filename, ray_cluster_4cpu):
    input_features = [
        text_feature(
            output_feature=False,
            name="body",
            encoder={"type": "passthrough"},  # need to use the default encoder for LLMTextInputFeatureConfig
        )
    ]
    output_features = [
        category_feature(
            output_feature=True,
            name="output",
            preprocessing={
                "fallback_label": "3",
            },
            decoder={
                "type": "category_extractor",
                "match": {
                    "1": {"type": "contains", "value": "1"},
                    "2": {"type": "contains", "value": "2"},
                    "3": {"type": "contains", "value": "3"},
                    "4": {"type": "contains", "value": "4"},
                    "5": {"type": "contains", "value": "5"},
                },
            },
        )
    ]

    config = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: TEST_MODEL_NAME,
        GENERATION: get_generation_config(),
        PROMPT: {
            "retrieval": {"type": "random", "k": 3},
            "task": (
                "Given the sample input, complete this sentence by replacing XXXX: The review rating is XXXX. "
                "Choose one value in this list: [1, 2, 3, 4, 5]."
            ),
        },
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        PREPROCESSING: {
            "split": {TYPE: "fixed"},
        },
        BACKEND: {**backend, "cache_dir": str(tmpdir)},
    }

    dataset_path = generate_data(
        input_features,
        output_features,
        filename=csv_filename,
        num_examples=25,
        nan_percent=0.1,
        with_split=True,
    )
    df = pd.read_csv(dataset_path)
    df["output"] = np.random.choice([1, 2, 3, 4, 5], size=len(df)).astype(str)  # ensure labels match the feature config
    df.to_csv(dataset_path, index=False)

    model = LudwigModel(config)
    model.train(dataset=dataset_path, output_directory=str(tmpdir), skip_save_processed_input=True)

    # TODO: fix LLM model loading
    # model = LudwigModel.load(os.path.join(results.output_directory, "model"), backend=backend)
    preds, _ = model.predict(dataset=dataset_path)
    preds = convert_preds(preds)

    assert preds


def _prepare_finetuning_test(
    csv_filename: str, finetune_strategy: str, backend: dict, adapter_args: dict
) -> tuple[dict, str]:
    input_features = [text_feature(name="input", encoder={"type": "passthrough"})]
    output_features = [text_feature(name="output")]

    train_df = generate_data(input_features, output_features, filename=csv_filename, num_examples=25)
    prediction_df = pd.DataFrame(
        [
            {"input": "The food was amazing!", "output": "positive"},
            {"input": "The service was terrible.", "output": "negative"},
            {"input": "The food was okay.", "output": "neutral"},
        ]
    )

    model_name = TEST_MODEL_NAME
    if finetune_strategy == "adalora":
        # Adalora isn't supported for GPT-J model types, so use tiny bart
        model_name = "hf-internal-testing/tiny-random-BartModel"
    elif finetune_strategy == "adaption_prompt":
        # At the time of writing this test, Adaption Prompt fine-tuning is only supported for Llama models
        model_name = "yujiepan/llama-2-tiny-random"

    config = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: model_name,
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        GENERATION: {"max_new_tokens": 64},
        TRAINER: {
            TYPE: "finetune",
            BATCH_SIZE: "auto",
            EVAL_BATCH_SIZE: "auto",
            EPOCHS: 2,
        },
        BACKEND: backend,
    }

    if finetune_strategy is not None:
        config[ADAPTER] = {
            TYPE: finetune_strategy,
            **adapter_args,
        }

    return train_df, prediction_df, config


def _finetune_strategy_requires_cuda(finetune_strategy_name: str, quantization_args: dict | None) -> bool:
    """This method returns whether a given finetine_strategy requires CUDA.

    For all finetune strategies, except "qlora", the decision is based just on the name of the finetine_strategy; in the
    case of qlora, if the quantization dictionary is non-empty (i.e., contains quantization specifications), then the
    original finetine_strategy name of "lora" is interpreted as "qlora" and used in the lookup, based on the list of
    finetine strategies requiring CUDA.
    """
    cuda_only_finetune_strategy_names: list[str] = [
        "prompt_tuning",
        "prefix_tuning",
        "p_tuning",
        "qlora",
    ]

    if finetune_strategy_name == "lora" and quantization_args:
        finetune_strategy_name = "qlora"

    return finetune_strategy_name in cuda_only_finetune_strategy_names


def _verify_lm_lora_finetuning_layers(
    attention_layer: torch.nn.Module,
    target_modules: set[str],
    merge_adapter_into_base_model: bool,
    model_weights_directory: str,
    expected_lora_in_features: int,
    expected_lora_out_features: int,
    expected_file_names: list[str],
) -> None:
    """This method verifies that LoRA finetuning layers have correct types and shapes, depending on whether the
    optional "model.merge_and_unload()" method (based on the "merge_adapter_into_base_model" directive) was
    executed.

    If merge_adapter_into_base_model is True, then all specified LoRA projection layers in the attention layer must
    contain square weight matrices (with the dimensions expected_lora_in_features by expected_lora_in_features).
    However, if merge_adapter_into_base_model is False, then the LoRA part of the attention layer must include Lora_A
    and Lora_B children layers for each specified projection, such that the product of Lora_A and Lora_B is a square
    matrix (with the dimensions expected_lora_in_features by expected_lora_in_features) for each specified projection.
    """
    from peft.tuners.lora.layer import LoraLayer

    expected_lora_num_features_orig: tuple[int] = (expected_lora_in_features, expected_lora_out_features)

    file_names: list[str] = list_file_names_in_directory(directory_name=model_weights_directory)
    assert set(file_names) == set(expected_file_names)

    target_module_name: str
    target_module_obj: LoraLayer | torch.nn.Linear

    # Not providing default value to "getattr()" so that error is raised if incorrect projection layer name is supplied.

    for target_module_name in target_modules:
        target_module_obj = getattr(attention_layer, target_module_name)
        if merge_adapter_into_base_model:
            assert isinstance(target_module_obj, torch.nn.Linear)
        else:
            assert isinstance(target_module_obj, LoraLayer)

    if merge_adapter_into_base_model:
        # If LoRA A & B layers are merged, they must have no children layers, and projection matrices must be square.
        for target_module_name in target_modules:
            target_module_obj = getattr(attention_layer, target_module_name)
            assert not list(target_module_obj.children())
            assert (target_module_obj.in_features, target_module_obj.out_features) == (
                expected_lora_in_features,
                expected_lora_out_features,
            )
    else:
        # If LoRA A & B layers are not merged, their children layers must be correctly-dimensioned projection matrices.
        expected_lora_num_features: tuple[int]
        target_named_children: dict[str, torch.nn.Module]
        lora_matrix_name: str
        idx: int
        for target_module_name in target_modules:
            target_module_obj = getattr(attention_layer, target_module_name)
            target_named_children = dict(target_module_obj.named_children())

            for idx, lora_matrix_name in enumerate(["lora_A", "lora_B"]):
                assert isinstance(target_named_children[lora_matrix_name]["default"], torch.nn.Linear)

                # LoRA A and B matrix dimensions are transposes of one another so that their product is square matrix.
                expected_lora_num_features = (
                    expected_lora_num_features_orig
                    if idx % 2 == 0
                    else (expected_lora_num_features_orig[1], expected_lora_num_features_orig[0])
                )
                assert (
                    target_named_children[lora_matrix_name]["default"].in_features,
                    target_named_children[lora_matrix_name]["default"].out_features,
                ) == expected_lora_num_features


# TODO(arnav): p-tuning and prefix tuning have errors when enabled that seem to stem from DDP:
#
# prefix tuning:
# Sizes of tensors must match except in dimension 1. Expected size 320 but got size 32 for tensor number 1 in the list.
#
# p-tuning:
# 'PromptEncoder' object has no attribute 'mlp_head'
@pytest.mark.llm
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(LOCAL_BACKEND, id="local"),
        # TODO(Arnav): Re-enable once we can run tests on GPUs
        # This is because fine-tuning requires Ray with the deepspeed strategy, and deepspeed
        # only works with GPUs
        # pytest.param(RAY_BACKEND, id="ray"),
    ],
)
@pytest.mark.parametrize(
    "finetune_strategy,adapter_args",
    [
        pytest.param(
            None,
            {},
            id="full",
        ),
        pytest.param(
            "lora",
            {},
            id="lora-defaults",
        ),
        pytest.param(
            "lora",
            {"r": 4, "dropout": 0.1},
            id="lora-modified-defaults",
        ),
        pytest.param(
            "lora",
            {TARGET_MODULES: ["q_proj", "k_proj", "v_proj"]},
            id="lora-target-modules",
        ),
        pytest.param(
            "lora",
            {"use_rslora": True},
            id="lora-rslora-enabled",
        ),
        pytest.param(
            "lora",
            {"use_dora": True},
            id="lora-dora-enabled",
        ),
        pytest.param(
            "lora",
            {"use_rslora": True, "use_dora": True},
            id="lora-rslora-and-dora-enabled",
        ),
        pytest.param(
            "lora",
            {POSTPROCESSOR: {MERGE_ADAPTER_INTO_BASE_MODEL: True, PROGRESSBAR: True}},
            id="lora_merged",
        ),
        pytest.param(
            "lora",
            {POSTPROCESSOR: {MERGE_ADAPTER_INTO_BASE_MODEL: False}},
            id="lora_not_merged",
        ),
        pytest.param(
            "adalora",
            {},
            id="adalora-defaults",
        ),
        pytest.param(
            "adalora",
            {"init_r": 8, "beta1": 0.8},
            id="adalora-modified-defaults",
        ),
        pytest.param(
            "adalora",
            {POSTPROCESSOR: {MERGE_ADAPTER_INTO_BASE_MODEL: True, PROGRESSBAR: True}},
            id="adalora_merged",
        ),
        pytest.param(
            "adalora",
            {POSTPROCESSOR: {MERGE_ADAPTER_INTO_BASE_MODEL: False}},
            id="adalora_not_merged",
        ),
        # TODO: <Alex>02/21/2024: Disabling AdaptionPrompt (waiting for PEFT release to fix
        # "TypeError: LlamaRotaryEmbedding.forward() missing 1 required positional argument: 'position_ids')"
        # (this is reflected in https://github.com/ludwig-ai/ludwig/issues/3938).
        # </Alex>
        # pytest.param(
        #     "adaption_prompt",
        #     {},
        #     id="adaption_prompt-defaults",
        # ),
        # pytest.param(
        #     "adaption_prompt",
        #     {"adapter_len": 6, "adapter_layers": 1},
        #     id="adaption_prompt-modified-defaults",
        # ),
        pytest.param(
            "ia3",
            {},
            id="ia3-defaults",
        ),
        pytest.param(
            "ia3",
            {"init_ia3_weights": False},
            id="ia3-modified-defaults",
        ),
        # pytest.param(
        #     "prompt_tuning",
        #     {
        #         "num_virtual_tokens": 8,
        #         "prompt_tuning_init": "RANDOM",
        #     },
        #     id="prompt_tuning_init_random",
        # ),
        # pytest.param(
        #     "prompt_tuning",
        #     {
        #         "num_virtual_tokens": 8,
        #         "prompt_tuning_init": "TEXT",
        #         "prompt_tuning_init_text": "Classify if the review is positive, negative, or neutral: ",
        #     },
        #     id="prompt_tuning_init_text",
        # ),
        # pytest.param(
        #     "prefix_tuning",
        #     {
        #         "num_virtual_tokens": 8,
        #     },
        #     id="prefix_tuning",
        # ),
        # pytest.param(
        #     "p_tuning",
        #     {"num_virtual_tokens": 8, "encoder_reparameterization_type": "MLP"},
        #     id="p_tuning_mlp_reparameterization",
        # ),
        # pytest.param(
        #     "p_tuning",
        #     {"num_virtual_tokens": 8, "encoder_reparameterization_type": "LSTM"},
        #     id="p_tuning_lstm_reparameterization",
        # ),
    ],
)
def test_llm_finetuning_strategies(tmpdir, csv_filename, backend, finetune_strategy, adapter_args):
    train_df, prediction_df, config = _prepare_finetuning_test(csv_filename, finetune_strategy, backend, adapter_args)

    output_directory: str = str(tmpdir)
    model_directory: str = pathlib.Path(output_directory) / "api_experiment_run" / MODEL_FILE_NAME

    model = LudwigModel(config)
    model.train(dataset=train_df, output_directory=output_directory, skip_save_processed_input=False)

    # Make sure we can load the saved model and then use it for predictions
    model = LudwigModel.load(str(model_directory), backend=backend)

    base_model = LLM(ModelConfig.from_dict(config))
    assert not _compare_models(base_model, model.model)  # noqa F821

    preds, _ = model.predict(dataset=prediction_df, output_directory=output_directory)
    preds = convert_preds(preds)

    assert preds


@pytest.mark.llm
@pytest.mark.parametrize(
    "finetune_strategy,adapter_args,quantization",
    [
        pytest.param(
            "lora",
            {},
            {"bits": 4},
            id="qlora-4bit",
        ),
        pytest.param(
            "lora",
            {},
            {"bits": 8},
            id="qlora-8bit",
        ),
    ],
)
def test_llm_finetuning_strategies_quantized(tmpdir, csv_filename, finetune_strategy, adapter_args, quantization):
    pytest.importorskip("bitsandbytes", reason="bitsandbytes required for quantization tests")
    if (
        _finetune_strategy_requires_cuda(finetune_strategy_name=finetune_strategy, quantization_args=quantization)
        and not (torch.cuda.is_available() and torch.cuda.device_count()) > 0
    ):
        pytest.skip("Skip: quantization requires GPU and none are available.")

    backend = LOCAL_BACKEND

    train_df, prediction_df, config = _prepare_finetuning_test(csv_filename, finetune_strategy, backend, adapter_args)
    config["backend"] = backend
    config[QUANTIZATION] = quantization

    model = LudwigModel(config)
    model.train(dataset=train_df, output_directory=str(tmpdir), skip_save_processed_input=False)

    # Make sure we can load the saved model and then use it for predictions
    model = LudwigModel.load(os.path.join(str(tmpdir), "api_experiment_run", MODEL_FILE_NAME))

    base_model = LLM(ModelConfig.from_dict(config))
    assert not _compare_models(base_model, model.model)  # noqa F821

    preds, _ = model.predict(dataset=prediction_df, output_directory=str(tmpdir))
    preds = convert_preds(preds)

    assert preds


@pytest.mark.llm
@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="test requires at least 1 gpu")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires gpu support")
@pytest.mark.parametrize(
    "finetune_strategy,adapter_args,quantization,error_raised",
    [
        pytest.param(
            "lora",
            {POSTPROCESSOR: {MERGE_ADAPTER_INTO_BASE_MODEL: False}},
            {"bits": 4},
            (
                ImportError,
                "Using `load_in_8bit=True` requires Accelerate: `pip install accelerate` and the latest version of bitsandbytes `pip install -i https://test.pypi.org/simple/ bitsandbytes` or pip install bitsandbytes` ",  # noqa: E501
            ),
            id="qlora-4bit-not-merged",
        ),
        pytest.param(
            "lora",
            {POSTPROCESSOR: {MERGE_ADAPTER_INTO_BASE_MODEL: True, PROGRESSBAR: True}},
            {"bits": 8},
            (
                ImportError,
                "Using `load_in_8bit=True` requires Accelerate: `pip install accelerate` and the latest version of bitsandbytes `pip install -i https://test.pypi.org/simple/ bitsandbytes` or pip install bitsandbytes` ",  # noqa: E501
            ),
            id="qlora-8bit-merged",
        ),
        pytest.param(
            "lora",
            {POSTPROCESSOR: {MERGE_ADAPTER_INTO_BASE_MODEL: False}},
            {"bits": 8},
            (
                ImportError,
                "Using `load_in_8bit=True` requires Accelerate: `pip install accelerate` and the latest version of bitsandbytes `pip install -i https://test.pypi.org/simple/ bitsandbytes` or pip install bitsandbytes` ",  # noqa E501
            ),
            id="qlora-8bit-not-merged",
        ),
    ],
)
def test_llm_lora_finetuning_merge_and_unload_quantized_accelerate_required(
    csv_filename, finetune_strategy, adapter_args, quantization, error_raised
):
    pytest.importorskip("bitsandbytes", reason="bitsandbytes required for quantization tests")
    input_features: list[dict] = [text_feature(name="input", encoder={"type": "passthrough"})]
    output_features: list[dict] = [text_feature(name="output")]

    config: dict = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: TEST_MODEL_NAME,
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        TRAINER: {
            TYPE: "finetune",
            BATCH_SIZE: 8,
            EPOCHS: 2,
        },
        ADAPTER: {
            TYPE: finetune_strategy,
            **adapter_args,
        },
        QUANTIZATION: quantization,
    }

    model = LudwigModel(config)

    error_class: type  # noqa [F842]  # incorrect flagging of "local variable is annotated but never used
    error_message: str  # noqa [F842]  # incorrect flagging of "local variable is annotated but never used
    error_class, error_message = error_raised
    with pytest.raises(error_class) as excinfo:
        train_df = generate_data(input_features, output_features, filename=csv_filename, num_examples=3)
        model.train(dataset=train_df)

    assert str(excinfo.value) == error_message


@pytest.mark.llm
def test_llm_lora_finetuning_merge_and_unload_4_bit_quantization_not_supported(local_backend: dict):
    input_features: list[dict] = [text_feature(name="input", encoder={"type": "passthrough"})]
    output_features: list[dict] = [text_feature(name="output")]
    finetune_strategy: str = "lora"

    config: dict = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: TEST_MODEL_NAME,
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        TRAINER: {
            TYPE: "finetune",
            BATCH_SIZE: 8,
            EPOCHS: 2,
        },
        ADAPTER: {
            TYPE: finetune_strategy,
            POSTPROCESSOR: {MERGE_ADAPTER_INTO_BASE_MODEL: True, PROGRESSBAR: True},
        },
        QUANTIZATION: {"bits": 4},
        BACKEND: local_backend,
    }

    expected_error_class: type = ludwig_error.ConfigValidationError
    expected_error_message: str = """This operation will entail merging LoRA layers on a 4-bit quantized model.  \
Calling "save_pretrained()" on that model is currently unsupported.  If you want to merge the LoRA adapter weights \
into the base model, you need to use 8-bit quantization or do non-quantized based training by removing the \
quantization section from your Ludwig configuration."""
    with pytest.raises(expected_error_class) as excinfo:
        _ = LudwigModel(config)

    assert str(excinfo.value) == expected_error_message


@pytest.mark.llm
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(LOCAL_BACKEND, id="local"),
        # TODO: Re-enable once we can run tests on GPUs
        # This is because fine-tuning requires Ray with the deepspeed strategy, and deepspeed
        # only works with GPUs
        # pytest.param(RAY_BACKEND, id="ray"),
    ],
)
@pytest.mark.parametrize(
    "target_modules,merge_adapter_into_base_model,expected_lora_in_features,expected_lora_out_features,expected_file_names",  # noqa: E501
    [
        pytest.param(
            None,
            False,
            32,
            8,
            [
                "README.md",
                "adapter_config.json",
                "adapter_model.safetensors",
            ],
            id="lora_default_not_merged",
        ),
        pytest.param(
            None,
            True,
            32,
            32,
            [
                "README.md",
                "adapter_config.json",
                "adapter_model.safetensors",
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
            id="lora_default_merged",
        ),
        pytest.param(
            ["q_proj", "k_proj", "v_proj"],
            False,
            32,
            8,
            [
                "README.md",
                "adapter_config.json",
                "adapter_model.safetensors",
            ],
            id="lora_custom_not_merged",
        ),
        pytest.param(
            ["q_proj", "k_proj", "v_proj"],
            True,
            32,
            32,
            [
                "README.md",
                "adapter_config.json",
                "adapter_model.safetensors",
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
            id="lora_custom_merged",
        ),
    ],
)
def test_llm_lora_finetuning_merge_and_unload(
    tmpdir: str,
    csv_filename: str,
    backend: dict,
    target_modules: list[str] | set[str] | None,
    merge_adapter_into_base_model: bool,
    expected_lora_in_features: int,
    expected_lora_out_features: int,
    expected_file_names: list[str],
):
    from peft.tuners.lora.config import LoraConfig
    from peft.tuners.lora.model import LoraModel

    finetune_strategy: str = "lora"

    adapter_args: dict = {
        POSTPROCESSOR: {
            MERGE_ADAPTER_INTO_BASE_MODEL: merge_adapter_into_base_model,
        },
    }
    # If "target_modules" is None, then ["q_proj", "v_proj"] is used (HuggingFace Transformers/PEFT internal default).
    if target_modules:
        adapter_args[TARGET_MODULES] = target_modules

    train_df, prediction_df, config = _prepare_finetuning_test(
        csv_filename=csv_filename, finetune_strategy=finetune_strategy, backend=backend, adapter_args=adapter_args
    )

    output_directory: str = str(tmpdir)
    model_directory: str = pathlib.Path(output_directory) / "api_experiment_run" / MODEL_FILE_NAME
    model_weights_directory: str = (
        pathlib.Path(output_directory) / "api_experiment_run" / MODEL_FILE_NAME / MODEL_WEIGHTS_FILE_NAME
    )

    model = LudwigModel(config)
    model.train(dataset=train_df, output_directory=output_directory, skip_save_processed_input=False)

    # Get actual "target_modules" from trained model (to be used in assertions).
    lora_model: LoraModel = model.model.model.base_model
    peft_config: dict = lora_model.peft_config
    lora_config: LoraConfig = peft_config["default"]
    target_modules = lora_config.target_modules

    _verify_lm_lora_finetuning_layers(
        attention_layer=model.model.model.base_model.model.transformer.h[1].attn,
        target_modules=target_modules,
        merge_adapter_into_base_model=merge_adapter_into_base_model,
        model_weights_directory=model_weights_directory,
        expected_lora_in_features=expected_lora_in_features,
        expected_lora_out_features=expected_lora_out_features,
        expected_file_names=expected_file_names,
    )

    # Make sure we can load the saved model and verify that the LoRA layers have expected shapes.
    model = LudwigModel.load(str(model_directory), backend=backend)
    _verify_lm_lora_finetuning_layers(
        attention_layer=model.model.model.base_model.model.transformer.h[1].attn,
        target_modules=target_modules,
        merge_adapter_into_base_model=merge_adapter_into_base_model,
        model_weights_directory=model_weights_directory,
        expected_lora_in_features=expected_lora_in_features,
        expected_lora_out_features=expected_lora_out_features,
        expected_file_names=expected_file_names,
    )


@pytest.mark.llm
@pytest.mark.parametrize("use_adapter", [True, False], ids=["with_adapter", "without_adapter"])
def test_llm_training_with_gradient_checkpointing(tmpdir, csv_filename, use_adapter):
    input_features = [text_feature(name="input", encoder={"type": "passthrough"})]
    output_features = [text_feature(name="output")]

    df = generate_data(input_features, output_features, filename=csv_filename, num_examples=25)

    config = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: "hf-internal-testing/tiny-random-BartModel",
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        TRAINER: {
            TYPE: "finetune",
            BATCH_SIZE: 8,
            EPOCHS: 1,
            "enable_gradient_checkpointing": True,
        },
    }

    if use_adapter:
        config[ADAPTER] = {TYPE: "lora"}

    model = LudwigModel(config)
    assert model.config_obj.trainer.enable_gradient_checkpointing

    model.train(dataset=df, output_directory=str(tmpdir), skip_save_processed_input=False)


@pytest.mark.llm
def test_lora_wrap_on_init():
    from peft import PeftModel
    from transformers import PreTrainedModel

    config = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: TEST_MODEL_NAME,
        INPUT_FEATURES: [text_feature(name="input", encoder={"type": "passthrough"})],
        OUTPUT_FEATURES: [text_feature(name="output")],
        TRAINER: {
            TYPE: "finetune",
            BATCH_SIZE: 8,
            EPOCHS: 2,
        },
    }
    config_obj = ModelConfig.from_dict(config)
    model = LLM(config_obj)
    assert isinstance(model.model, PreTrainedModel)
    assert not isinstance(model.model, PeftModel)

    # Now add adapter
    config[ADAPTER] = {
        TYPE: "lora",
    }
    config_obj = ModelConfig.from_dict(config)
    model = LLM(config_obj)
    # We need to explicitly make this call since we now load the adapter
    # in the trainer as opposed to the point of LLM model initialization.
    model.prepare_for_training()
    assert not isinstance(model.model, PreTrainedModel)
    assert isinstance(model.model, PeftModel)


def test_llama_rope_scaling():
    config = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: "HuggingFaceH4/tiny-random-LlamaForCausalLM",
        INPUT_FEATURES: [text_feature(name="input", encoder={"type": "passthrough"})],
        OUTPUT_FEATURES: [text_feature(name="output")],
        TRAINER: {
            TYPE: "finetune",
            BATCH_SIZE: 8,
            EPOCHS: 2,
        },
        "model_parameters": {
            "rope_scaling": {
                "rope_type": "dynamic",
                "factor": 2.0,
            }
        },
    }
    config_obj = ModelConfig.from_dict(config)
    model = LLM(config_obj)

    assert model.model.config.rope_scaling
    assert model.model.config.rope_scaling["rope_type"] == "dynamic"
    assert model.model.config.rope_scaling["factor"] == 2.0


def test_default_max_sequence_length():
    config = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: TEST_MODEL_NAME,
        INPUT_FEATURES: [text_feature(name="input", encoder={"type": "passthrough"})],
        OUTPUT_FEATURES: [text_feature(name="output")],
        TRAINER: {
            TYPE: "finetune",
            BATCH_SIZE: 8,
            EPOCHS: 2,
        },
        ADAPTER: {TYPE: "lora", PRETRAINED_ADAPTER_WEIGHTS: "Infernaught/test_adapter_weights"},
        BACKEND: {TYPE: "local"},
    }
    config_obj = ModelConfig.from_dict(config)
    assert config_obj.input_features[0].preprocessing.max_sequence_length is None
    assert config_obj.output_features[0].preprocessing.max_sequence_length is None


@pytest.mark.llm
@pytest.mark.parametrize(
    "adapter",
    [
        "lora",
        "adalora",
        # TODO: <Alex>02/21/2024: Disabling AdaptionPrompt (waiting for PEFT release to fix
        # "TypeError: LlamaRotaryEmbedding.forward() missing 1 required positional argument: 'position_ids')"
        # (this is reflected in https://github.com/ludwig-ai/ludwig/issues/3938).
        # </Alex>
        # "adaption_prompt",
    ],
)
def test_load_pretrained_adapter_weights(adapter):
    from peft import PeftModel
    from transformers import PreTrainedModel

    if adapter == "lora":
        weights = "Infernaught/test_adapter_weights"
        base_model = TEST_MODEL_NAME
    elif adapter == "adalora":
        weights = "Infernaught/test_adalora_weights"
        base_model = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
    elif adapter == "adaption_prompt":
        weights = "Infernaught/test_ap_weights"
        base_model = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
    else:
        raise ()

    config = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: base_model,
        INPUT_FEATURES: [text_feature(name="input", encoder={"type": "passthrough"})],
        OUTPUT_FEATURES: [text_feature(name="output")],
        TRAINER: {
            TYPE: "none",
            BATCH_SIZE: 8,
            EPOCHS: 2,
        },
        ADAPTER: {TYPE: adapter, PRETRAINED_ADAPTER_WEIGHTS: weights},
        BACKEND: {TYPE: "local"},
    }
    config_obj = ModelConfig.from_dict(config)
    model = LLM(config_obj)

    assert model.config_obj.adapter.pretrained_adapter_weights
    assert model.config_obj.adapter.pretrained_adapter_weights == weights

    model.prepare_for_training()
    assert not isinstance(model.model, PreTrainedModel)
    assert isinstance(model.model, PeftModel)

    config_obj = ModelConfig.from_dict(config)
    assert config_obj.input_features[0].preprocessing.max_sequence_length is None
    assert config_obj.output_features[0].preprocessing.max_sequence_length is None


def _compare_models(model_1: torch.nn.Module, model_2: torch.nn.Module) -> bool:
    # For a full explanation of this 8-bit workaround, see https://github.com/ludwig-ai/ludwig/pull/3606

    # TODO: Uncomment "filter_for_weight_format()" method definition and enable its usage once GPU tests are set up.
    # def filter_for_weight_format(i):
    #     """Remove bitsandbytes metadata keys added on state dict creation.
    #
    #     8-bit quantized models that have been put on gpu will have a set of `weight_format` keys in their state dict.
    #     These contain strings that are used to reshape quantized tensors, however these have no impact until the state
    #     dict is loaded into a model. These keys were causing `torch.equal` to raise an exception, so we skip them in
    #     the evaluation.
    #     """
    #     return "weight_format" not in i[0]

    # model_1_filtered_state_dict = filter(filter_for_weight_format, model_1.state_dict().items())
    # model_2_filtered_state_dict = filter(filter_for_weight_format, model_2.state_dict().items())

    # Source: https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/6

    if model_1.__class__.__name__ != model_2.__class__.__name__:
        return False

    if (
        hasattr(model_1, "model")
        and hasattr(model_2, "model")
        and not _compare_models(model_1=model_1.model, model_2=model_2.model)
    ):
        return False

    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if not torch.equal(key_item_1[1], key_item_2[1]):
            return False

    return True


def test_global_max_sequence_length_for_llms():
    """Ensures that user specified global_max_sequence_length can never be greater than the model's context
    length."""
    config = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: "HuggingFaceH4/tiny-random-LlamaForCausalLM",
        INPUT_FEATURES: [text_feature(name="input", encoder={"type": "passthrough"})],
        OUTPUT_FEATURES: [text_feature(name="output")],
    }
    config_obj = ModelConfig.from_dict(config)
    model = LLM(config_obj)

    # Default value is set based on model's context_len
    assert model.global_max_sequence_length == 2048

    # Override to a larger value in the config
    config["preprocessing"] = {"global_max_sequence_length": 4096}
    config_obj = ModelConfig.from_dict(config)
    model = LLM(config_obj)

    # Check that the value can never be larger than the model's context_len
    assert model.global_max_sequence_length == 2048


def test_local_path_loading(tmpdir):
    """Tests that local paths can be used to load models."""

    from huggingface_hub import snapshot_download

    # Download the model to a local directory
    local_path: str = f"{str(tmpdir)}/test_local_path_loading"
    repo_id: str = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
    os.makedirs(local_path, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=local_path)

    # Load the model using the local path
    config1 = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: local_path,
        INPUT_FEATURES: [text_feature(name="input", encoder={"type": "passthrough"})],
        OUTPUT_FEATURES: [text_feature(name="output")],
    }
    config_obj1 = ModelConfig.from_dict(config1)
    model1 = LLM(config_obj1)

    # Load the model using the repo id
    config2 = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: repo_id,
        INPUT_FEATURES: [text_feature(name="input", encoder={"type": "passthrough"})],
        OUTPUT_FEATURES: [text_feature(name="output")],
    }
    config_obj2 = ModelConfig.from_dict(config2)
    model2 = LLM(config_obj2)

    # Check that the models are the same
    assert _compare_models(model1.model, model2.model)


@pytest.mark.parametrize(
    "finetuning_strategy, embedding_noise",
    [
        pytest.param(None, 0, id="full_finetuning_without_noise"),
        pytest.param(None, 5, id="full_finetuning_with_noise"),
        pytest.param("lora", 0, id="lora_without_noise"),
        pytest.param("lora", 5, id="lora_with_noise"),
    ],
)
def test_llm_finetuning_with_embedding_noise(
    tmpdir,
    csv_filename,
    finetuning_strategy,
    embedding_noise,
):
    train_df, prediction_df, config = _prepare_finetuning_test(csv_filename, finetuning_strategy, LOCAL_BACKEND, {})

    # Add embedding noise
    if embedding_noise:
        config["model_parameters"] = {"neftune_noise_alpha": embedding_noise}

    model = LudwigModel(config)

    if embedding_noise:
        assert model.config_obj.model_parameters.neftune_noise_alpha == embedding_noise

    output_directory: str = str(tmpdir)
    model_directory: str = pathlib.Path(output_directory) / "api_experiment_run" / MODEL_FILE_NAME
    model.train(dataset=train_df, output_directory=output_directory, skip_save_processed_input=False)

    # Make sure we can load the saved model and then use it for predictions
    model = LudwigModel.load(str(model_directory), backend=LOCAL_BACKEND)

    base_model = LLM(ModelConfig.from_dict(config))
    assert not _compare_models(base_model, model.model)  # noqa F821

    preds, _ = model.predict(dataset=prediction_df, output_directory=output_directory)
    preds = convert_preds(preds)

    assert preds


@pytest.fixture()
def llm_encoder_config() -> dict[str, Any]:
    encoder_config = {
        TYPE: "llm",
        BASE_MODEL: "HuggingFaceH4/tiny-random-LlamaForCausalLM",
    }

    return encoder_config


@pytest.mark.parametrize(
    "adapter,quantization",
    [
        (None, None),
        ("lora", None),
        ("lora", {"bits": 4}),
        ("lora", {"bits": 8}),
        ("adalora", None),
        ("adalora", {"bits": 4}),
        ("adalora", {"bits": 8}),
    ],
    ids=["FFT", "LoRA", "LoRA 4-bit", "LoRA 8-bit", "AdaLoRA", "AdaLoRA 4-bit", "AdaLoRA 8-bit"],
)
def test_llm_encoding(llm_encoder_config, adapter, quantization, tmpdir):
    if quantization:
        pytest.importorskip("bitsandbytes", reason="bitsandbytes required for quantization tests")
    if (
        _finetune_strategy_requires_cuda(
            finetune_strategy_name="lora" if adapter else None, quantization_args=quantization
        )
        and not (torch.cuda.is_available() and torch.cuda.device_count()) > 0
    ):
        pytest.skip("Skip: quantization requires GPU and none are available.")

    dataset_path = os.path.join(tmpdir, "llm_classification_data.csv")

    config = {
        MODEL_TYPE: MODEL_ECD,
        OUTPUT_FEATURES: [category_feature(name="output")],
        COMBINER: {TYPE: "sequence"},
        TRAINER: {EPOCHS: 1},
    }

    encoder_config = copy.deepcopy(llm_encoder_config)

    if adapter:
        encoder_config[ADAPTER] = {TYPE: adapter}
    if quantization:
        encoder_config[QUANTIZATION] = quantization
        config[BACKEND] = LOCAL_BACKEND

    config[INPUT_FEATURES] = [text_feature(name="input", encoder=encoder_config)]

    generate_data(input_features=config[INPUT_FEATURES], output_features=config[OUTPUT_FEATURES], filename=dataset_path)

    model = LudwigModel(config)
    model.train(dataset=dataset_path, output_directory=str(tmpdir))


def test_llm_batch_size_tuning():
    dataset = pd.DataFrame({"instruction": ["a"] * 100, "output": ["a"] * 100})
    config = yaml.safe_load(
        """
    model_type: llm
    input_features:
        - name: instruction
          type: text
    output_features:
        - name: output
          type: text
    prompt:
        template: >-
            {instruction}
    adapter:
        type: lora
    trainer:
        type: finetune
        optimizer:
            type: adam
        batch_size: auto
        train_steps: 1
        learning_rate: 0.0002
        eval_batch_size: 2
    backend:
        type: local
    base_model: HuggingFaceH4/tiny-random-LlamaForCausalLM
        """
    )
    model = LudwigModel(config=config)
    model.train(dataset=dataset)
    assert model.config_obj.trainer.batch_size > 1


@pytest.mark.llm
def test_llm_used_tokens(tmpdir):
    input_features = [text_feature(name="input", encoder={"type": "passthrough"})]
    output_features = [text_feature(name="output")]

    df = pd.read_json("https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json").head(
        10
    )

    # df = generate_data(input_features, output_features, filename=csv_filename, num_examples=25)

    config = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: "hf-internal-testing/tiny-random-BartModel",
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        TRAINER: {
            TYPE: "finetune",
            BATCH_SIZE: 1,
            EPOCHS: 3,
            "enable_gradient_checkpointing": True,
        },
    }

    config[ADAPTER] = {TYPE: "lora"}

    model = LudwigModel(config)
    assert model.config_obj.trainer.enable_gradient_checkpointing

    model.train(dataset=df, output_directory=str(tmpdir), skip_save_processed_input=False)

    with open(
        os.path.join(str(tmpdir), "api_experiment_run", MODEL_FILE_NAME, "training_progress.json"), encoding="utf-8"
    ) as f:
        progress_tracker = json.load(f)

    assert progress_tracker["cumulative_step_token_usage"]["11"] == progress_tracker["total_tokens_used"] == 621
    assert progress_tracker["checkpoint_to_epoch"] == {"1": 1, "2": 1, "3": 2, "4": 2, "5": 3, "6": 3}
    assert progress_tracker["checkpoint_to_step"] == {"1": 4, "2": 4, "3": 8, "4": 8, "5": 12, "6": 12}
    assert progress_tracker["cumulative_checkpoint_token_usage"] == {
        "1": 207,
        "2": 207,
        "3": 414,
        "4": 414,
        "5": 621,
        "6": 621,
    }
    assert progress_tracker["incremental_checkpoint_token_usage"] == {
        "1": 207,
        "2": 0,
        "3": 207,
        "4": 0,
        "5": 207,
        "6": 0,
    }
