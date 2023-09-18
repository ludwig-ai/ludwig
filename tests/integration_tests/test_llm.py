import os

import numpy as np
import pandas as pd
import pytest
import torch

from ludwig.api import LudwigModel
from ludwig.constants import (
    ADAPTER,
    BACKEND,
    BASE_MODEL,
    BATCH_SIZE,
    EPOCHS,
    GENERATION,
    INPUT_FEATURES,
    MODEL_LLM,
    MODEL_TYPE,
    OUTPUT_FEATURES,
    PREPROCESSING,
    PRETRAINED_ADAPTER_WEIGHTS,
    PROMPT,
    TRAINER,
    TYPE,
)
from ludwig.models.llm import LLM
from ludwig.schema.model_types.base import ModelConfig
from ludwig.utils.types import DataFrame
from tests.integration_tests.utils import category_feature, generate_data, text_feature

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
    "finetune_strategy,adapter_args,quantization",
    [
        (None, {}, None),
        # (
        #     "prompt_tuning",
        #     {
        #         "num_virtual_tokens": 8,
        #         "prompt_tuning_init": "RANDOM",
        #     },
        # ),
        # (
        #     "prompt_tuning",
        #     {
        #         "num_virtual_tokens": 8,
        #         "prompt_tuning_init": "TEXT",
        #         "prompt_tuning_init_text": "Classify if the review is positive, negative, or neutral: ",
        #     },
        # ),
        # ("prefix_tuning", {"num_virtual_tokens": 8}),
        # ("p_tuning", {"num_virtual_tokens": 8, "encoder_reparameterization_type": "MLP"}),
        # ("p_tuning", {"num_virtual_tokens": 8, "encoder_reparameterization_type": "LSTM"}),
        ("lora", {}, None),
        ("lora", {}, {"bits": 4}),  # qlora
        ("lora", {}, {"bits": 8}),  # qlora 8-bit
        # ("adalora", {}),
        ("adaption_prompt", {"adapter_len": 6, "adapter_layers": 1}, None),
    ],
    ids=[
        "none",
        # "prompt_tuning_init_random",
        # "prompt_tuning_init_text",
        # "prefix_tuning",
        # "p_tuning_mlp_reparameterization",
        # "p_tuning_lstm_reparameterization",
        "lora",
        "qlora",
        "qlora-8bit",
        # "adalora",
        "adaption_prompt",
    ],
)
def test_llm_finetuning_strategies(tmpdir, csv_filename, backend, finetune_strategy, adapter_args, quantization):
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        pytest.skip("Skip: quantization requires GPU and none are available.")

    input_features = [text_feature(name="input", encoder={"type": "passthrough"})]
    output_features = [text_feature(name="output")]

    df = generate_data(input_features, output_features, filename=csv_filename, num_examples=25)

    model_name = TEST_MODEL_NAME
    if finetune_strategy == "adalora":
        # Adalora isn't supported for GPT-J model types, so use tiny bart
        model_name = "hf-internal-testing/tiny-random-BartModel"
    elif finetune_strategy == "adaption_prompt":
        # At the time of writing this test, Adaption Prompt fine-tuning is only supported for Llama models
        model_name = "HuggingFaceM4/tiny-random-LlamaForCausalLM"

    config = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: model_name,
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        TRAINER: {
            TYPE: "finetune",
            BATCH_SIZE: 8,
            EPOCHS: 2,
        },
        BACKEND: backend,
    }

    if finetune_strategy is not None:
        config[ADAPTER] = {
            TYPE: finetune_strategy,
            **adapter_args,
        }

    if quantization is not None:
        config["quantization"] = quantization

    model = LudwigModel(config)
    model.train(dataset=df, output_directory=str(tmpdir), skip_save_processed_input=False)

    prediction_df = pd.DataFrame(
        [
            {"input": "The food was amazing!", "output": "positive"},
            {"input": "The service was terrible.", "output": "negative"},
            {"input": "The food was okay.", "output": "neutral"},
        ]
    )

    # Make sure we can load the saved model and then use it for predictions
    model = LudwigModel.load(os.path.join(str(tmpdir), "api_experiment_run", "model"), backend=backend)

    base_model = LLM(ModelConfig.from_dict(config))
    assert not _compare_models(base_model, model.model)

    preds, _ = model.predict(dataset=prediction_df, output_directory=str(tmpdir))
    preds = convert_preds(preds)

    assert preds


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
                "type": "dynamic",
                "factor": 2.0,
            }
        },
    }
    config_obj = ModelConfig.from_dict(config)
    model = LLM(config_obj)

    assert model.model.config.rope_scaling
    assert model.model.config.rope_scaling["type"] == "dynamic"
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


@pytest.mark.parametrize("adapter", ["lora", "adalora", "adaption_prompt"])
def test_load_pretrained_adapter_weights(adapter):
    from peft import PeftModel
    from transformers import PreTrainedModel

    weights = ""
    model = ""
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
    def filter_for_weight_format(i):
        """Remove bitsandbytes metadata keys added on state dict creation.

        8-bit quantized models that have been put on gpu will have a set of `weight_format` keys in their state dict.
        These contain strings that are used to reshape quantized tensors, however these have no impact until the state
        dict is loaded into a model. These keys were causing `torch.equal` to raise an exception, so we skip them in the
        evaluation.
        """
        return "weight_format" not in i[0]

    model_1_filtered_state_dict = filter(filter_for_weight_format, model_1.state_dict().items())
    model_2_filtered_state_dict = filter(filter_for_weight_format, model_2.state_dict().items())

    # Source: https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/6
    for key_item_1, key_item_2 in zip(model_1_filtered_state_dict, model_2_filtered_state_dict):
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


def test_local_path_loading():
    """Tests that local paths can be used to load models."""

    from huggingface_hub import snapshot_download

    # Download the model to a local directory
    LOCAL_PATH = "~/test_local_path_loading"
    REPO_ID = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
    os.makedirs(LOCAL_PATH, exist_ok=True)
    snapshot_download(repo_id=REPO_ID, local_dir=LOCAL_PATH)

    # Load the model using the local path
    config1 = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: LOCAL_PATH,
        INPUT_FEATURES: [text_feature(name="input", encoder={"type": "passthrough"})],
        OUTPUT_FEATURES: [text_feature(name="output")],
    }
    config_obj1 = ModelConfig.from_dict(config1)
    model1 = LLM(config_obj1)

    # Load the model using the repo id
    config2 = {
        MODEL_TYPE: MODEL_LLM,
        BASE_MODEL: REPO_ID,
        INPUT_FEATURES: [text_feature(name="input", encoder={"type": "passthrough"})],
        OUTPUT_FEATURES: [text_feature(name="output")],
    }
    config_obj2 = ModelConfig.from_dict(config2)
    model2 = LLM(config_obj2)

    # Check that the models are the same
    assert _compare_models(model1.model, model2.model)
