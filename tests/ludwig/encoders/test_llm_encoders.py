import copy

import pytest
import torch
from transformers import AutoConfig, PreTrainedModel

from ludwig.encoders.text_encoders import LLMEncoder
from ludwig.schema.encoders.text_encoders import LLMEncoderConfig
from ludwig.schema.llms.peft import LoraConfig
from ludwig.utils.llm_utils import get_context_len

# import os


@pytest.fixture()
def encoder_config() -> LLMEncoderConfig:
    return LLMEncoderConfig(
        type="llm",
        max_sequence_length=256,
        base_model="HuggingFaceH4/tiny-random-LlamaForCausalLM",
        adapter=None,
        quantization=None,
    )


@pytest.fixture()
def encoder_config_with_adapter(encoder_config) -> LLMEncoderConfig:
    config = copy.deepcopy(encoder_config)
    config.adapter = LoraConfig()
    return config


@pytest.fixture()
def model_config(encoder_config):
    return AutoConfig.from_pretrained(encoder_config.base_model)


class TestLLMEncoder:
    def test_init(self, encoder_config: LLMEncoderConfig, encoder_config_with_adapter: LLMEncoderConfig, model_config):
        # Test initializing without an adapter
        encoder = LLMEncoder(encoder_config=encoder_config)

        assert encoder.model_name == encoder_config.base_model
        assert isinstance(encoder.model, PreTrainedModel)
        assert all(map(lambda k: "lora_" not in k, encoder.state_dict().keys()))  # Check adapter was not initialized
        assert encoder.input_shape == torch.Size([encoder_config.max_sequence_length])
        assert encoder.output_shape == torch.Size([encoder_config.max_sequence_length, model_config.hidden_size])

        # Test initializing with an adapter
        from peft import PeftModel

        encoder = LLMEncoder(encoder_config=encoder_config_with_adapter)

        assert encoder.model_name == encoder_config.base_model
        assert isinstance(encoder.model, PeftModel)
        assert any(map(lambda k: "lora_" in k, encoder.state_dict().keys()))  # Check adapter was initialized
        assert encoder.input_shape == torch.Size([encoder_config.max_sequence_length])
        assert encoder.output_shape == torch.Size([encoder_config.max_sequence_length, model_config.hidden_size])

        # Test that max sequence length falls back to the context length when too large
        context_len = get_context_len(model_config)
        cl_config = copy.deepcopy(encoder_config)
        cl_config.max_sequence_length = context_len + 1

        encoder = LLMEncoder(encoder_config=cl_config)

        assert encoder.model_name == encoder_config.base_model
        assert isinstance(encoder.model, PreTrainedModel)
        assert all(map(lambda k: "lora_" not in k, encoder.state_dict().keys()))  # Check adapter was not initialized
        assert encoder.input_shape == torch.Size([context_len])
        assert encoder.output_shape == torch.Size([context_len, model_config.hidden_size])

    def test_save_to_state_dict(self, encoder_config: LLMEncoderConfig, tmpdir):
        # With no adapter, the state dict should only contain the model parameters
        encoder = LLMEncoder(encoder_config=encoder_config)
        assert all(map(lambda k: "lora_" not in k, encoder.state_dict().keys()))

    def test_save_to_state_dict_adapter(self, encoder_config_with_adapter: LLMEncoderConfig, tmpdir):
        # With an adapter, the state dict should only contain adapter parameters
        encoder = LLMEncoder(encoder_config=encoder_config_with_adapter)
        assert all(map(lambda k: "lora_" in k, encoder.state_dict().keys()))

    def test_load_from_state_dict(self, encoder_config: LLMEncoderConfig):
        def weights_init(m):
            """Reinitialize the weights of a torch module."""
            if hasattr(m, "weight") and m.weight.ndim > 1:
                torch.nn.init.xavier_uniform_(m.weight.data)

        # Create two encoders from the same config
        encoder1 = LLMEncoder(encoder_config=encoder_config)
        encoder2 = LLMEncoder(encoder_config=encoder_config)

        # Reinitialize the weights of one encoder so the two are not identical
        encoder2.apply(weights_init)

        # Ensure that the weights are different
        encoder1_sd = encoder1.state_dict()
        encoder2_sd = encoder2.state_dict()
        assert any(map(lambda k: not torch.equal(encoder1_sd[k], encoder2_sd[k]), encoder1_sd.keys()))

        # Load the weights of encoder1 back into encoder2 and ensure the weights are equal
        encoder2.load_state_dict(encoder1_sd)
        encoder2_sd = encoder2.state_dict()
        assert all(map(lambda k: torch.equal(encoder1_sd[k], encoder2_sd[k]), encoder1_sd.keys()))

    def test_load_from_state_dict_adapter(self, encoder_config_with_adapter: LLMEncoderConfig):
        def weights_init(m):
            """Reinitialize the weights of a torch module."""
            if hasattr(m, "weight") and m.weight.ndim > 1:
                torch.nn.init.xavier_uniform_(m.weight.data)

        # Create two encoders from the same config
        encoder1 = LLMEncoder(encoder_config=encoder_config_with_adapter)
        encoder2 = LLMEncoder(encoder_config=encoder_config_with_adapter)

        encoder2.apply(weights_init)

        encoder1_sd = encoder1.state_dict()
        encoder2_sd = encoder2.state_dict()
        adapter_keys = [k for k in encoder1_sd.keys() if "lora_" in k and "weight" in k]
        model_keys = [k for k in encoder1_sd.keys() if "lora_" not in k]

        # The LoRA weights should no longer be equal
        assert all(map(lambda k: not torch.equal(encoder1_sd[k], encoder2_sd[k]), adapter_keys))

        # The remaining weights should also no longer be equal
        assert all(map(lambda k: not torch.equal(encoder1_sd[k], encoder2_sd[k]), model_keys))

        # Load the weights of encoder1 back into encoder2
        encoder2.load_state_dict(encoder1_sd)
        encoder2_sd = encoder2.state_dict()

        # The LoRA weights should now be equal again
        assert all(map(lambda k: torch.equal(encoder1_sd[k], encoder2_sd[k]), adapter_keys))

        # The remaining weights should still be unequal
        assert all(map(lambda k: not torch.equal(encoder1_sd[k], encoder2_sd[k]), model_keys))
