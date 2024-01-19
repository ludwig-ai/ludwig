import copy

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, PreTrainedModel

from ludwig.encoders.text_encoders import LLMEncoder
from ludwig.schema.encoders.text_encoders import LLMEncoderConfig
from ludwig.schema.llms.peft import BaseAdapterConfig, LoraConfig
from ludwig.utils.llm_utils import get_context_len

# Mapping of adapter types to test against and their respective config objects.
ADAPTER_CONFIG_MAP = {"lora": LoraConfig}


@pytest.fixture()
def encoder_config() -> LLMEncoderConfig:
    """Create a baseline LLMEncoderConfig.

    Returns:
        A baseline LLMEncoderConfig with a small model, no adapter, and no quantization
    """
    return LLMEncoderConfig(
        type="llm",
        max_sequence_length=256,
        base_model="HuggingFaceH4/tiny-random-LlamaForCausalLM",
        adapter=None,
        quantization=None,
    )


@pytest.fixture()
def model_config(encoder_config):
    return AutoConfig.from_pretrained(encoder_config.base_model)


class WrapperModule(nn.Module):
    def __init__(self, encoder: LLMEncoder):
        super().__init__()
        self.encoder = encoder


class TestLLMEncoder:
    def create_encoder_config_with_adapter(
        self, encoder_config: LLMEncoderConfig, adapter: str, **kwargs
    ) -> BaseAdapterConfig:
        """Create a config for the requested adapter.

        Args:
            adapter: name of the adapter

        Returns:
            A config object for the requested adapter. If any keyword args are passed, they will be used to initialize
            the config.
        """
        new_config = copy.deepcopy(encoder_config)
        new_config.adapter = ADAPTER_CONFIG_MAP[adapter](**kwargs)
        return new_config

    def test_init(self, encoder_config: LLMEncoderConfig, model_config):
        # Test initializing without an adapter
        encoder = LLMEncoder(encoder_config=encoder_config)

        assert encoder.model_name == encoder_config.base_model
        assert isinstance(encoder.model, PreTrainedModel)
        assert all(map(lambda k: "lora_" not in k, encoder.state_dict().keys()))  # Check adapter was not initialized
        assert encoder.input_shape == torch.Size([encoder_config.max_sequence_length])
        assert encoder.output_shape == torch.Size([encoder_config.max_sequence_length, model_config.hidden_size])

        # The final layer must not be trainable because it is not used
        last_module = list(encoder.model.modules())[-1]
        assert all(not p.requires_grad for p in last_module.parameters())

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

        # The final layer must not be trainable because it is not used
        last_module = list(encoder.model.modules())[-1]
        assert all(not p.requires_grad for p in last_module.parameters())

    @pytest.mark.parametrize("adapter", list(ADAPTER_CONFIG_MAP.keys()))
    def test_init_with_adapter(self, encoder_config: LLMEncoderConfig, adapter: str, model_config):
        from peft import PeftModel

        encoder_config_with_adapter = self.create_encoder_config_with_adapter(encoder_config, adapter)
        encoder = LLMEncoder(encoder_config=encoder_config_with_adapter)

        # The adapter should not be initialized until `prepare_for_training` is called
        assert not isinstance(encoder.model, PeftModel)
        assert not any(map(lambda k: "lora_" in k, encoder.state_dict().keys()))

        assert encoder.model_name == encoder_config.base_model
        assert encoder.input_shape == torch.Size([encoder_config.max_sequence_length])
        assert encoder.output_shape == torch.Size([encoder_config.max_sequence_length, model_config.hidden_size])

        # The final layer must not be trainable because it is not used
        last_module = list(encoder.model.modules())[-1]
        assert all(not p.requires_grad for p in last_module.parameters())

    @pytest.mark.parametrize("adapter", list(ADAPTER_CONFIG_MAP.keys()))
    def test_prepare_for_training(self, encoder_config: LLMEncoderConfig, adapter: str):
        from peft import PeftModel

        encoder_config_with_adapter = self.create_encoder_config_with_adapter(encoder_config, adapter)
        encoder = LLMEncoder(encoder_config=encoder_config_with_adapter)

        # The adapter should not be initialized until `prepare_for_training` is called
        assert not isinstance(encoder.model, PeftModel)
        assert not any(map(lambda k: "lora_" in k, encoder.state_dict().keys()))

        # Initialize the adapter
        encoder.prepare_for_training()

        # At this point, the adapter should be initialized and the state dict should contain adapter parameters
        assert isinstance(encoder.model, PeftModel)
        assert any(map(lambda k: "lora_" in k, encoder.state_dict().keys()))

    def test_save_to_state_dict(self, encoder_config: LLMEncoderConfig, tmpdir):
        # With no adapter, the state dict should only contain the model parameters
        encoder = LLMEncoder(encoder_config=encoder_config)
        assert all(map(lambda k: "lora_" not in k, encoder.state_dict().keys()))

    @pytest.mark.parametrize("adapter", list(ADAPTER_CONFIG_MAP.keys()))
    def test_save_to_state_dict_adapter(self, encoder_config: LLMEncoderConfig, adapter: str, tmpdir):
        # With an adapter, the state dict should only contain adapter parameters
        encoder_config_with_adapter = self.create_encoder_config_with_adapter(encoder_config, adapter)
        encoder = LLMEncoder(encoder_config=encoder_config_with_adapter)
        # Initialize the adapters
        encoder.prepare_for_training()
        assert all(map(lambda k: "lora_" in k, encoder.state_dict().keys()))

    @pytest.mark.parametrize("wrap", [False, True], ids=["no_wrapper", "with_wrapper"])
    def test_load_from_state_dict(self, encoder_config: LLMEncoderConfig, wrap: bool):
        def weights_init(m):
            """Reinitialize the weights of a torch module."""
            if hasattr(m, "weight") and m.weight.ndim > 1:
                torch.nn.init.xavier_uniform_(m.weight.data)

        # Create two encoders from the same config
        encoder1 = LLMEncoder(encoder_config=encoder_config)
        encoder2 = LLMEncoder(encoder_config=encoder_config)

        if wrap:
            encoder1 = WrapperModule(encoder1)
            encoder2 = WrapperModule(encoder2)

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

    @pytest.mark.parametrize("wrap", [False, True], ids=["no_wrapper", "with_wrapper"])
    @pytest.mark.parametrize("adapter", list(ADAPTER_CONFIG_MAP.keys()))
    def test_load_from_state_dict_adapter(self, encoder_config: LLMEncoderConfig, adapter: str, wrap: bool):
        def weights_init(m):
            """Reinitialize the weights of a torch module."""
            if hasattr(m, "weight") and m.weight.ndim > 1:
                torch.nn.init.xavier_uniform_(m.weight.data)

        # Update the config with an adapter
        encoder_config_with_adapter = self.create_encoder_config_with_adapter(encoder_config, adapter)

        # Create two encoders from the same config
        encoder1 = LLMEncoder(encoder_config=encoder_config_with_adapter)
        encoder2 = LLMEncoder(encoder_config=encoder_config_with_adapter)

        # Initialize the adapters
        encoder1.prepare_for_training()
        encoder2.prepare_for_training()

        if wrap:
            encoder1 = WrapperModule(encoder1)
            encoder2 = WrapperModule(encoder2)

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
